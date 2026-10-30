"""Physics checks: rate formula vs. a naive reference, constraint satisfaction by construction, and the
analytic gradient of the surrogate V (eq. 24) vs. central finite differences."""

import numpy as np
import pytest

from llm4rrm.beamforming.channels import generate_channels, normalize_channels, perturb_csi
from llm4rrm.beamforming.physics import STARRISSystem, Solution
from llm4rrm.config import SystemConfig


def _setup(B=3, N=8, M=4, U=4, seed=0, r_min_mbps=3.0, p_max=1.0):
    cfg = SystemConfig(M=M, U=U, N=N, r_min_mbps=r_min_mbps, p_max_w=p_max)
    G, H = generate_channels(cfg, B, seed)
    return cfg, STARRISSystem(G, H, cfg), Solution.random(cfg, B, np.random.default_rng(seed + 1))


def _naive_rates(cfg, G, H, sol, b):
    """Direct per-user loop transcription of paper eqs. (15)-(17)."""
    Wh = sol.W_hat[b]
    W = Wh / np.linalg.norm(Wh) * (1 / (1 + np.exp(-sol.alpha[b]))) * np.sqrt(cfg.p_max_w)
    sa = 1 / (1 + np.exp(-sol.a[b]))
    bt, br = np.sqrt(sa), np.sqrt(1 - sa)
    Tt = np.diag(bt * np.exp(1j * sol.phi_t[b]))
    Tr = np.diag(br * np.exp(1j * sol.phi_r[b]))
    out = []
    for u in range(cfg.U):
        Th = Tt if u < cfg.n_transmit_users else Tr
        row = H[b][:, u].conj() @ Th @ G[b]  # (M,)
        sig = abs(row @ W[:, u]) ** 2
        itf = sum(abs(row @ W[:, i]) ** 2 for i in range(cfg.U) if i != u)
        out.append(np.log2(1 + sig / (itf + cfg.noise_power_w)))
    return np.array(out)


def test_rates_match_naive_loop():
    cfg, sys_, sol = _setup()
    R = sys_.rates_bps_hz(sol)
    for b in range(sys_.B):
        np.testing.assert_allclose(R[b], _naive_rates(cfg, sys_.G, sys_.H, sol, b), rtol=1e-9)


def test_constraints_hold_by_construction():
    cfg, sys_, sol = _setup(p_max=1.5)
    tr = sys_.transformed(sol)
    assert np.all(tr["power"] <= cfg.p_max_w + 1e-12)
    np.testing.assert_allclose(tr["beta_t"] ** 2 + tr["beta_r"] ** 2, 1.0, atol=1e-12)
    assert np.all((tr["phi_t"] >= 0) & (tr["phi_t"] < 2 * np.pi))


@pytest.mark.parametrize("branch", ["feasible", "violated"])  # s = 1 refinement branch / s = 0 ReLU branch
def test_gradient_matches_finite_differences(branch):
    cfg, sys_, sol = _setup(B=2, N=6, M=3, U=4, seed=3)
    R0 = sys_.rates_bps_hz(sol)
    # pick Rmin so that every instance is feasible (0.5 * min rate) or every instance is infeasible (2 * max rate)
    r_min_bps = 0.5 * R0.min() if branch == "feasible" else 2.0 * R0.max()
    cfg.r_min_mbps = r_min_bps * cfg.bandwidth_hz / 1e6
    V, grad, R = sys_.surrogate_and_grad(sol)
    feas = np.all(R >= cfg.r_min_bps_hz, axis=1)
    assert feas.all() if branch == "feasible" else not feas.any()

    def Vof(s):
        return sys_.surrogate(s)

    eps = 1e-6
    rng = np.random.default_rng(0)

    def check(name, getter, setter, analytic):
        flat = getter(sol).ravel()
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for ix in idxs:
            plus, minus = sol.copy(), sol.copy()
            fp = getter(plus).copy().reshape(-1); fp[ix] += eps; setter(plus, fp.reshape(getter(sol).shape))
            fm = getter(minus).copy().reshape(-1); fm[ix] -= eps; setter(minus, fm.reshape(getter(sol).shape))
            num = (Vof(plus) - Vof(minus)) / (2 * eps)
            b = ix // (flat.size // sol.batch)
            ana = analytic.ravel()[ix]
            np.testing.assert_allclose(ana, num[b], rtol=2e-4, atol=1e-6 * max(1.0, abs(num[b])), err_msg=name)

    check("alpha", lambda s: s.alpha, lambda s, v: setattr(s, "alpha", v), grad.alpha)
    check("a", lambda s: s.a, lambda s, v: setattr(s, "a", v), grad.a)
    check("phi_t", lambda s: s.phi_t, lambda s, v: setattr(s, "phi_t", v), grad.phi_t)
    check("phi_r", lambda s: s.phi_r, lambda s, v: setattr(s, "phi_r", v), grad.phi_r)
    check("Wr", lambda s: s.W_hat.real, lambda s, v: setattr(s, "W_hat", v + 1j * s.W_hat.imag), grad.W_hat.real)
    check("Wi", lambda s: s.W_hat.imag, lambda s, v: setattr(s, "W_hat", s.W_hat.real + 1j * v), grad.W_hat.imag)


def test_normalize_and_csi_perturbation_shapes():
    cfg, sys_, _ = _setup()
    Gn, Hn, sg, sh = normalize_channels(sys_.G, sys_.H)
    np.testing.assert_allclose(np.abs(Gn).mean(axis=(1, 2)), 1.0)
    G2, H2 = perturb_csi(sys_.G, sys_.H, 0.0, cfg.noise_power_w, 0)
    np.testing.assert_array_equal(G2, sys_.G)
    G3, H3 = perturb_csi(sys_.G, sys_.H, 0.6, cfg.noise_power_w, 0)
    assert G3.shape == sys_.G.shape and not np.array_equal(G3, sys_.G)
    # error std = sqrt(eta*sigma^2) ~ 7.7e-11 -- tiny w.r.t. |G| ~ 5e-5 (as in the paper, where Table 7 degrades only mildly)
    assert 0 < np.abs(G3 - sys_.G).mean() < 1e-9
