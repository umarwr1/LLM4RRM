"""STAR-RIS physics: closed-form transformations T(.), rates, and the projection surrogate V with an
analytic gradient (paper eqs. (15)-(17), (23)-(25)).

Decision variables (unconstrained, y_hat): W_hat in C^{M x U}, alpha in R, a in R^N, phi_t, phi_r in R^N.

    W        = W_hat / ||W_hat||_F * Sig(alpha) * sqrt(Pmax)         (power constraint (16b) holds by design)
    beta_t   = sqrt(Sig(a)),  beta_r = sqrt(1 - Sig(a))              (energy splitting (16d) holds by design)
    Theta_l  = diag(beta_l * exp(j phi_l)),  l in {t, r}

    gamma_u  = |h_u^H Theta_l(u) G w_u|^2 / (sum_{i != u} |h_u^H Theta_l(u) G w_i|^2 + sigma^2)
    R_u      = log2(1 + gamma_u)   [bps/Hz]   (x bandwidth -> Mbps)

The surrogate (24) is differentiated analytically (no autograd framework needed); the gradient is
verified against finite differences in ``tests/test_physics.py``.

All functions are batched: leading dimension B = number of independent channel instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from scipy.special import expit as sigmoid

from ..config import SystemConfig

LN2 = float(np.log(2.0))
TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------------------
# Decision variables
# --------------------------------------------------------------------------------------
@dataclass
class Solution:
    """Unconstrained decision vector y_hat = {W_hat, alpha, a, phi_t, phi_r} (batched)."""

    W_hat: np.ndarray  # (B, M, U) complex
    alpha: np.ndarray  # (B,)
    a: np.ndarray  # (B, N)
    phi_t: np.ndarray  # (B, N)
    phi_r: np.ndarray  # (B, N)

    # ---- constructors ------------------------------------------------------------
    @classmethod
    def random(cls, cfg: SystemConfig, batch: int, rng: np.random.Generator) -> "Solution":
        """Random initial point (as used for the memory-bank generator and the Random-Init reference)."""
        return cls(
            W_hat=rng.standard_normal((batch, cfg.M, cfg.U)) + 1j * rng.standard_normal((batch, cfg.M, cfg.U)),
            alpha=rng.standard_normal(batch),
            a=rng.standard_normal((batch, cfg.N)),
            phi_t=rng.uniform(0.0, TWO_PI, (batch, cfg.N)),
            phi_r=rng.uniform(0.0, TWO_PI, (batch, cfg.N)),
        )

    @classmethod
    def zeros_like(cls, other: "Solution") -> "Solution":
        return cls(
            W_hat=np.zeros_like(other.W_hat),
            alpha=np.zeros_like(other.alpha),
            a=np.zeros_like(other.a),
            phi_t=np.zeros_like(other.phi_t),
            phi_r=np.zeros_like(other.phi_r),
        )

    @classmethod
    def from_arrays(cls, Wr, Wi, alpha, a, phi_t, phi_r) -> "Solution":
        """Single instance from raw arrays (M,U),(M,U),scalar,(N,),(N,),(N,) -> batch of 1."""
        W = np.asarray(Wr, dtype=float) + 1j * np.asarray(Wi, dtype=float)
        return cls(
            W_hat=W[None],
            alpha=np.asarray([float(alpha)]),
            a=np.asarray(a, dtype=float)[None],
            phi_t=np.asarray(phi_t, dtype=float)[None],
            phi_r=np.asarray(phi_r, dtype=float)[None],
        )

    # ---- helpers -----------------------------------------------------------------
    @property
    def batch(self) -> int:
        return self.alpha.shape[0]

    def copy(self) -> "Solution":
        return Solution(self.W_hat.copy(), self.alpha.copy(), self.a.copy(), self.phi_t.copy(), self.phi_r.copy())

    def __getitem__(self, idx: int) -> "Solution":
        """Instance ``idx`` as a batch of one."""
        return Solution(
            self.W_hat[idx : idx + 1].copy(),
            self.alpha[idx : idx + 1].copy(),
            self.a[idx : idx + 1].copy(),
            self.phi_t[idx : idx + 1].copy(),
            self.phi_r[idx : idx + 1].copy(),
        )

    def axpy(self, other: "Solution", coef: float) -> "Solution":
        """Return self + coef * other (used for the momentum update)."""
        return Solution(
            self.W_hat + coef * other.W_hat,
            self.alpha + coef * other.alpha,
            self.a + coef * other.a,
            self.phi_t + coef * other.phi_t,
            self.phi_r + coef * other.phi_r,
        )

    def combine(self, other: "Solution", c_self: float, c_other: float) -> "Solution":
        """Return c_self * self + c_other * other (used for the momentum velocity)."""
        return Solution(
            c_self * self.W_hat + c_other * other.W_hat,
            c_self * self.alpha + c_other * other.alpha,
            c_self * self.a + c_other * other.a,
            c_self * self.phi_t + c_other * other.phi_t,
            c_self * self.phi_r + c_other * other.phi_r,
        )

    def wrapped(self) -> "Solution":
        """Phases wrapped to [0, 2*pi) (part of the transformation T(.))."""
        out = self.copy()
        out.phi_t = np.mod(out.phi_t, TWO_PI)
        out.phi_r = np.mod(out.phi_r, TWO_PI)
        return out

    def is_finite(self) -> bool:
        return all(np.all(np.isfinite(x)) for x in (self.W_hat, self.alpha, self.a, self.phi_t, self.phi_r))

    # ---- (de)serialisation of a single instance -------------------------------------
    def to_dict(self, idx: int = 0, decimals: int | None = None) -> Dict[str, Any]:
        def r(x):
            x = np.asarray(x)
            return np.round(x, decimals) if decimals is not None else x

        return {
            "Wr": r(self.W_hat[idx].real).tolist(),
            "Wi": r(self.W_hat[idx].imag).tolist(),
            "alpha": float(r(self.alpha[idx])),
            "a": r(self.a[idx]).tolist(),
            "phi_t": r(self.phi_t[idx]).tolist(),
            "phi_r": r(self.phi_r[idx]).tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Solution":
        return cls.from_arrays(d["Wr"], d["Wi"], d["alpha"], d["a"], d["phi_t"], d["phi_r"])


# --------------------------------------------------------------------------------------
# System (channels + analytic evaluation)
# --------------------------------------------------------------------------------------
class STARRISSystem:
    """A batch of STAR-RIS channel instances with differentiable evaluation of rates and the surrogate V."""

    def __init__(self, G: np.ndarray, H: np.ndarray, cfg: SystemConfig):
        G = np.asarray(G)
        H = np.asarray(H)
        if G.ndim == 2:
            G, H = G[None], H[None]
        B, N, M = G.shape
        if H.shape[0] != B or H.shape[1] != N or H.shape[2] != cfg.U or M != cfg.M or N != cfg.N:
            raise ValueError(f"Channel shapes G{G.shape}, H{H.shape} do not match config (M={cfg.M},N={cfg.N},U={cfg.U})")
        self.cfg = cfg
        self.G = G.astype(np.complex128)
        self.H = H.astype(np.complex128)
        self.B, self.N, self.M, self.U = B, N, M, cfg.U
        # Z[b,u,n,m] = conj(H[b,n,u]) * G[b,n,m]  so that  h_u^H Theta G = theta_u^T Z_u
        self.Z = np.einsum("bnu,bnm->bunm", self.H.conj(), self.G)
        mask = np.zeros(self.U)
        mask[: int(cfg.n_transmit_users)] = 1.0
        self.mask_t = mask  # 1 -> user on the transmission side
        self.offdiag = 1.0 - np.eye(self.U)
        self.noise = cfg.noise_power_w
        self.p_max = cfg.p_max_w

    # ---- forward pass ---------------------------------------------------------------
    def _forward(self, sol: Solution) -> Dict[str, np.ndarray]:
        W_hat = sol.W_hat
        nW = np.sqrt(np.sum(np.abs(W_hat) ** 2, axis=(1, 2)))
        nW = np.maximum(nW, 1e-12)
        s = sigmoid(sol.alpha)
        kappa = s * np.sqrt(self.p_max) / nW
        W = W_hat * kappa[:, None, None]
        sig_a = sigmoid(sol.a)
        beta_t = np.sqrt(sig_a)
        beta_r = np.sqrt(1.0 - sig_a)
        ph_t = np.exp(1j * sol.phi_t)
        ph_r = np.exp(1j * sol.phi_r)
        theta_t = beta_t * ph_t
        theta_r = beta_r * ph_r
        mt = self.mask_t[None, :, None]
        theta_u = mt * theta_t[:, None, :] + (1.0 - mt) * theta_r[:, None, :]  # (B,U,N)
        c = np.einsum("bun,bunm->bum", theta_u, self.Z)  # (B,U,M) effective channel of user u
        S = np.einsum("bum,bmi->bui", c, W)  # (B,U,U)
        P = np.abs(S) ** 2
        sig = np.einsum("buu->bu", P)
        interf = np.sum(P * self.offdiag[None], axis=2)
        D = interf + self.noise
        Tt = D + sig
        R = np.log2(Tt / D)  # = log2(1 + SINR)   [bps/Hz]
        return dict(
            nW=nW, s=s, kappa=kappa, W=W, sig_a=sig_a, beta_t=beta_t, beta_r=beta_r, ph_t=ph_t, ph_r=ph_r,
            theta_u=theta_u, c=c, S=S, P=P, D=D, T=Tt, R=R,
        )

    # ---- public evaluation -------------------------------------------------------------
    def rates_bps_hz(self, sol: Solution) -> np.ndarray:
        return self._forward(sol)["R"]

    def evaluate(self, sol: Solution) -> Dict[str, np.ndarray]:
        """Objective F and constraint flags f_j for each instance (paper eq. (21))."""
        R = self.rates_bps_hz(sol)
        r_min = self.cfg.r_min_bps_hz
        violated = R < r_min
        rates_mbps = R * self.cfg.mbps_per_bps_hz
        return dict(
            rates_bps_hz=R,
            rates_mbps=rates_mbps,
            sum_rate_mbps=rates_mbps.sum(axis=1),
            violated=violated,
            n_violations=violated.sum(axis=1),
            feasible=~violated.any(axis=1),
        )

    def transformed(self, sol: Solution) -> Dict[str, np.ndarray]:
        """T(y_hat): the constrained variables (W, beta_t, beta_r, phi_t, phi_r), phases wrapped to [0,2pi)."""
        f = self._forward(sol)
        return dict(
            W=f["W"], beta_t=f["beta_t"], beta_r=f["beta_r"],
            phi_t=np.mod(sol.phi_t, TWO_PI), phi_r=np.mod(sol.phi_r, TWO_PI),
            power=np.sum(np.abs(f["W"]) ** 2, axis=(1, 2)),
        )

    # ---- surrogate V (24) and its gradient ------------------------------------------
    def surrogate(self, sol: Solution) -> np.ndarray:
        return self.surrogate_and_grad(sol, want_grad=False)[0]

    def surrogate_and_grad(
        self, sol: Solution, want_grad: bool = True
    ) -> Tuple[np.ndarray, Solution | None, np.ndarray]:
        """Piecewise feasibility-and-refinement surrogate of eq. (24).

        V = (1 - s) * sum_u ReLU(exp(Rmin - R_u) - 1) + s * sum_u (exp(Rmin - R_u) - 1),
        s = 1 iff every user meets Rmin.  Rates are in bps/Hz (Rmin = r_min_bps_hz).

        Returns (V (B,), grad Solution or None, R (B,U) in bps/Hz).  The returned gradient stores, for
        W_hat, the complex number  dV/dRe + j dV/dIm.
        """
        f = self._forward(sol)
        R = f["R"]
        r_min = self.cfg.r_min_bps_hz
        feasible = np.all(R >= r_min, axis=1)  # s(x, y)  (non-differentiable indicator)
        ex = np.exp(r_min - R)
        w = np.where(feasible[:, None], 1.0, (R < r_min).astype(float))  # ReLU mask / always-on refinement
        V = np.sum(w * (ex - 1.0), axis=1)
        if not want_grad:
            return V, None, R

        g = -w * ex  # dV/dR_u
        # dV/dP_ui with R_u = log2(T_u) - log2(D_u), D_u = T_u - P_uu
        A = (g[:, :, None] / LN2) * (1.0 / f["T"][:, :, None] - self.offdiag[None] / f["D"][:, :, None])
        S, c, W = f["S"], f["c"], f["W"]
        C = 2.0 * A * np.conj(S)  # (B,U,U): C[b,u,i]

        # --- beamformer ---
        Q = np.einsum("bui,bum->bmi", C, c)
        gW = np.conj(Q)  # complex gradient wrt (Re W, Im W)
        W_hat = sol.W_hat
        kappa, nW, s = f["kappa"], f["nW"], f["s"]
        inner_hat = np.real(np.sum(np.conj(gW) * W_hat, axis=(1, 2)))
        dW_hat = kappa[:, None, None] * (gW - (inner_hat / nW**2)[:, None, None] * W_hat)
        d_alpha = (1.0 - s) * np.real(np.sum(np.conj(gW) * W, axis=(1, 2)))

        # --- STAR-RIS ---
        zeta = np.einsum("bunm,bmi->buni", self.Z, W)  # (B,U,N,U)
        E = np.einsum("bui,buni->bun", C, zeta)  # (B,U,N)
        th_u = f["theta_u"]
        mt = self.mask_t[None, :, None]
        ph_u = mt * f["ph_t"][:, None, :] + (1.0 - mt) * f["ph_r"][:, None, :]
        dphi_u = -np.imag(th_u * E)
        dbeta_u = np.real(ph_u * E)
        d_phi_t = np.sum(mt * dphi_u, axis=1)
        d_phi_r = np.sum((1.0 - mt) * dphi_u, axis=1)
        d_beta_t = np.sum(mt * dbeta_u, axis=1)
        d_beta_r = np.sum((1.0 - mt) * dbeta_u, axis=1)
        bt, br = f["beta_t"], f["beta_r"]
        # d beta_t / d a = beta_t * beta_r^2 / 2 ; d beta_r / d a = - beta_r * beta_t^2 / 2
        d_a = d_beta_t * bt * br**2 / 2.0 - d_beta_r * br * bt**2 / 2.0
        grad = Solution(W_hat=dW_hat, alpha=d_alpha, a=d_a, phi_t=d_phi_t, phi_r=d_phi_r)
        return V, grad, R
