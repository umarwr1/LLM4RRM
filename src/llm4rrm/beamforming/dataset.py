"""Memory bank D of past successful optimisation examples (paper Sec. II-A.c, Alg. 1 step 1).

Each entry is the tuple (x, y, F, c): channels x = {G, H}, a *feasible* high-quality decision vector y_hat,
its objective F (sum-rate) and the constraint evaluation c (per-user rates; all >= Rmin).

The bank is built from previously solved instances.  This module generates them *without any LLM*: draw channels, start from a random decision vector and run the
differentiable projection of eq. (25) (feasibility + sum-rate refinement) until convergence, keeping
only feasible results.  Retrieval uses the Frobenius norm between (normalised) channels, as in the paper.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import SystemConfig
from .channels import generate_channels, normalize_channels
from .physics import STARRISSystem, Solution
from .projection import project_fixed_batch


@dataclass
class MemoryEntry:
    G: np.ndarray  # (N, M) normalised channel shown in prompts
    H: np.ndarray  # (N, U)
    solution: Solution  # batch of 1
    sum_rate_mbps: float
    rates_mbps: np.ndarray  # (U,)


class MemoryBank:
    """Container + retrieval function phi(x, D)."""

    def __init__(self, G, H, W_hat, alpha, a, phi_t, phi_r, sum_rate_mbps, rates_mbps, meta: Dict[str, Any] | None = None):
        # G, H are stored *normalised* (mean |entry| = 1) because that is what the LLM sees and what similarity uses
        self.G = np.asarray(G)
        self.H = np.asarray(H)
        self.W_hat = np.asarray(W_hat)
        self.alpha = np.asarray(alpha)
        self.a = np.asarray(a)
        self.phi_t = np.asarray(phi_t)
        self.phi_r = np.asarray(phi_r)
        self.sum_rate_mbps = np.asarray(sum_rate_mbps)
        self.rates_mbps = np.asarray(rates_mbps)
        self.meta = dict(meta or {})

    def __len__(self) -> int:
        return int(self.G.shape[0])

    # ---- retrieval ----------------------------------------------------------------
    def retrieve(
        self,
        G_n: np.ndarray,
        H_n: np.ndarray,
        n: int,
        method: str = "frobenius",
        rng: Optional[np.random.Generator] = None,
    ) -> List[MemoryEntry]:
        """The ``n`` most similar entries to the (normalised) current channels, most similar first."""
        n = min(n, len(self))
        if n <= 0:
            return []
        if method == "random":
            rng = rng or np.random.default_rng()
            idx = rng.choice(len(self), size=n, replace=False)
        elif method == "frobenius":
            d = np.sqrt(np.sum(np.abs(self.G - G_n[None]) ** 2, axis=(1, 2))) + np.sqrt(
                np.sum(np.abs(self.H - H_n[None]) ** 2, axis=(1, 2))
            )
            idx = np.argsort(d)[:n]
        elif method == "cosine":
            def flat(g, h):
                z = np.concatenate([g.reshape(g.shape[0], -1), h.reshape(h.shape[0], -1)], axis=1)
                return np.concatenate([z.real, z.imag], axis=1)

            bank = flat(self.G, self.H)
            cur = flat(G_n[None], H_n[None])[0]
            sim = bank @ cur / (np.linalg.norm(bank, axis=1) * np.linalg.norm(cur) + 1e-12)
            idx = np.argsort(-sim)[:n]
        else:
            raise ValueError(f"unknown similarity '{method}'")
        return [self.entry(int(i)) for i in idx]

    def entry(self, i: int) -> MemoryEntry:
        return MemoryEntry(
            G=self.G[i], H=self.H[i],
            solution=Solution(self.W_hat[i : i + 1], self.alpha[i : i + 1], self.a[i : i + 1],
                              self.phi_t[i : i + 1], self.phi_r[i : i + 1]),
            sum_rate_mbps=float(self.sum_rate_mbps[i]),
            rates_mbps=self.rates_mbps[i],
        )

    # ---- update D (Alg. 1, line 10) ---------------------------------------------------
    def add(self, G_n: np.ndarray, H_n: np.ndarray, sol: Solution, sum_rate_mbps: float, rates_mbps: np.ndarray) -> None:
        self.G = np.concatenate([self.G, G_n[None]], axis=0)
        self.H = np.concatenate([self.H, H_n[None]], axis=0)
        self.W_hat = np.concatenate([self.W_hat, sol.W_hat[:1]], axis=0)
        self.alpha = np.concatenate([self.alpha, sol.alpha[:1]], axis=0)
        self.a = np.concatenate([self.a, sol.a[:1]], axis=0)
        self.phi_t = np.concatenate([self.phi_t, sol.phi_t[:1]], axis=0)
        self.phi_r = np.concatenate([self.phi_r, sol.phi_r[:1]], axis=0)
        self.sum_rate_mbps = np.concatenate([self.sum_rate_mbps, [sum_rate_mbps]])
        self.rates_mbps = np.concatenate([self.rates_mbps, np.asarray(rates_mbps)[None]], axis=0)

    # ---- persistence --------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, G=self.G, H=self.H, W_hat=self.W_hat, alpha=self.alpha, a=self.a, phi_t=self.phi_t,
            phi_r=self.phi_r, sum_rate_mbps=self.sum_rate_mbps, rates_mbps=self.rates_mbps,
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return cls(z["G"], z["H"], z["W_hat"], z["alpha"], z["a"], z["phi_t"], z["phi_r"],
                   z["sum_rate_mbps"], z["rates_mbps"], meta)

    def check_compatible(self, cfg: SystemConfig) -> List[str]:
        """Raise on dimension mismatch; return warnings for constraint mismatches."""
        m = self.meta
        for key, val in (("M", cfg.M), ("N", cfg.N), ("U", cfg.U)):
            if key in m and int(m[key]) != int(val):
                raise ValueError(f"Memory bank was generated for {key}={m[key]} but the run uses {key}={val}")
        msgs = []
        for key, val in (("r_min_mbps", cfg.r_min_mbps), ("p_max_w", cfg.p_max_w), ("bandwidth_hz", cfg.bandwidth_hz)):
            if key in m and abs(float(m[key]) - float(val)) > 1e-9 * max(1.0, abs(val)):
                msgs.append(f"memory bank {key}={m[key]} differs from run {key}={val}; examples may not be feasible here")
        for msg in msgs:
            warnings.warn(msg, stacklevel=2)
        return msgs


# --------------------------------------------------------------------------------------
# Generation (no LLM involved)
# --------------------------------------------------------------------------------------
def generate_memory_bank(
    cfg: SystemConfig,
    size: int,
    seed: int,
    mu_schedule: Sequence[float] = (0.2, 0.05, 0.01, 0.5, 1.0),
    gamma: float = 0.9,
    max_steps: int = 500,
    max_attempts: int = 10,
    verbose: bool = True,
) -> MemoryBank:
    """Generate ``size`` feasible, sum-rate-refined examples for the given system configuration.

    Channels are drawn with ``seed`` (use a seed different from the test seed). For every channel a random
    decision vector is refined with the momentum projection of eq. (25); different step sizes are tried on
    instances that did not become feasible (a larger step is not always better, see the paper's Sec. V-B.3).
    """
    G_all, H_all = generate_channels(cfg, size, seed)
    system = STARRISSystem(G_all, H_all, cfg)
    rng = np.random.default_rng(seed + 10_000)
    out_sol = Solution.zeros_like(Solution.random(cfg, size, rng))
    done = np.zeros(size, dtype=bool)

    for attempt in range(max_attempts):
        todo = np.where(~done)[0]
        if todo.size == 0:
            break
        mu = mu_schedule[attempt % len(mu_schedule)]
        sub = STARRISSystem(G_all[todo], H_all[todo], cfg)
        y0 = Solution.random(cfg, todo.size, rng)
        best, feas, _ = project_fixed_batch(sub, y0, mu, gamma, max_steps)
        for j, idx in enumerate(todo):
            if feas[j]:
                out_sol.W_hat[idx] = best.W_hat[j]
                out_sol.alpha[idx] = best.alpha[j]
                out_sol.a[idx] = best.a[j]
                out_sol.phi_t[idx] = best.phi_t[j]
                out_sol.phi_r[idx] = best.phi_r[j]
                done[idx] = True
        if verbose:
            print(f"[memory bank] attempt {attempt + 1} (mu={mu}): {int(done.sum())}/{size} feasible examples")

    keep = np.where(done)[0]
    if keep.size < size:
        warnings.warn(f"Only {keep.size}/{size} feasible examples could be generated; bank truncated")
    G_k, H_k = G_all[keep], H_all[keep]
    sol_k = Solution(out_sol.W_hat[keep], out_sol.alpha[keep], out_sol.a[keep], out_sol.phi_t[keep], out_sol.phi_r[keep]).wrapped()
    ev = STARRISSystem(G_k, H_k, cfg).evaluate(sol_k)
    assert ev["feasible"].all()
    Gn, Hn, _, _ = normalize_channels(G_k, H_k)
    meta = dict(M=cfg.M, N=cfg.N, U=cfg.U, r_min_mbps=cfg.r_min_mbps, p_max_w=cfg.p_max_w,
                bandwidth_hz=cfg.bandwidth_hz, noise_power_dbm=cfg.noise_power_dbm, seed=seed,
                generator="random-init + momentum projection (no LLM)")
    return MemoryBank(Gn, Hn, sol_k.W_hat, sol_k.alpha, sol_k.a, sol_k.phi_t, sol_k.phi_r,
                      ev["sum_rate_mbps"], ev["rates_mbps"], meta)
