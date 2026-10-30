"""LLM-guided adaptive differentiable projection (paper Sec. II-A.e and III-C, eqs. (9)-(12), (24)-(25)).

Momentum gradient descent on the surrogate V (eq. (24)):

    v[i+1] = gamma * v[i] + (1 - gamma) * grad V(y[i])          y[i+1] = y[i] - mu * v[i+1]

with v[0] = 0 and y[0] = y^(T) (the multi-agent solution).  The hyper-parameters theta = {mu, gamma}
are held fixed within a *round*; after ``probe_steps`` steps without feasibility the hyper-parameter
agent LLM_H is consulted (via a callback) for theta^(l+1).  At most L consultations happen and the
whole projection uses at most I steps.  Once every QoS constraint holds the projection keeps
refining the sum-rate (second term of V) until the sum-rate has converged.

This module is LLM-agnostic: the hyper-parameter agent is injected as ``theta_provider``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import ProjectionConfig
from .physics import STARRISSystem, Solution


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class RoundContext:
    """Information handed to the hyper-parameter agent at the end of round l (the 'sp^(l)' of eq. (10))."""

    round_idx: int  # l
    theta: Tuple[float, float]  # theta^(l) = (mu, gamma) used in this round
    records: List[Dict[str, Any]]  # per-step records of this round: step, V, sum_rate_mbps, min_rate_mbps, n_viol, rates_mbps
    solution: Solution  # y[i] at the end of the round (batch of 1)
    step: int  # global step index i at the end of the round
    steps_left: int  # remaining budget (I - i)


ThetaProvider = Callable[[RoundContext], Tuple[float, float]]


@dataclass
class ProjectionResult:
    solution: Solution  # best feasible iterate if feasibility was reached, else the last iterate
    last_solution: Solution
    feasible: bool
    s_zv: Optional[int]  # S_ZV: projection steps until zero constraint violation (None if never feasible)
    s_conv: Optional[int]  # S_conv: steps until the sum-rate converged (None if never feasible)
    steps_taken: int
    n_hyper_calls: int  # number of LLM_H consultations (l at the end)
    theta_history: List[Tuple[float, float]]
    trajectory: Dict[str, List[float]]  # per step i: V, sum_rate_mbps, n_viol, min_rate_mbps
    rounds: List[Dict[str, Any]]
    time_s: float = 0.0


# --------------------------------------------------------------------------------------
# One momentum step
# --------------------------------------------------------------------------------------
def momentum_step(
    system: STARRISSystem, y: Solution, v: Solution, mu: float, gamma: float
) -> Tuple[Solution, Solution, np.ndarray, np.ndarray]:
    """Eq. (25).  Returns (y[i+1], v[i+1], V(y[i]), R(y[i]) [bps/Hz])."""
    V, grad, R = system.surrogate_and_grad(y)
    v_new = v.combine(grad, gamma, 1.0 - gamma)  # type: ignore[arg-type]
    y_new = y.axpy(v_new, -mu)
    return y_new, v_new, V, R


def clamp_theta(mu: float, gamma: float, cfg: ProjectionConfig) -> Tuple[float, float]:
    mu = float(np.clip(mu, *cfg.mu_bounds))
    gamma = float(np.clip(gamma, *cfg.gamma_bounds))
    return mu, gamma


# --------------------------------------------------------------------------------------
# Single-instance adaptive projection (used by LLM4RRM)
# --------------------------------------------------------------------------------------
def run_projection(
    system: STARRISSystem,
    y0: Solution,
    cfg: ProjectionConfig,
    theta_provider: Optional[ThetaProvider] = None,
    theta0: Optional[Tuple[float, float]] = None,
) -> ProjectionResult:
    """Adaptive projection of one instance (system.B == 1).

    ``theta_provider`` is called at the end of a round that failed to reach feasibility (at most L times) and
    must return the next (mu, gamma).  If ``None`` (or ``cfg.hyper_mode == 'fixed'``) theta^(0) is used throughout.
    """
    if system.B != 1 or y0.batch != 1:
        raise ValueError("run_projection handles a single instance; use project_fixed_batch for batches")
    t_start = time.perf_counter()
    r_min = system.cfg.r_min_bps_hz
    mbps = system.cfg.mbps_per_bps_hz
    use_llm = theta_provider is not None and cfg.hyper_mode == "llm"

    theta = clamp_theta(*(theta0 or (cfg.mu0, cfg.gamma0)), cfg)
    theta_history = [theta]
    y = y0.copy()
    v = Solution.zeros_like(y)
    l = 0  # tuning round index == number of LLM_H consultations so far
    round_start = 0
    round_records: List[Dict[str, Any]] = []
    rounds: List[Dict[str, Any]] = []
    traj: Dict[str, List[float]] = {"V": [], "sum_rate_mbps": [], "n_viol": [], "min_rate_mbps": []}

    s_zv: Optional[int] = None
    best_rate = -np.inf
    best_y: Optional[Solution] = None
    best_hist: List[float] = []  # running best feasible sum-rate per step (for the convergence test)
    feasible_flags: List[bool] = []
    i = 0
    last_y = y

    while True:
        V, grad, R = system.surrogate_and_grad(y)
        rates_mbps = R[0] * mbps
        sum_rate = float(rates_mbps.sum())
        n_viol = int(np.sum(R[0] < r_min))
        feasible = n_viol == 0
        rec = dict(step=i, V=float(V[0]), sum_rate_mbps=sum_rate, min_rate_mbps=float(rates_mbps.min()),
                   n_viol=n_viol, rates_mbps=[float(x) for x in rates_mbps])
        round_records.append(rec)
        for k, key in (("V", "V"), ("sum_rate_mbps", "sum_rate_mbps"), ("n_viol", "n_viol"), ("min_rate_mbps", "min_rate_mbps")):
            traj[k].append(float(rec[key]))
        feasible_flags.append(feasible)
        last_y = y
        if feasible:
            if s_zv is None:
                s_zv = i
            if sum_rate > best_rate:
                best_rate, best_y = sum_rate, y.copy()
        best_hist.append(best_rate if np.isfinite(best_rate) else 0.0)

        # ---- stopping tests ----
        if i >= cfg.I or not y.is_finite():
            break
        if s_zv is not None and i - s_zv >= cfg.patience:
            gain = best_hist[i] - best_hist[i - cfg.patience]
            if gain < cfg.conv_rel_tol * max(best_hist[i], 1e-12):
                break

        # ---- end of a tuning round? (only while still infeasible) ----
        if use_llm and s_zv is None and l < cfg.L and (i - round_start) >= cfg.probe_steps:
            ctx = RoundContext(round_idx=l, theta=theta, records=list(round_records), solution=y.copy(),
                               step=i, steps_left=cfg.I - i)
            rounds.append(dict(round=l, theta=theta, start_step=round_start, end_step=i,
                               V_start=round_records[0]["V"], V_end=rec["V"], n_viol_end=n_viol))
            mu_new, gamma_new = theta_provider(ctx)  # type: ignore[misc]
            theta = clamp_theta(mu_new, gamma_new, cfg)
            theta_history.append(theta)
            l += 1
            round_start = i
            round_records = []

        y, v, _, _ = momentum_step(system, y, v, theta[0], theta[1])
        i += 1

    rounds.append(dict(round=l, theta=theta, start_step=round_start, end_step=i,
                       V_start=round_records[0]["V"] if round_records else None,
                       V_end=traj["V"][-1], n_viol_end=int(traj["n_viol"][-1])))

    feasible_any = s_zv is not None
    s_conv: Optional[int] = None
    if feasible_any:
        target = cfg.conv_fraction * best_rate
        for j, (ok, sr) in enumerate(zip(feasible_flags, traj["sum_rate_mbps"])):
            if ok and sr >= target:
                s_conv = j
                break
    return ProjectionResult(
        solution=(best_y if best_y is not None else last_y).copy(),
        last_solution=last_y.copy(),
        feasible=feasible_any,
        s_zv=s_zv,
        s_conv=s_conv,
        steps_taken=i,
        n_hyper_calls=l,
        theta_history=theta_history,
        trajectory=traj,
        rounds=rounds,
        time_s=time.perf_counter() - t_start,
    )


# --------------------------------------------------------------------------------------
# Batched fixed-theta projection (memory-bank generation, random-init reference points)
# --------------------------------------------------------------------------------------
def project_fixed_batch(
    system: STARRISSystem,
    y0: Solution,
    mu: float,
    gamma: float,
    max_steps: int,
    patience: int = 10,
    conv_rel_tol: float = 1e-3,
) -> Tuple[Solution, np.ndarray, np.ndarray]:
    """Run eq. (25) with fixed theta on a whole batch.

    Returns (best feasible iterate per instance [last iterate if never feasible], feasible (B,), steps_to_feasible (B,) [-1 if never]).
    """
    B = system.B
    r_min = system.cfg.r_min_bps_hz
    mbps = system.cfg.mbps_per_bps_hz
    y = y0.copy()
    v = Solution.zeros_like(y)
    best = y0.copy()
    best_rate = np.full(B, -np.inf)
    first_feas = np.full(B, -1, dtype=int)
    hist_rate = np.full(B, -np.inf)  # best rate `patience` steps ago
    rate_ring: List[np.ndarray] = []
    done = np.zeros(B, dtype=bool)

    for i in range(max_steps + 1):
        V, grad, R = system.surrogate_and_grad(y)
        feas = np.all(R >= r_min, axis=1)
        sr = R.sum(axis=1) * mbps
        newly = feas & (first_feas < 0)
        first_feas[newly] = i
        improved = feas & (sr > best_rate)
        if improved.any():
            best_rate[improved] = sr[improved]
            best.W_hat[improved] = y.W_hat[improved]
            best.alpha[improved] = y.alpha[improved]
            best.a[improved] = y.a[improved]
            best.phi_t[improved] = y.phi_t[improved]
            best.phi_r[improved] = y.phi_r[improved]
        rate_ring.append(np.where(np.isfinite(best_rate), best_rate, 0.0))
        if len(rate_ring) > patience:
            old = rate_ring[-patience - 1]
            cur = rate_ring[-1]
            done |= (first_feas >= 0) & ((cur - old) < conv_rel_tol * np.maximum(cur, 1e-12))
            if done.all():
                break
        if i == max_steps:
            break
        v = v.combine(grad, gamma, 1.0 - gamma)  # type: ignore[arg-type]
        y = y.axpy(v, -mu)
        if not y.is_finite():
            break

    never = best_rate == -np.inf
    if never.any():  # report last iterate for instances that never became feasible
        best.W_hat[never] = y.W_hat[never]
        best.alpha[never] = y.alpha[never]
        best.a[never] = y.a[never]
        best.phi_t[never] = y.phi_t[never]
        best.phi_r[never] = y.phi_r[never]
    return best, ~never, first_feas
