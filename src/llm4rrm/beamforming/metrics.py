"""Result containers and the performance metrics reported in the paper (Sec. V).

Notation (paper): R_pre / R_post = average sum-rate before / after projection; V_pre / V_post = QoS violation
probability = share of instances with at least one user below Rmin; S_ZV = projection steps to zero violation
(averaged only over instances that reached it); S_conv = steps to convergence; T_conv = wall-clock time of the
whole method; l_avg = average number of hyper-parameter tuning rounds (LLM_H calls).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class InstanceResult:
    index: int
    repeat: int = 0
    ok: bool = True  # False if the LLM backend failed for this instance
    error: Optional[str] = None
    pre: Dict[str, Any] = field(default_factory=dict)  # sum_rate_mbps, rates_mbps, n_violations, feasible (true channels)
    post: Dict[str, Any] = field(default_factory=dict)
    s_zv: Optional[int] = None
    s_conv: Optional[int] = None
    projection_steps: int = 0
    n_hyper_calls: int = 0
    theta_history: List[List[float]] = field(default_factory=list)
    projection_trajectory: Dict[str, List[float]] = field(default_factory=dict)
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    stopped_at_iteration: int = 0
    llm: Dict[str, float] = field(default_factory=dict)  # calls, prompt_tokens, completion_tokens, latency_s
    time_llm_s: float = 0.0
    time_projection_s: float = 0.0
    time_total_s: float = 0.0
    y_T: Dict[str, Any] = field(default_factory=dict)  # pre-projection decision vector (for UQ / grid search)
    y_final: Dict[str, Any] = field(default_factory=dict)
    n_agent_failures: int = 0
    transcripts: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def aggregate(results: List[InstanceResult]) -> Dict[str, Any]:
    """Average the per-instance results into the quantities of the paper's tables."""
    ok = [r for r in results if r.ok]
    if not ok:
        return {"n": 0, "n_failed": len(results)}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else float("nan")

    zv = [r for r in ok if r.post.get("feasible")]
    n_users = len(ok[0].post.get("rates_mbps", [])) or 1
    out = {
        "n": len(ok),
        "n_failed": len(results) - len(ok),
        "R_pre": mean(r.pre["sum_rate_mbps"] for r in ok),
        "R_post": mean(r.post["sum_rate_mbps"] for r in ok),
        "V_pre_pct": 100.0 * mean(0.0 if r.pre["feasible"] else 1.0 for r in ok),
        "V_post_pct": 100.0 * mean(0.0 if r.post["feasible"] else 1.0 for r in ok),
        "user_violation_pre_pct": 100.0 * mean(r.pre["n_violations"] / n_users for r in ok),
        "user_violation_post_pct": 100.0 * mean(r.post["n_violations"] / n_users for r in ok),
        "S_ZV": mean(r.s_zv for r in zv),
        "S_conv": mean(r.s_conv for r in zv),
        "T_conv_s": mean(r.time_total_s for r in ok),
        "T_llm_s": mean(r.time_llm_s for r in ok),
        "T_projection_s": mean(r.time_projection_s for r in ok),
        "l_avg": mean(r.n_hyper_calls for r in ok),
        "llm_calls": mean(r.llm.get("calls") for r in ok),
        "prompt_tokens": mean(r.llm.get("prompt_tokens") for r in ok),
        "completion_tokens": mean(r.llm.get("completion_tokens") for r in ok),
        "agent_failures": mean(r.n_agent_failures for r in ok),
    }
    return out


def repeat_std(results: List[InstanceResult]) -> Dict[str, float]:
    """Table 9: per-channel standard deviation of the post-projection sum-rate over repeated runs, averaged over channels."""
    by_idx: Dict[int, List[float]] = {}
    for r in results:
        if r.ok:
            by_idx.setdefault(r.index, []).append(r.post["sum_rate_mbps"])
    stds = [float(np.std(v, ddof=1)) for v in by_idx.values() if len(v) > 1]
    means = [float(np.mean(v)) for v in by_idx.values() if len(v) > 1]
    if not stds:
        return {"mean_post_sum_rate": float("nan"), "mean_std": float("nan"), "n_channels": 0}
    return {"mean_post_sum_rate": float(np.mean(means)), "mean_std": float(np.mean(stds)), "n_channels": len(stds)}


def format_row(label: str, agg: Dict[str, Any]) -> str:
    if not agg.get("n"):
        return f"{label:<28} (no successful instances; {agg.get('n_failed', 0)} failed)"
    return (f"{label:<28} R_pre={agg['R_pre']:8.3f}  R_post={agg['R_post']:8.3f}  V_pre={agg['V_pre_pct']:5.1f}%  "
            f"V_post={agg['V_post_pct']:5.1f}%  S_ZV={agg['S_ZV']:6.1f}  S_conv={agg['S_conv']:6.1f}  "
            f"T={agg['T_conv_s']:7.1f}s  l_avg={agg['l_avg']:.2f}  calls={agg['llm_calls']:.1f}")
