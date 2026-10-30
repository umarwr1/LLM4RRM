"""Extraction functions E(.), E_E(.) and E_H(.) for the beamforming use case (paper eqs. (2), (6), (9))."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.parsing import ParseError, last_json_with_keys, reasoning_before, strip_think

# Variable blocks owned by each solver (paper eq. (18)):  y1 = {W_hat, alpha},  y2 = {a, phi_t, phi_r}
BF_KEYS = ["wr", "wi", "alpha"]
RIS_KEYS = ["a", "phi_t", "phi_r"]
ALL_KEYS = BF_KEYS + RIS_KEYS


@dataclass
class SolverOutput:
    block: Dict[str, object]  # the parsed variables of this block (numpy arrays / float), keys as in the JSON schema
    reasoning: str  # RT_k^(t)


def _finite(arr: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(arr)):
        raise ParseError(f"'{name}' contains NaN/Inf")
    return arr


def _vector(x, n: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"'{name}' must be a list of {n} numbers") from exc
    if arr.size != n:
        raise ParseError(f"'{name}' must have exactly {n} entries, got {arr.size}")
    return _finite(arr, name)


def _matrix(x, m: int, u: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"'{name}' must be a {m}x{u} matrix (list of {m} rows with {u} numbers)") from exc
    if arr.shape == (m, u):
        return _finite(arr, name)
    if arr.size == m * u and arr.ndim == 1:  # tolerate a flat row-major list
        return _finite(arr.reshape(m, u), name)
    raise ParseError(f"'{name}' must be a {m}x{u} matrix (list of {m} rows with {u} numbers); got shape {arr.shape}")


def parse_solver_reply(text: str, keys: List[str], M: int, U: int, N: int, trace_max_chars: int) -> SolverOutput:
    """Parse the JSON solution of one solver block and its reasoning trace."""
    obj, start = last_json_with_keys(text, keys)
    block: Dict[str, object] = {}
    if "wr" in keys:
        block["wr"] = _matrix(obj["wr"], M, U, "Wr")
        block["wi"] = _matrix(obj["wi"], M, U, "Wi")
        alpha = obj["alpha"]
        if isinstance(alpha, list) and len(alpha) == 1:
            alpha = alpha[0]
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ParseError("'alpha' must be a single real number") from exc
        if not np.isfinite(alpha):
            raise ParseError("'alpha' is not finite")
        block["alpha"] = alpha
    if "a" in keys:
        block["a"] = _vector(obj["a"], N, "a")
        block["phi_t"] = _vector(obj["phi_t"], N, "phi_t")
        block["phi_r"] = _vector(obj["phi_r"], N, "phi_r")
    return SolverOutput(block=block, reasoning=reasoning_before(text, start, trace_max_chars))


# --------------------------------------------------------------------------------------
# Evaluator feedback  E_E(.)  (eq. (6))
# --------------------------------------------------------------------------------------
_FB_CLOSE_RE = re.compile(r"</\s*FEEDBACK[_\s]*AGENT[_\s]*\d+\s*>", re.IGNORECASE)
_FB_OPEN_RE = re.compile(r"<?\s*FEEDBACK[_\s]*AGENT[_\s]*(\d+)\s*>?\s*\**\s*:?", re.IGNORECASE)
_NO_CHANGE = "No major changes needed."


def parse_evaluator_feedback(text: str, K: int, max_lines: int = 5) -> List[str]:
    """Split the evaluator reply into per-agent feedback Delta_k (k = 1..K), each at most ``max_lines`` lines.

    Expected format: ``<FEEDBACK_AGENT_1> ... </FEEDBACK_AGENT_1>`` (tolerant to ``FEEDBACK_AGENT_1:``). If the
    reply is not tagged, the text after the last ``FEEDBACK`` keyword is given to every solver.
    """
    text = _FB_CLOSE_RE.sub("", strip_think(text))
    markers = list(_FB_OPEN_RE.finditer(text))
    fb: Dict[int, str] = {}
    for j, m in enumerate(markers):
        end = markers[j + 1].start() if j + 1 < len(markers) else len(text)
        seg = text[m.end():end].strip(" \n*:")
        if seg:
            fb[int(m.group(1))] = seg  # later (final) blocks override earlier prose mentions
    if not fb:
        kw = list(re.finditer(r"FEEDBACK", text, flags=re.IGNORECASE))
        if kw:
            seg = text[kw[-1].end():].strip(" :*\n")
        else:
            seg = text.strip().splitlines()[-1] if text.strip() else ""
        fb = {k: (seg or _NO_CHANGE) for k in range(1, K + 1)}
    out = []
    for k in range(1, K + 1):
        seg = fb.get(k) or fb.get(1) or _NO_CHANGE
        lines = [ln.strip() for ln in seg.splitlines() if ln.strip()][:max_lines]
        out.append("\n".join(lines))
    return out


# --------------------------------------------------------------------------------------
# Hyper-parameter agent  E_H(.)  (eq. (9))
# --------------------------------------------------------------------------------------
@dataclass
class HyperOutput:
    mu: float
    gamma: float
    reasoning: str


def parse_hyperparams(text: str, trace_max_chars: int = 3000) -> HyperOutput:
    text_clean = strip_think(text)
    from ..core.parsing import find_json_objects

    for obj, start, _ in reversed(find_json_objects(text_clean)):
        low = {str(k).lower(): v for k, v in obj.items()}
        mu = low.get("mu", low.get("μ", low.get("step_size", low.get("learning_rate"))))
        gamma = low.get("gamma", low.get("γ", low.get("momentum")))
        if mu is None or gamma is None:
            continue
        try:
            mu_f, gamma_f = float(mu), float(gamma)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(mu_f) and np.isfinite(gamma_f)) or mu_f <= 0 or not (0.0 <= gamma_f < 1.0):
            raise ParseError(f"invalid hyper-parameters mu={mu_f}, gamma={gamma_f}: need mu > 0 and 0 <= gamma < 1")
        return HyperOutput(mu_f, gamma_f, reasoning_before(text_clean, start, trace_max_chars))
    raise ParseError('no JSON object with keys "mu" and "gamma" found in the reply')
