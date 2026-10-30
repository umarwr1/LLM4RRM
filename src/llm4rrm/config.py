"""Typed configuration for LLM4RRM (beamforming use case).

All experiment knobs live here. Configs are plain dataclasses that can be loaded from YAML and
overridden from the command line with dotted ``section.key=value`` strings.

No secrets are ever stored in configs: API keys are read from environment variables only
(see :mod:`llm4rrm.llm.providers`).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# --------------------------------------------------------------------------------------
# System (Sec. III + Sec. V-A.1 of the paper)
# --------------------------------------------------------------------------------------
@dataclass
class SystemConfig:
    """STAR-RIS aided MISO downlink (Use-Case Study I)."""

    M: int = 8  # BS antennas
    U: int = 4  # single-antenna users (U = Ut + Ur)
    N: int = 16  # STAR-RIS elements
    n_transmit_users: Optional[int] = None  # Ut; default U // 2 (first Ut users are on the transmission side)

    p_max_w: float = 1.0  # BS power budget Pmax [W]
    r_min_mbps: float = 3.0  # per-user minimum rate Rmin [Mbps]
    bandwidth_hz: float = 5e6  # rate = BW * log2(1 + SINR); 5 MHz (Rmin 3 Mbps = 0.6 bps/Hz)
    noise_power_dbm: float = -170.0  # sigma_u^2 (paper: -170 dBm)

    # Channel model (paper Sec. V-A.1) and array geometry.
    d_bs_ris_m: float = 200.0  # BS - STAR-RIS distance
    user_radius_m: float = 50.0  # users uniform in a disc of this radius around the STAR-RIS
    rician_factor: float = 10.0  # linear Rician K-factor (paper: epsilon = 10)
    pathloss_intercept_db: float = 35.6  # L(d) = 35.6 + 22 log10(d) [dB]
    pathloss_slope_db: float = 22.0
    min_distance_m: float = 1.0  # clamp for user distance in the path-loss law
    fading: str = "rician"  # "rician" | "rayleigh"
    antenna_spacing_wavelengths: float = 0.5
    bs_height_m: float = 25.0
    ris_height_m: float = 10.0
    user_height_m: float = 1.5

    def __post_init__(self) -> None:
        if self.n_transmit_users is None:
            self.n_transmit_users = self.U // 2
        if not (0 <= self.n_transmit_users <= self.U):
            raise ValueError("n_transmit_users must be in [0, U]")
        if self.fading not in ("rician", "rayleigh"):
            raise ValueError("fading must be 'rician' or 'rayleigh'")

    # -- derived quantities ---------------------------------------------------------
    @property
    def noise_power_w(self) -> float:
        return 10.0 ** ((self.noise_power_dbm - 30.0) / 10.0)

    @property
    def r_min_bps_hz(self) -> float:
        """Rmin as spectral efficiency; the projection surrogate V works in bps/Hz."""
        return self.r_min_mbps * 1e6 / self.bandwidth_hz

    @property
    def mbps_per_bps_hz(self) -> float:
        return self.bandwidth_hz / 1e6

    @property
    def n_reflect_users(self) -> int:
        return self.U - int(self.n_transmit_users)


# --------------------------------------------------------------------------------------
# LLM backend
# --------------------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Which LLM to call. ``provider`` is a preset name (see ``llm/providers.py``) or ``custom``/``mock``.

    API keys are *never* part of the config: they are read from the environment variable named by
    ``api_key_env``.
    """

    provider: str = "gpt-5-mini"  # preset key, "custom" (OpenAI-compatible endpoint) or "mock"
    model: Optional[str] = None  # overrides the preset's model id
    base_url: Optional[str] = None  # overrides the preset's endpoint
    api_key_env: Optional[str] = None  # overrides the preset's env-var name
    temperature: float = 1.0  # paper default: 1
    max_tokens: int = 12000  # per agent call (paper: 12,000)
    timeout_s: float = 600.0
    api_max_retries: int = 3  # transport-level retries handled by the SDK


# --------------------------------------------------------------------------------------
# Multi-agent framework (Sec. II-A)
# --------------------------------------------------------------------------------------
@dataclass
class FrameworkConfig:
    T: int = 5  # reasoning iterations (paper: T = 5)
    num_solvers: int = 2  # K in {1, 2}
    use_evaluator: bool = True
    n_fewshot: int = 3  # |S_M| (paper: 3)
    similarity: str = "frobenius"  # phi(.): frobenius | cosine | random
    prompt_level: int = 7  # 1..7, cumulative prompt styles of Table 4 (7 = full LLM4RRM)
    parse_retries: int = 3  # re-asks when an agent reply cannot be parsed
    trace_max_chars: int = 3000  # reasoning-trace length forwarded to the evaluator
    final_selection: str = "last"  # y^(T) = last iterate ("last") or best iterate so far ("best")
    stop_rel_tol: float = 1e-3  # "performance has converged" test used for early stopping at feasibility
    update_memory: bool = False  # Alg. 1 line 10 (add solved instance to D). Off while benchmarking.
    csi_error_eta: float = 0.0  # eta in Table 7 (error variance = eta * sigma^2)
    projection_uses_true_csi: bool = True  # CSI-error study: LLM sees perturbed CSI, projection/evaluation use true CSI

    def __post_init__(self) -> None:
        if self.num_solvers not in (1, 2):
            raise ValueError("num_solvers must be 1 or 2 for the beamforming use case")
        if not (1 <= self.prompt_level <= 7):
            raise ValueError("prompt_level must be in 1..7")
        if self.final_selection not in ("last", "best"):
            raise ValueError("final_selection must be 'last' or 'best'")
        if self.similarity not in ("frobenius", "cosine", "random"):
            raise ValueError("similarity must be frobenius | cosine | random")

    # -- effective feature flags implied by the prompt level (Table 4 / Table 12) -----
    @property
    def eff_num_solvers(self) -> int:
        return self.num_solvers if self.prompt_level >= 6 else 1

    @property
    def eff_use_evaluator(self) -> bool:
        return bool(self.use_evaluator) and self.prompt_level >= 7

    @property
    def eff_n_fewshot(self) -> int:
        return self.n_fewshot if self.prompt_level >= 2 else 0

    @property
    def style_physics(self) -> bool:
        return self.prompt_level >= 3

    @property
    def style_confidence(self) -> bool:
        return self.prompt_level >= 4

    @property
    def style_cot(self) -> bool:
        return self.prompt_level >= 5


# --------------------------------------------------------------------------------------
# LLM-guided adaptive differentiable projection (Sec. II-A.e, Sec. III-C)
# --------------------------------------------------------------------------------------
@dataclass
class ProjectionConfig:
    L: int = 5  # max number of hyper-parameter tuning rounds (LLM_H calls)
    I: int = 200  # max number of projection steps
    probe_steps: int = 10  # steps run with theta^(l) before LLM_H is consulted (Appendix D: "10 projection steps")
    mu0: float = 0.01  # theta^(0): step size (paper default)
    gamma0: float = 0.9  # theta^(0): momentum (paper default)
    mu_bounds: Tuple[float, float] = (1e-4, 10.0)  # clamp for LLM_H suggestions
    gamma_bounds: Tuple[float, float] = (0.0, 0.99)
    hyper_mode: str = "llm"  # "llm" (LLM_H) | "fixed" (theta^(0) for the whole run)
    patience: int = 10  # post-feasibility convergence window (steps)
    conv_rel_tol: float = 1e-3  # stop when sum-rate gain over `patience` steps < tol * sum-rate
    conv_fraction: float = 0.99  # S_conv: first step reaching this fraction of the final sum-rate
    max_records_in_prompt: int = 20  # per-step records shown to LLM_H
    hyper_include_channels: bool = True  # H_proj^(0) = {x, T_H, y^(T)} includes x
    hyper_retries: int = 2

    def __post_init__(self) -> None:
        self.mu_bounds = tuple(self.mu_bounds)  # type: ignore[assignment]
        self.gamma_bounds = tuple(self.gamma_bounds)  # type: ignore[assignment]
        if self.hyper_mode not in ("llm", "fixed"):
            raise ValueError("hyper_mode must be 'llm' or 'fixed'")


# --------------------------------------------------------------------------------------
# Experiment / evaluation
# --------------------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    n_instances: int = 100  # paper: 100 channel realisations per configuration
    seed: int = 0  # test-set seed
    memory_bank_path: Optional[str] = None  # .npz produced by scripts/generate_dataset.py
    memory_bank_size: int = 200
    memory_bank_seed: int = 1234  # must differ from `seed` so the bank never contains test channels
    output_dir: str = "results/run"
    save_transcripts: bool = False
    repeats: int = 1  # independent LLM4RRM runs per channel (Table 9 uses 10)


@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # ------------------------------------------------------------------ io helpers --
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = copy.deepcopy(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown config sections: {sorted(unknown)}")
        kwargs = {}
        for f in dataclasses.fields(cls):
            sub = d.get(f.name, {}) or {}
            sub_cls = f.default_factory().__class__  # type: ignore[misc]
            sub_known = {sf.name for sf in dataclasses.fields(sub_cls)}
            bad = set(sub) - sub_known
            if bad:
                raise KeyError(f"Unknown keys in config section '{f.name}': {sorted(bad)}")
            kwargs[f.name] = sub_cls(**sub)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(_plain(self.to_dict()), fh, sort_keys=False)

    def with_overrides(self, overrides: List[str] | Dict[str, Any] | None) -> "Config":
        """Return a new Config with ``section.key=value`` overrides applied (values parsed as YAML)."""
        d = self.to_dict()
        items: List[Tuple[str, Any]]
        if isinstance(overrides, dict):
            items = list(overrides.items())
        else:
            items = []
            for s in overrides or []:
                if "=" not in s:
                    raise ValueError(f"Override must look like section.key=value, got '{s}'")
                k, v = s.split("=", 1)
                items.append((k.strip(), yaml.safe_load(v)))
        for key, val in items:
            if "." not in key:
                raise ValueError(f"Override key must be 'section.key', got '{key}'")
            sec, name = key.split(".", 1)
            if sec not in d or name not in d[sec]:
                raise KeyError(f"Unknown config key '{key}'")
            d[sec][name] = val
        return Config.from_dict(d)


def _plain(obj: Any) -> Any:
    """Make tuples/np types YAML-safe."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, float) and math.isfinite(obj):
        return float(obj)
    return obj
