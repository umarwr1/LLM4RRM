"""Experiment driver: memory bank, test channels, evaluation loop, result files."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..config import Config
from ..llm.base import LLMClient, LLMError
from ..utils import append_jsonl, write_json
from .channels import generate_channels
from .dataset import MemoryBank, generate_memory_bank
from .framework import LLM4RRMBeamformer
from .metrics import InstanceResult, aggregate, format_row, repeat_std


def default_bank_path(cfg: Config) -> Path:
    s = cfg.system
    name = f"bank_M{s.M}_N{s.N}_U{s.U}_R{s.r_min_mbps:g}_P{s.p_max_w:g}_n{cfg.experiment.memory_bank_size}_s{cfg.experiment.memory_bank_seed}.npz"
    return Path("data") / name


def load_or_build_memory(cfg: Config, verbose: bool = True) -> Optional[MemoryBank]:
    """Memory bank D for this system configuration (generated without any LLM if it does not exist yet)."""
    if cfg.framework.eff_n_fewshot == 0:
        return None
    ex = cfg.experiment
    if ex.memory_bank_seed == ex.seed:
        raise ValueError("experiment.memory_bank_seed must differ from experiment.seed (no test channels in the bank)")
    path = Path(ex.memory_bank_path) if ex.memory_bank_path else default_bank_path(cfg)
    if path.is_file():
        bank = MemoryBank.load(path)
        if verbose:
            print(f"[memory] loaded {len(bank)} examples from {path}")
        return bank
    if verbose:
        print(f"[memory] {path} not found: generating {ex.memory_bank_size} feasible examples (no LLM involved) ...")
    bank = generate_memory_bank(cfg.system, ex.memory_bank_size, ex.memory_bank_seed, verbose=verbose)
    bank.save(path)
    if verbose:
        print(f"[memory] saved to {path}")
    return bank


def run_experiment(
    cfg: Config,
    client: LLMClient,
    memory: Optional[MemoryBank],
    out_dir: Optional[str | Path] = None,
    progress: Optional[Callable[[int, int, InstanceResult], None]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run LLM4RRM on ``experiment.n_instances`` fresh channel realisations (x ``repeats``)."""
    ex = cfg.experiment
    out = Path(out_dir or ex.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out / "config.yaml")
    jsonl = out / "instances.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    G_all, H_all = generate_channels(cfg.system, ex.n_instances, ex.seed)
    solver = LLM4RRMBeamformer(cfg, client, memory, seed=ex.seed)
    results: List[InstanceResult] = []
    total = ex.n_instances * ex.repeats
    done = 0
    t0 = time.time()
    for i in range(ex.n_instances):
        for rep in range(ex.repeats):
            try:
                res = solver.solve(G_all[i], H_all[i], index=i, repeat=rep, csi_seed=ex.seed * 100003 + i)
            except LLMError as exc:  # backend failure: record and go on (never silently drop)
                warnings.warn(f"instance {i} (repeat {rep}) failed: {exc}")
                res = InstanceResult(index=i, repeat=rep, ok=False, error=str(exc))
            results.append(res)
            append_jsonl(jsonl, res.to_dict())
            done += 1
            if progress:
                progress(done, total, res)
            elif verbose:
                r = res
                msg = (f"[{done}/{total}] inst {i}.{rep}: " + (
                    f"pre {r.pre['sum_rate_mbps']:.2f} Mbps/{r.pre['n_violations']} viol -> post {r.post['sum_rate_mbps']:.2f} Mbps/"
                    f"{r.post['n_violations']} viol, S_ZV={r.s_zv}, calls={r.llm.get('calls', 0):.0f}, {r.time_total_s:.1f}s"
                    if r.ok else f"FAILED ({r.error})"))
                print(msg, flush=True)
    agg = aggregate(results)
    summary: Dict[str, object] = dict(aggregate=agg, elapsed_s=time.time() - t0, n_instances=ex.n_instances, repeats=ex.repeats)
    if ex.repeats > 1:
        summary["repeat_std"] = repeat_std(results)
    write_json(out / "summary.json", summary)
    if verbose:
        print(format_row(cfg.llm.provider, agg))
        if ex.repeats > 1:
            print("repeat statistics:", summary["repeat_std"])
        if getattr(client, "is_mock", False):
            print("Mock LLM used: synthetic outputs (pipeline test).")
    return summary
