#!/usr/bin/env python
"""Generate the few-shot memory bank D (no LLM, no API key needed).

Random initialisation + the differentiable projection of eq. (25) produce feasible, sum-rate-refined
examples for the configured system (M, N, U, Rmin, Pmax).  The default output path is the one the
runner looks for, so a bank is also built automatically on first use.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4rrm.beamforming.dataset import generate_memory_bank  # noqa: E402
from llm4rrm.beamforming.runner import default_bank_path  # noqa: E402
from llm4rrm.cli import add_common_args, build_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--size", type=int, default=None, help="number of examples (default experiment.memory_bank_size)")
    ap.add_argument("--seed", type=int, default=None, help="channel seed (default experiment.memory_bank_seed)")
    ap.add_argument("--out", default=None, help="output .npz (default data/bank_M*_N*_U*_R*_P*_n*_s*.npz)")
    args = ap.parse_args()
    cfg = build_config(args)
    size = args.size or cfg.experiment.memory_bank_size
    seed = args.seed if args.seed is not None else cfg.experiment.memory_bank_seed
    if seed == cfg.experiment.seed:
        ap.error("bank seed must differ from the test seed (experiment.seed)")
    cfg = cfg.with_overrides({"experiment.memory_bank_size": size, "experiment.memory_bank_seed": seed})
    out = Path(args.out) if args.out else default_bank_path(cfg)
    bank = generate_memory_bank(cfg.system, size, seed)
    bank.save(out)
    print(f"saved {len(bank)} examples to {out}")
    print(f"mean sum-rate of bank examples: {float(bank.sum_rate_mbps.mean()):.2f} Mbps")


if __name__ == "__main__":
    main()
