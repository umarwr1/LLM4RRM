#!/usr/bin/env python
"""Run a table-style sweep defined in configs/experiments/*.yaml.

A sweep file has ``base`` (overrides applied to every row) and ``rows`` (label -> overrides).
Each row is a full run_experiment() call; results go to <output_root>/<sweep>/<row>/ and a combined
table is printed and written to summary.json.
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4rrm.beamforming.metrics import format_row  # noqa: E402
from llm4rrm.beamforming.runner import load_or_build_memory, run_experiment  # noqa: E402
from llm4rrm.cli import add_common_args, build_config  # noqa: E402
from llm4rrm.llm.factory import create_llm_client  # noqa: E402
from llm4rrm.utils import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("sweep", help="path to a sweep YAML, e.g. configs/experiments/table4_prompt_levels.yaml")
    ap.add_argument("--output-root", default="results", help="root directory for sweep outputs")
    ap.add_argument("--only", default=None, help="regex: run only rows whose label matches")
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))
    name = Path(args.sweep).stem
    base_cfg = build_config(args, [f"{k}={v}" for k, v in (spec.get("base") or {}).items()])
    summary = {}
    for label, ov in spec["rows"].items():
        if args.only and not re.search(args.only, label):
            continue
        cfg = base_cfg.with_overrides(dict(ov or {}))
        out_dir = Path(args.output_root) / name / re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        cfg = cfg.with_overrides({"experiment.output_dir": str(out_dir)})
        print(f"\n=== {name} :: {label} ===", flush=True)
        client = create_llm_client(cfg.llm, seed=cfg.experiment.seed)
        res = run_experiment(cfg, client, load_or_build_memory(cfg), verbose=False)
        summary[label] = res
    print(f"\n=== {name} ===")
    for label, res in summary.items():
        print(format_row(label, res["aggregate"]))
    write_json(Path(args.output_root) / name / "summary.json", summary)


if __name__ == "__main__":
    main()
