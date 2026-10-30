#!/usr/bin/env python
"""Run LLM4RRM (Use-Case I: STAR-RIS beamforming) on fresh channel realisations.

Examples
    python scripts/run_beamforming.py --provider gpt-5-mini --n-instances 100
    python scripts/run_beamforming.py --provider mock --n-instances 3        # offline pipeline test
    python scripts/run_beamforming.py --set framework.prompt_level=5 --set system.r_min_mbps=4

API keys are read from environment variables (or a git-ignored .env); see .env.example.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4rrm.beamforming.runner import load_or_build_memory, run_experiment  # noqa: E402
from llm4rrm.cli import add_common_args, build_config  # noqa: E402
from llm4rrm.llm.factory import create_llm_client  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--transcripts", action="store_true", help="store full agent transcripts in instances.jsonl")
    args = ap.parse_args()
    cfg = build_config(args, ["experiment.save_transcripts=true"] if args.transcripts else None)
    client = create_llm_client(cfg.llm, seed=cfg.experiment.seed)
    memory = load_or_build_memory(cfg)
    run_experiment(cfg, client, memory)


if __name__ == "__main__":
    main()
