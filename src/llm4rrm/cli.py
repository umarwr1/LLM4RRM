"""Shared command-line helpers for the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .config import Config
from .utils import load_env_file


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/beamforming_default.yaml", help="YAML config (default: configs/beamforming_default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.key=value",
                   help="override a config value, e.g. --set framework.T=3 (repeatable)")
    p.add_argument("--provider", default=None, help="llm.provider preset (gpt-5-mini, gpt-4.1-mini, grok-4-fast, gemini-2.5-flash, llama-4-scout, custom, mock)")
    p.add_argument("--model", default=None, help="override llm.model")
    p.add_argument("--n-instances", type=int, default=None, help="override experiment.n_instances")
    p.add_argument("--output-dir", default=None, help="override experiment.output_dir")
    p.add_argument("--env-file", default=".env", help="git-ignored file with API keys (KEY=VALUE); never committed")


def build_config(args: argparse.Namespace, extra: Optional[List[str]] = None) -> Config:
    load_env_file(args.env_file)
    cfg = Config.from_yaml(args.config)
    ov = list(args.overrides)
    if args.provider:
        ov.append(f"llm.provider={args.provider}")
    if args.model:
        ov.append(f"llm.model={args.model}")
    if args.n_instances is not None:
        ov.append(f"experiment.n_instances={args.n_instances}")
    if args.output_dir:
        ov.append(f"experiment.output_dir={args.output_dir}")
    return cfg.with_overrides(ov + list(extra or []))
