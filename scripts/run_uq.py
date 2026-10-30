#!/usr/bin/env python
"""Output-based uncertainty quantification (paper Sec. IV-B): D_UQ (eq. 13) and MAD_UQ (eq. 14).

For Psi channel realisations the identical prompt is run Lambda times; the pre-projection decision vectors
y^(T) are compared per variable group.  The projection is disabled (projection.I=0) because UQ analyses the
LLM output itself.  Paper setting: Psi = 20, Lambda = 10 (=> 200 full multi-agent runs; costly with real APIs).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4rrm.beamforming.channels import generate_channels  # noqa: E402
from llm4rrm.beamforming.framework import LLM4RRMBeamformer  # noqa: E402
from llm4rrm.beamforming.runner import load_or_build_memory  # noqa: E402
from llm4rrm.cli import add_common_args, build_config  # noqa: E402
from llm4rrm.core.uq import d_uq, mad_uq  # noqa: E402
from llm4rrm.llm.factory import create_llm_client  # noqa: E402
from llm4rrm.utils import write_json  # noqa: E402

GROUPS = {
    "W (beamforming)": ("Wr", "Wi"),
    "alpha": ("alpha",),
    "a (energy split)": ("a",),
    "phi_t": ("phi_t",),
    "phi_r": ("phi_r",),
}


def _vec(y: dict, keys) -> np.ndarray:
    return np.concatenate([np.asarray(y[k], dtype=float).ravel() for k in keys])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--psi", type=int, default=20, help="number of channel realisations (paper: 20)")
    ap.add_argument("--lam", type=int, default=10, help="outputs per realisation (paper: 10)")
    args = ap.parse_args()
    cfg = build_config(args, ["projection.I=0", "experiment.repeats=1"])
    client = create_llm_client(cfg.llm, seed=cfg.experiment.seed)
    memory = load_or_build_memory(cfg)
    G, H = generate_channels(cfg.system, args.psi, cfg.experiment.seed)
    solver = LLM4RRMBeamformer(cfg, client, memory, seed=cfg.experiment.seed)

    samples = {g: [] for g in GROUPS}
    for p in range(args.psi):
        per = {g: [] for g in GROUPS}
        for lam in range(args.lam):
            res = solver.solve(G[p], H[p], index=p, repeat=lam)
            for g, keys in GROUPS.items():
                per[g].append(_vec(res.y_T, keys))
        for g in GROUPS:
            samples[g].append(per[g])
        print(f"[{p + 1}/{args.psi}] done", flush=True)

    out = {}
    for g in GROUPS:
        arr = np.asarray(samples[g])  # (Psi, Lambda, dim)
        out[g] = dict(D_UQ=d_uq(arr), MAD_UQ=mad_uq(arr))
        print(f"{g:<18} D_UQ={out[g]['D_UQ']:.4f}  MAD_UQ={out[g]['MAD_UQ']:.4f}")
    out_dir = Path(args.output_dir or cfg.experiment.output_dir)
    write_json(out_dir / "uq.json", dict(psi=args.psi, lam=args.lam, prompt_level=cfg.framework.prompt_level, metrics=out))
    if getattr(client, "is_mock", False):
        print("Mock LLM used: synthetic outputs (pipeline test).")


if __name__ == "__main__":
    main()
