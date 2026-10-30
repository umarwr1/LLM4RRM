import re
from pathlib import Path

import pytest

from llm4rrm.config import Config

ROOT = Path(__file__).resolve().parents[1]


def test_overrides_and_yaml_roundtrip(tmp_path):
    cfg = Config().with_overrides(["framework.T=3", "system.r_min_mbps=5", "projection.mu_bounds=[0.001, 1]"])
    assert cfg.framework.T == 3 and cfg.system.r_min_mbps == 5 and cfg.projection.mu_bounds == (0.001, 1)
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    assert Config.from_yaml(p).to_dict() == cfg.to_dict()


def test_bad_override_keys_rejected():
    with pytest.raises(KeyError):
        Config().with_overrides(["framework.nope=1"])
    with pytest.raises(ValueError):
        Config().with_overrides(["framework.prompt_level=9"])


def test_default_yaml_matches_dataclass_defaults():
    assert Config.from_yaml(ROOT / "configs" / "beamforming_default.yaml").to_dict() == Config().to_dict()


@pytest.mark.parametrize("path", sorted((ROOT / "configs" / "experiments").glob("*.yaml")))
def test_sweep_configs_are_valid(path):
    import yaml
    spec = yaml.safe_load(path.read_text())
    base = Config().with_overrides(spec.get("base") or {})
    for label, ov in spec["rows"].items():
        base.with_overrides(dict(ov or {}))


SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_\-]{20,}", r"AIza[0-9A-Za-z_\-]{30,}", r"xai-[A-Za-z0-9]{20,}",
    r"(?i)api[_-]?key\s*=\s*[\"'][A-Za-z0-9_\-]{16,}[\"']", r"gsk_[A-Za-z0-9]{20,}",
]


def test_no_secrets_in_repository():
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "results", "data", ".venv", "venv"}
    offenders = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in skip_dirs for part in p.relative_to(ROOT).parts):
            continue
        if p.name == ".env" or p.suffix in {".npz", ".pyc", ".png", ".pdf"}:
            continue
        if p.name == "test_config_and_secrets.py":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            if re.search(pat, text):
                offenders.append(f"{p.relative_to(ROOT)} matches {pat}")
    assert not offenders, offenders
