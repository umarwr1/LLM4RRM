"""Small shared helpers: .env loading (keys only ever come from the environment), seeding, IO."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_env_file(path: str | Path = ".env") -> bool:
    """Load KEY=VALUE pairs from a local, git-ignored ``.env`` into ``os.environ`` (existing variables win).

    Uses python-dotenv when installed, otherwise a tiny built-in parser. Values are never printed.
    """
    p = Path(path)
    if not p.is_file():
        return False
    try:
        from dotenv import load_dotenv

        return bool(load_dotenv(p, override=False))
    except ImportError:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return True


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def append_jsonl(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")
