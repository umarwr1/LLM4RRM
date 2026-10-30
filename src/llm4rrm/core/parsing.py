"""Generic helpers to extract structured data from free-form LLM text (the extraction function E(.))."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


class ParseError(ValueError):
    """The LLM reply could not be converted into the requested structure."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_DECODER = json.JSONDecoder()


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by some open-weight reasoning models."""
    return _THINK_RE.sub("", text).strip()


def find_json_objects(text: str) -> List[Tuple[Dict[str, Any], int, int]]:
    """All top-level JSON objects in ``text`` as (obj, start, end)."""
    out: List[Tuple[Dict[str, Any], int, int]] = []
    pos = 0
    while True:
        i = text.find("{", pos)
        if i < 0:
            break
        try:
            obj, end = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            pos = i + 1
            continue
        if isinstance(obj, dict):
            out.append((obj, i, end))
        pos = end
    return out


def last_json_with_keys(text: str, required: List[str]) -> Tuple[Dict[str, Any], int]:
    """Last JSON object whose (lower-cased) keys contain all ``required``; returns (obj_lowercased, start_index)."""
    text = strip_think(text)
    for obj, start, _ in reversed(find_json_objects(text)):
        low = {str(k).lower(): v for k, v in obj.items()}
        if all(k in low for k in required):
            return low, start
    raise ParseError(f"no JSON object with keys {required} found in the reply")


def reasoning_before(text: str, start: int, max_chars: int) -> str:
    """Reasoning trace RT: the prose before the final JSON object (code fences removed, tail-truncated)."""
    text = strip_think(text)
    head = text[:start]
    head = re.sub(r"```(?:json)?", "", head, flags=re.IGNORECASE).strip()
    if len(head) > max_chars:
        head = "..." + head[-max_chars:]
    return head
