"""Minimal LLM client interface shared by the real (OpenAI-compatible) and the mock backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

Message = Dict[str, str]  # {"role": "system" | "user" | "assistant", "content": str}


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot deliver a reply (network, auth, quota, ...)."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class LLMClient(ABC):
    """Chat-completion client.  Subclasses implement :meth:`_complete`; usage statistics are tracked here."""

    is_mock: bool = False

    def __init__(self) -> None:
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_s = 0.0

    @abstractmethod
    def _complete(self, messages: List[Message], temperature: Optional[float], max_tokens: Optional[int]) -> LLMResponse:
        ...

    def chat(
        self, messages: List[Message], temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self._complete(messages, temperature, max_tokens)
        resp.latency_s = resp.latency_s or (time.perf_counter() - t0)
        self.n_calls += 1
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.latency_s += resp.latency_s
        return resp

    def usage(self) -> Dict[str, float]:
        return dict(calls=self.n_calls, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens, latency_s=self.latency_s)
