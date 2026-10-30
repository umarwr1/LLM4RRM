"""OpenAI-compatible chat client (OpenAI, xAI, Gemini's OpenAI endpoint, vLLM, LM Studio, ...).

Credentials are read from environment variables only and are never logged, printed or stored.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

from .base import LLMClient, LLMError, LLMResponse, Message

_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class MissingAPIKeyError(LLMError):
    pass


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key_env: str,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 12000,
        timeout_s: float = 600.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__()
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise MissingAPIKeyError(
                f"Environment variable {api_key_env} is not set. Export it or put it in a local, git-ignored .env file "
                f"(see .env.example). Never hard-code API keys in the repository."
            )
        try:
            from openai import OpenAI  # lazy import: only needed for real runs
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'openai' package is required for real LLM calls: pip install openai") from exc
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self._reasoning = model.lower().startswith(_REASONING_PREFIXES)
        self._warned_temp = False
        self._client = OpenAI(api_key=key, base_url=base_url, timeout=timeout_s, max_retries=max_retries)
        self.base_url = base_url
        del key  # do not keep a second reference around

    def __repr__(self) -> str:  # never expose credentials
        return f"OpenAICompatibleClient(model={self.model!r}, base_url={self.base_url!r})"

    def _complete(self, messages: List[Message], temperature: Optional[float], max_tokens: Optional[int]) -> LLMResponse:
        temp = self.default_temperature if temperature is None else temperature
        limit = self.default_max_tokens if max_tokens is None else max_tokens
        kwargs = dict(model=self.model, messages=messages)
        if self._reasoning:
            # Reasoning models only accept the provider-default temperature (1) and use max_completion_tokens.
            kwargs["max_completion_tokens"] = limit
            if temp != 1.0 and not self._warned_temp:
                warnings.warn(f"{self.model} only supports temperature=1; requested {temp} is ignored.")
                self._warned_temp = True
        else:
            kwargs["max_tokens"] = limit
            kwargs["temperature"] = temp
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # SDK already retried transport errors
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
