"""Build an :class:`LLMClient` from an :class:`~llm4rrm.config.LLMConfig`."""

from __future__ import annotations

import os

from ..config import LLMConfig
from .base import LLMClient, LLMError
from .mock import MockLLMClient
from .openai_compat import OpenAICompatibleClient
from .providers import PRESETS


def create_llm_client(cfg: LLMConfig, seed: int = 0) -> LLMClient:
    if cfg.provider == "mock":
        return MockLLMClient(seed=seed)
    if cfg.provider not in PRESETS:
        raise LLMError(f"Unknown llm.provider '{cfg.provider}'. Choose one of {sorted(PRESETS) + ['mock']}.")
    preset = PRESETS[cfg.provider]
    model = cfg.model or preset["model"]
    if not model:
        raise LLMError("llm.model must be set for provider 'custom'")
    base_url = cfg.base_url or preset["base_url"] or (os.environ.get(preset["base_url_env"]) if preset["base_url_env"] else None)
    api_key_env = cfg.api_key_env or preset["api_key_env"]
    return OpenAICompatibleClient(
        model=model, api_key_env=api_key_env, base_url=base_url, temperature=cfg.temperature,
        max_tokens=cfg.max_tokens, timeout_s=cfg.timeout_s, max_retries=cfg.api_max_retries,
    )
