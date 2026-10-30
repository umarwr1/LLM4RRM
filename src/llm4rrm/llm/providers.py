"""Provider presets for the LLM backbones evaluated in the paper (Sec. V-A.1, Table 6).

IMPORTANT: no credentials live in this repository.  Each preset only names the *environment variable*
that must contain the API key (optionally loaded from a git-ignored ``.env`` file).  All backends are
called through the OpenAI-compatible chat-completions protocol.

The model identifiers are the ones quoted in the paper.  Provider-side availability of dated/preview
snapshots changes over time; override with ``llm.model=...`` / ``llm.base_url=...`` when needed.
"""

from __future__ import annotations

from typing import Dict, Optional

PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    # name -> model id, base url (None = SDK default = OpenAI), env var holding the key, env var holding a base url
    "gpt-5-mini": dict(model="gpt-5-mini-2025-08-07", base_url=None, api_key_env="OPENAI_API_KEY", base_url_env=None),
    "gpt-4.1-mini": dict(model="gpt-4.1-mini-2025-04-14", base_url=None, api_key_env="OPENAI_API_KEY", base_url_env=None),
    "grok-4-fast": dict(model="grok-4-fast-reasoning", base_url="https://api.x.ai/v1", api_key_env="XAI_API_KEY", base_url_env=None),
    "gemini-2.5-flash": dict(
        model="gemini-2.5-flash-preview-09-2025",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY", base_url_env=None,
    ),
    # Llama-4-Scout is served by several vendors; point LLAMA_BASE_URL (and LLAMA_API_KEY) at yours.
    "llama-4-scout": dict(model="llama-4-scout-17b-16e-instruct", base_url=None, api_key_env="LLAMA_API_KEY", base_url_env="LLAMA_BASE_URL"),
    # Any OpenAI-compatible endpoint: set llm.model, llm.base_url and llm.api_key_env explicitly.
    "custom": dict(model=None, base_url=None, api_key_env="LLM_API_KEY", base_url_env="LLM_BASE_URL"),
}
