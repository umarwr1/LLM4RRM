"""A stateful chat agent with its own private memory (paper Remark in Sec. II-A).

Each agent (solver k, evaluator, hyper-parameter agent) owns an independent message history, which realises
the historical contexts H_k^(t), H_E^(t), H_proj^(l) of eqs. (4), (8), (11): the system prompt carries the task
description T, the first user turn carries x and the few-shot set, and every later turn appends the new feedback
together with the agent's previous answers (solutions + reasoning traces).  No agent can read another's history.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from ..llm.base import LLMClient, LLMError, Message
from .parsing import ParseError

T = TypeVar("T")


class AgentError(RuntimeError):
    """The agent could not produce a parsable answer within the allowed number of attempts."""


class ChatAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        client: LLMClient,
        parse_retries: int = 3,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.name = name
        self.client = client
        self.parse_retries = parse_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages: List[Message] = [{"role": "system", "content": system_prompt}]
        self.transcript: List[Dict[str, Any]] = []  # full log of (user, assistant) turns incl. failed attempts
        self.n_failed_parses = 0

    def ask(self, user_text: str, parser: Callable[[str], T], tag: str = "") -> Tuple[T, str]:
        """Send ``user_text`` (appended to the private history) and parse the reply with ``parser``.

        Unparsable replies are re-asked (up to ``parse_retries`` times) with an explicit error message; only the
        final successful exchange is stored in the persistent history. Raises :class:`AgentError` on failure and
        propagates :class:`LLMError` for backend problems.
        """
        pending = self.messages + [{"role": "user", "content": user_text}]
        last_err: Optional[Exception] = None
        for attempt in range(self.parse_retries + 1):
            resp = self.client.chat(pending, temperature=self.temperature, max_tokens=self.max_tokens)
            self.transcript.append(dict(tag=tag, attempt=attempt, user=pending[-1]["content"], assistant=resp.text,
                                        prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens))
            try:
                parsed = parser(resp.text)
            except ParseError as exc:
                last_err = exc
                self.n_failed_parses += 1
                pending = pending + [
                    {"role": "assistant", "content": resp.text},
                    {"role": "user", "content": f"Your reply could not be used ({exc}). Reply again and strictly follow the "
                                                 f"required output format (final answer as one JSON object in a ```json block)."},
                ]
                continue
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": resp.text})
            return parsed, resp.text
        raise AgentError(f"{self.name}: no parsable answer after {self.parse_retries + 1} attempts ({last_err})")
