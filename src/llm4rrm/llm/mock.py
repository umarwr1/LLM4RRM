"""Offline mock LLM used for unit tests, CI and dry runs of the pipeline.

It reads the agent role and the problem dimensions from the prompts, so it exercises the real prompt-building
and parsing code, but its "solutions" are random: results obtained with this client say NOTHING about the
quality of LLM4RRM.  Use a real provider for paper reproduction.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import numpy as np

from .base import LLMClient, LLMResponse, Message


class MockLLMClient(LLMClient):
    is_mock = True

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    def __repr__(self) -> str:
        return f"MockLLMClient(seed={self.seed})"

    @staticmethod
    def _int(pattern: str, text: str, default: int) -> int:
        m = re.search(pattern, text)
        return int(m.group(1)) if m else default

    def _complete(self, messages: List[Message], temperature: Optional[float], max_tokens: Optional[int]) -> LLMResponse:
        rng = np.random.default_rng([self.seed, self.n_calls])
        system = messages[0]["content"]
        full = "\n".join(m["content"] for m in messages)
        role = (re.match(r"You are ([A-Z\-]+)", system) or [None, ""])[1]
        # dimensions come from the "SYSTEM PARAMETERS x" line: "M = 8 BS antennas, N = 16 STAR-RIS elements, U = 4 users"
        M = self._int(r"M = (\d+) BS antennas", full, 4)
        N = self._int(r"N = (\d+) STAR-RIS elements", full, 8)
        U = self._int(r"U = (\d+) users", full, 4)
        n_chars = len(full)

        def bf():
            return {"Wr": rng.standard_normal((M, U)).round(4).tolist(), "Wi": rng.standard_normal((M, U)).round(4).tolist(),
                    "alpha": round(float(rng.uniform(0.0, 2.0)), 4)}

        def ris():
            return {"a": rng.standard_normal(N).round(4).tolist(), "phi_t": rng.uniform(0, 2 * np.pi, N).round(4).tolist(),
                    "phi_r": rng.uniform(0, 2 * np.pi, N).round(4).tolist()}

        if role == "OPTIBEAM-STAR":
            body = "I align each beam with the strongest effective channel.\nConfidence: moderate.\n```json\n" + json.dumps(bf()) + "\n```"
        elif role == "OPTIRIS-STAR":
            body = "I set the phases to combine the cascaded paths coherently.\nConfidence: moderate.\n```json\n" + json.dumps(ris()) + "\n```"
        elif role == "OPTISOLVE-STAR":
            body = "Joint design.\n```json\n" + json.dumps({**bf(), **ris()}) + "\n```"
        elif role == "OPTEVAL":
            k = self._int(r"supervise (\d+) solver", system, 1)
            body = "<THOUGHT PROCESS> Some users are weak.\n" + "\n".join(
                f"<FEEDBACK_AGENT_{i}>\nIncrease the power of the weakest user's block.\n</FEEDBACK_AGENT_{i}>" for i in range(1, k + 1))
        elif role == "HYPEAGENT":
            mu = float(np.exp(rng.uniform(np.log(0.05), np.log(1.0))))
            body = "Convergence is slow; increase mu.\n```json\n" + json.dumps({"mu": round(mu, 4), "gamma": 0.85}) + "\n```"
        else:
            body = "(mock) unknown role"
        return LLMResponse(text=body, prompt_tokens=n_chars // 4, completion_tokens=len(body) // 4, latency_s=1e-4)
