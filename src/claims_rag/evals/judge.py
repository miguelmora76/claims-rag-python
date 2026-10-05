from __future__ import annotations

import re

from ..config import Settings
from ..llm import LlmClient, LlmRequest

_SCORE = re.compile(r'"score"\s*:\s*([1-5])')
_SYSTEM = """\
You grade answers from a retrieval-augmented assistant. Given a context, a question and an answer,
score how well every claim in the answer is supported by the context:
5 = fully supported, 3 = partly supported, 1 = unsupported or contradicts the context.
Reply with JSON only: {"score": <1-5>, "reason": "<short>"}
"""


class LlmJudge:
    """LLM-as-judge groundedness score (1-5). Only meaningful with a real model, so the offline run skips it."""

    def __init__(self, llm: LlmClient, settings: Settings):
        self._llm, self._settings = llm, settings

    def score(self, context: str, question: str, answer: str) -> int | None:
        user = f"<context>\n{context}\n</context>\n<question>{question}</question>\n<answer>{answer}</answer>"
        reply = self._llm.complete(LlmRequest(self._settings.judge_model, _SYSTEM, user, 200)).text
        m = _SCORE.search(reply)
        return int(m.group(1)) if m else None
