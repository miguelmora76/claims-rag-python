from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

import anthropic


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, o: Usage) -> Usage:
        return Usage(
            self.input_tokens + o.input_tokens,
            self.output_tokens + o.output_tokens,
            self.cache_read_tokens + o.cache_read_tokens,
            self.cache_write_tokens + o.cache_write_tokens,
        )


@dataclass(frozen=True)
class LlmRequest:
    model: str
    system: str
    user: str
    max_tokens: int
    effort: str | None = None


@dataclass(frozen=True)
class LlmResponse:
    text: str
    usage: Usage
    model: str
    latency_ms: int


class LlmError(Exception):
    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class LlmClient(Protocol):
    def complete(self, request: LlmRequest) -> LlmResponse: ...


class ClaudeLlmClient:
    """Calls Claude through the official Anthropic SDK. The SDK retries 408/409/429/5xx itself."""

    def __init__(self, client: anthropic.Anthropic | None = None):
        if client is None:
            # Reads ANTHROPIC_API_KEY. User-scoped keys (sk-ant-usr-...) also need ANTHROPIC_WORKSPACE_ID.
            workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": workspace} if workspace else None
            client = anthropic.Anthropic(default_headers=headers)
        self._client = client

    def complete(self, req: LlmRequest) -> LlmResponse:
        import time

        kwargs: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": [{"role": "user", "content": req.user}],
        }
        if req.effort:
            kwargs["output_config"] = {"effort": req.effort}
        start = time.perf_counter()
        try:
            m = self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise LlmError("Claude rate limited", retryable=True) from e
        except anthropic.APIStatusError as e:
            raise LlmError(
                f"Claude API error {e.status_code}: {e.message}", retryable=e.status_code >= 500
            ) from e
        except anthropic.APIConnectionError as e:
            raise LlmError("Claude connection error", retryable=True) from e
        ms = int((time.perf_counter() - start) * 1000)

        text = "".join(b.text for b in m.content if b.type == "text")
        u = m.usage
        usage = Usage(
            u.input_tokens,
            u.output_tokens,
            getattr(u, "cache_read_input_tokens", None) or 0,
            getattr(u, "cache_creation_input_tokens", None) or 0,
        )
        return LlmResponse(text, usage, req.model, ms)


_CHUNK = re.compile(r"\[(\S+?)\]\s*(.*?)(?=\n\[\S+?\]|</context>)", re.DOTALL)
_QUESTION = re.compile(r"<question>(.*?)</question>", re.DOTALL)
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "is",
    "are",
    "for",
    "and",
    "in",
    "on",
    "what",
    "how",
    "does",
    "do",
    "my",
    "i",
    "it",
    "be",
    "can",
    "or",
    "by",
    "with",
    "that",
    "this",
    "when",
    "if",
}


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9\-]+", s.lower()) if len(t) > 1 and t not in _STOP}


class FakeLlmClient:
    """Deterministic offline stand-in for an LLM so tests and CI need no network or key.

    It is extractive: it picks the context sentence that best overlaps the question and cites its chunk.
    It exercises the pipeline (retrieval, prompt, citations, cost tracking) and says nothing about model quality.
    """

    def complete(self, req: LlmRequest) -> LlmResponse:
        user = req.user
        q = _QUESTION.search(user)
        q_tokens = _tokens(q.group(1) if q else user)

        best, best_sentence, best_id = 0.0, None, None
        start = max(0, user.find("<context>"))
        for c in _CHUNK.finditer(user[start:]):
            for sentence in re.split(r"(?<=\.)\s+", c.group(2)):
                st = _tokens(sentence)
                if not st:
                    continue
                score = len(q_tokens & st) / max(1, len(q_tokens))
                if score > best:
                    best, best_sentence, best_id = score, sentence.strip(), c.group(1)
        text = "NOT_IN_CONTEXT" if best_sentence is None or best < 0.34 else f"{best_sentence} [{best_id}]"
        usage = Usage(self._est(req.system) + self._est(req.user), self._est(text))
        return LlmResponse(text, usage, "fake", 0)

    @staticmethod
    def _est(s: str) -> int:
        return max(1, len(s) // 4)
