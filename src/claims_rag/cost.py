from __future__ import annotations

import threading
from dataclasses import dataclass

from .config import Price
from .llm import Usage


@dataclass(frozen=True)
class Totals:
    requests: int
    usage: Usage
    cost_usd: float


class CostTracker:
    """Converts token usage to USD with a configured price table and keeps running totals per model."""

    def __init__(self, pricing: dict[str, Price]):
        self._pricing = pricing
        self._totals: dict[str, Totals] = {}
        self._lock = threading.Lock()

    def cost_of(self, model: str, u: Usage) -> float:
        p = self._pricing.get(model)
        if p is None:
            return 0.0
        return (
            u.input_tokens * p.input
            + u.output_tokens * p.output
            + u.cache_read_tokens * p.cache_read
            + u.cache_write_tokens * p.cache_write
        ) / 1_000_000

    def record(self, model: str, u: Usage) -> float:
        cost = self.cost_of(model, u)
        with self._lock:
            t = self._totals.get(model, Totals(0, Usage(), 0.0))
            self._totals[model] = Totals(t.requests + 1, t.usage + u, t.cost_usd + cost)
        return cost

    def snapshot(self) -> dict[str, Totals]:
        with self._lock:
            return dict(self._totals)
