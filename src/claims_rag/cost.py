from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter

from .config import Price
from .llm import Usage

_TOKEN_TYPES = ("input", "output", "cache_read", "cache_write")


@dataclass(frozen=True)
class Totals:
    requests: int
    usage: Usage
    cost_usd: float


class CostTracker:
    """Converts token usage to USD with a configured price table and records it as Prometheus counters.

    The process keeps no totals of its own: scrape /metrics and sum across instances for fleet-wide numbers.
    """

    def __init__(self, pricing: dict[str, Price], registry: CollectorRegistry | None = None):
        self._pricing = pricing
        self.registry = registry or CollectorRegistry()
        self._requests = Counter("assistant_llm_requests", "LLM calls", ["model"], registry=self.registry)
        self._tokens = Counter("assistant_llm_tokens", "Tokens used", ["model", "type"], registry=self.registry)
        self._cost = Counter("assistant_llm_cost_usd", "Estimated spend in USD", ["model"], registry=self.registry)

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
        self._requests.labels(model).inc()
        self._cost.labels(model).inc(cost)
        for kind, n in zip(_TOKEN_TYPES, (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens)):
            self._tokens.labels(model, kind).inc(n)
        return cost

    def snapshot(self) -> dict[str, Totals]:
        """This instance's totals since it started, read back from the registry."""
        models = {
            s.labels["model"]
            for family in self.registry.collect()
            for s in family.samples
            if s.name == "assistant_llm_requests_total"
        }
        out = {}
        for m in models:
            tokens = [int(self._sample("assistant_llm_tokens_total", model=m, type=t)) for t in _TOKEN_TYPES]
            out[m] = Totals(
                int(self._sample("assistant_llm_requests_total", model=m)),
                Usage(*tokens),
                self._sample("assistant_llm_cost_usd_total", model=m),
            )
        return out

    def _sample(self, name: str, **labels: str) -> float:
        return self.registry.get_sample_value(name, labels) or 0.0
