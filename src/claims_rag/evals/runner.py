from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

from ..service import AskResult, AskService
from .judge import LlmJudge


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    answerable: bool
    expected_docs: list[str]
    must_contain: list[str]


@dataclass(frozen=True)
class CaseResult:
    id: str
    answerable: bool
    retrieval_hit: bool
    context_has_facts: bool
    facts_present: bool
    citations_valid: bool
    citations_correct: bool
    abstained_correctly: bool
    judge_score: int | None
    answer: str

    @property
    def passed(self) -> bool:
        """A case passes when every check that applies to it passes."""
        if not self.answerable:
            return self.abstained_correctly
        return (
            self.retrieval_hit
            and self.context_has_facts
            and self.facts_present
            and self.citations_valid
            and self.citations_correct
        )


@dataclass(frozen=True)
class EvalReport:
    cases: list[CaseResult]
    retrieval_recall: float
    context_recall: float
    fact_coverage: float
    citation_validity: float
    citation_correctness: float
    abstain_accuracy: float
    pass_rate: float
    avg_judge_score: float | None
    total_cost_usd: float

    def to_markdown(self) -> str:
        rows = [
            ("retrieval recall (expected doc in top-k)", self.retrieval_recall),
            ("context recall (retrieved text contains the required facts)", self.context_recall),
            ("fact coverage (required facts in answer)", self.fact_coverage),
            ("citation validity (cites only retrieved chunks)", self.citation_validity),
            ("citation correctness (cites expected doc)", self.citation_correctness),
            ("abstain accuracy (declines unanswerable)", self.abstain_accuracy),
            ("case pass rate", self.pass_rate),
        ]
        out = ["# Eval report", "", "| metric | value |", "|---|---|"]
        out += [f"| {name} | {v * 100:.0f}% |" for name, v in rows]
        if self.avg_judge_score is not None:
            out.append(f"| judge groundedness (1-5) | {self.avg_judge_score:.2f} |")
        out += [f"| total cost (USD) | {self.total_cost_usd:.5f} |", "", "## Failures", ""]
        failures = [c for c in self.cases if not c.passed]
        if not failures:
            out.append("none")
        for c in failures:
            out.append(
                f"- `{c.id}` [retrieval={c.retrieval_hit} ctxFacts={c.context_has_facts} facts={c.facts_present} "
                f"cited={c.citations_correct}]: {c.answer.replace(chr(10), ' ')}"
            )
        return "\n".join(out) + "\n"


def load_golden() -> list[GoldenCase]:
    raw = json.loads((resources.files("claims_rag.evals") / "golden.json").read_text(encoding="utf-8"))
    return [
        GoldenCase(c["id"], c["question"], c["answerable"], c["expectedDocs"], c["mustContain"]) for c in raw
    ]


def _doc_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def score_case(g: GoldenCase, r: AskResult, judge: LlmJudge | None) -> CaseResult:
    retrieved_docs = {_doc_of(x.chunk_id) for x in r.retrieved}
    retrieved_ids = {x.chunk_id for x in r.retrieved}
    cited_docs = {_doc_of(c) for c in r.citations}
    ctx = "\n".join(x.text for x in r.retrieved).lower()

    ctx_facts = g.answerable and bool(g.must_contain) and all(f.lower() in ctx for f in g.must_contain)
    hit = any(d in retrieved_docs for d in g.expected_docs)
    answer = r.answer.lower()
    facts = r.answered and all(f.lower() in answer for f in g.must_contain)
    valid = retrieved_ids.issuperset(r.citations) and (not r.answered or bool(r.citations))
    correct = any(d in cited_docs for d in g.expected_docs)

    judged = None
    if judge is not None and g.answerable and r.answered:
        context = "\n".join(f"[{x.chunk_id}] {x.text}" for x in r.retrieved)
        judged = judge.score(context, g.question, r.answer)
    return CaseResult(
        g.id, g.answerable, hit, ctx_facts, facts, valid, correct, not r.answered, judged, r.answer
    )


def _rate(xs: list[CaseResult], pred) -> float:
    return 1.0 if not xs else sum(1 for x in xs if pred(x)) / len(xs)


def aggregate(results: list[CaseResult], cost: float) -> EvalReport:
    ans = [c for c in results if c.answerable]
    unans = [c for c in results if not c.answerable]
    judged = [c.judge_score for c in results if c.judge_score is not None]
    return EvalReport(
        results,
        _rate(ans, lambda c: c.retrieval_hit),
        _rate(ans, lambda c: c.context_has_facts),
        _rate(ans, lambda c: c.facts_present),
        _rate(ans, lambda c: c.citations_valid),
        _rate(ans, lambda c: c.citations_correct),
        _rate(unans, lambda c: c.abstained_correctly),
        _rate(results, lambda c: c.passed),
        sum(judged) / len(judged) if judged else None,
        cost,
    )


def run_eval(service: AskService, golden: list[GoldenCase], judge: LlmJudge | None = None) -> EvalReport:
    results, cost = [], 0.0
    for g in golden:
        r = service.ask(g.question)
        cost += r.cost_usd
        results.append(score_case(g, r, judge))
    return aggregate(results, cost)
