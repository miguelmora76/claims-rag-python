"""Offline eval gate: runs the golden set through the pipeline with the deterministic fake model.

It guards retrieval, prompt assembly, citation handling and abstention against regressions. Fact coverage and
pass rate are not gated: the fake is extractive and often picks the wrong sentence from correct context.
Model-quality evals run separately with ASSISTANT_PROVIDER=claude (see README).
"""

from claims_rag.config import Settings
from claims_rag.cost import CostTracker
from claims_rag.evals.runner import load_golden, run_eval
from claims_rag.llm import FakeLlmClient
from claims_rag.rag import build_retriever
from claims_rag.service import AskService


def test_golden_set_meets_thresholds():
    s = Settings(provider="fake")
    service = AskService(build_retriever(s.top_k, s.min_score), FakeLlmClient(), CostTracker(s.pricing), s)
    report = run_eval(service, load_golden())
    print(report.to_markdown())

    assert report.retrieval_recall >= 0.95
    assert report.context_recall >= 0.95
    assert report.citation_validity == 1.0
    assert report.abstain_accuracy == 1.0
