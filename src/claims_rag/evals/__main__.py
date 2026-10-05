"""Run the golden set: `uv run python -m claims_rag.evals` (offline) or with ASSISTANT_PROVIDER=claude (live, spends money)."""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import Settings
from ..cost import CostTracker
from ..llm import ClaudeLlmClient, FakeLlmClient, LlmError
from ..rag import build_retriever
from ..service import AskService
from .judge import LlmJudge
from .runner import load_golden, run_eval

MIN_PASS_RATE = 0.9


def main() -> int:
    settings = Settings()
    live = settings.provider.lower() == "claude"
    llm = ClaudeLlmClient() if live else FakeLlmClient()
    service = AskService(
        build_retriever(settings.top_k, settings.min_score), llm, CostTracker(settings.pricing), settings
    )
    try:
        report = run_eval(service, load_golden(), LlmJudge(llm, settings) if live else None)
    except LlmError as e:
        print(f"Eval aborted: {e}", file=sys.stderr)
        return 2
    md = report.to_markdown()
    print(md)
    Path("eval-report").mkdir(exist_ok=True)
    Path("eval-report/report.md").write_text(md, encoding="utf-8")
    if report.pass_rate < MIN_PASS_RATE:
        print(f"pass rate {report.pass_rate:.0%} below threshold {MIN_PASS_RATE:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
