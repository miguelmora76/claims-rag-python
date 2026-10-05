# claims-rag-python

Sample FastAPI project showing a retrieval-augmented question-answering service with an eval suite and
token/cost tracking, calling Claude through the official Anthropic Python SDK.

It is the Python counterpart of [`claims-rag-java`](../claims-rag-java) (same corpus, same golden set, same metrics).
This is a learning/portfolio project. It is not a production system, has no real users, and uses only synthetic data.

## What it demonstrates

| Area | Where |
|---|---|
| Calling an LLM API from Python (official SDK, typed errors, usage metadata) | `llm.py` (`ClaudeLlmClient`) |
| Provider protocol with an offline fake for tests and CI | `llm.py` (`LlmClient`, `FakeLlmClient`) |
| RAG: chunking, embeddings, vector search, grounded prompt with citations | `rag.py`, `service.py` |
| Prompt-injection posture: untrusted context, answer-only-from-context, abstain token | `service.SYSTEM` |
| Token and cost tracking per request and per model | `cost.py`, `GET /api/usage` |
| Eval suite: golden set, retrieval/context/citation/abstention metrics, optional LLM judge | `evals/` |
| Eval as a CI gate | `tests/test_eval_suite.py`, `.github/workflows/ci.yml` |
| Pattern-based redaction before text leaves the service | `privacy.py` |

## The domain

Billing staff asking questions about payer claim policy (timely filing, denial codes, appeals, prior auth).
The seven documents in `src/claims_rag/corpus/` are invented for this project. They are not real payer rules.

## Run it

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                   # tests + offline eval gate
uv run uvicorn --factory claims_rag.api:app   # fake provider, http://localhost:8000
curl -s localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question":"How many days do I have to file a first-level appeal?"}'
curl -s localhost:8000/api/usage
```

With real Claude:

```bash
export ANTHROPIC_API_KEY=...
# user-scoped keys (sk-ant-usr-...) also need: export ANTHROPIC_WORKSPACE_ID=wrkspc_...
ASSISTANT_PROVIDER=claude uv run uvicorn --factory claims_rag.api:app
```

Settings (model, effort, `top_k`, price table) are in `src/claims_rag/config.py` and can be overridden with
`ASSISTANT_*` environment variables. The default model is `claude-opus-5-5` with `effort=low`. Prices are copied
from published rates at the time of writing; verify them.

## Evals

```bash
uv run python -m claims_rag.evals                                   # offline pipeline eval
ASSISTANT_PROVIDER=claude uv run python -m claims_rag.evals         # live eval + LLM judge (spends real money)
```

The golden set has 15 cases: 12 answerable, 3 not (out of scope, plus a prompt-injection attempt). Metrics:
retrieval recall (expected doc in top-k), context recall (retrieved text contains the required facts), fact coverage,
citation validity, citation correctness, abstention accuracy, pass rate, and (live only) judge groundedness.
The report is written to `eval-report/report.md`. The CLI exits 1 below a 90% pass rate and 2 if the model call fails.

### What the offline eval does and does not tell you

The offline run uses a fake, extractive "model". It checks the plumbing: retrieval, prompt assembly, citation
parsing, abstention, cost accounting. The CI gate only asserts retrieval, context recall, citation validity and
abstention. Fact coverage and pass rate are reported but not gated offline, because the fake often picks the wrong
sentence from correct context, so the offline CLI run exits 1 by design. Answer quality needs a live run, and **this
repo has no live results yet**.

### Lessons carried over from the Java version

Building the Java sibling showed that "expected document in top-k" overstates retrieval quality, and that a purely
lexical embedder needs help. This version starts with those fixes: IDF weighting, light suffix stemming, an
exact-match boost for denial-code tokens such as `CO-27`, a context-recall metric, and a judge that is given the
retrieved chunk text. The retrieval parameters were tuned against this same small golden set, so the 100% offline
retrieval numbers are optimistic and not a claim about generalisation.

## Known limits

- Embeddings are lexical (hashed unigrams and bigrams with IDF, crude stemming, code-token boost), not semantic. The `Embedder` protocol is where a real embedding model would go.
- The vector store is brute force and in memory. Fine for hundreds of chunks only.
- The golden set is small and written by the same person who wrote the corpus.
- `redact()` is regex-based and misses names and free text. It shows where redaction belongs; it is not HIPAA de-identification.
- No auth, rate limiting, streaming, persistence or prompt caching (the system prompt is below the minimum cacheable size).
- Server-side refusal fallbacks are not configured.
