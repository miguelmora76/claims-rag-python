# claims-rag-python

A sample **FastAPI** service that answers questions about (invented) health-insurance claim policies using
**retrieval-augmented generation (RAG)** with **Claude**, plus an **evaluation suite** and **token/cost tracking**.

> **Honest scope:** this is a learning and portfolio project. It runs on synthetic data, has no real users, and makes no
> production-scale claims. It shows how I would structure an LLM-backed service, measure its quality and track what it costs.
> A sibling repo, `claims-rag-java`, implements the same service in Java and Spring Boot with the same corpus, golden set and metrics.

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [What this project demonstrates](#what-this-project-demonstrates)
- [Quick start (no API key needed)](#quick-start-no-api-key-needed)
- [Get an Anthropic API key](#get-an-anthropic-api-key)
- [Run with real Claude](#run-with-real-claude)
- [API reference](#api-reference)
- [Evals](#evals)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Known limits](#known-limits)

## What it does

You send a question like *"How many days do I have to file a first-level appeal?"* to an HTTP endpoint. The service:

1. Redacts obvious personal identifiers (SSNs, phone numbers, emails) from the question.
2. Retrieves the most relevant passages from a small knowledge base of policy documents.
3. Sends only those passages and the question to Claude, instructing it to answer **only** from the passages and to cite them.
4. Returns the answer, the citations, the passages it used, the token usage and the cost in USD.
5. If nothing relevant is found, or the passages don't contain the answer, it says so instead of guessing.

The knowledge base is seven short documents in `src/claims_rag/corpus/` (timely filing, denial codes, appeals, prior
authorization, coordination of benefits, payments, eligibility). **They are invented for this project and are not real payer rules.**

## How it works

```mermaid
flowchart LR
    Q[POST /api/ask] --> R[redact]
    R --> RT[Retriever<br/>embed + cosine search]
    RT -->|no relevant chunks| NA[NOT_IN_CONTEXT<br/>no model call]
    RT -->|top-k chunks| P[Prompt builder<br/>context + question]
    P --> L{LlmClient}
    L -->|provider=claude| C[Claude via Anthropic Python SDK]
    L -->|provider=fake| F[Deterministic fake]
    C --> A[AskService]
    F --> A
    A --> CT[CostTracker]
    A --> RESP[answer, citations,<br/>usage, cost]
```

The model sits behind a small `LlmClient` protocol. That is what lets the tests and CI run offline against a
deterministic fake, while the real thing calls Claude.

## What this project demonstrates

| Area | Where to look |
|---|---|
| Calling an LLM API from Python with the official SDK, typed error handling, usage metadata | `src/claims_rag/llm.py` (`ClaudeLlmClient`) |
| Provider protocol with an offline fake for tests and CI | `llm.py` (`LlmClient`, `FakeLlmClient`) |
| RAG: chunking, embeddings, vector search, grounded prompt with citations | `rag.py`, `service.py` |
| Prompt-injection posture: untrusted context, answer-only-from-context, explicit abstain token | `service.SYSTEM` |
| Token and cost tracking per request and per model | `cost.py`, `GET /api/usage` |
| Eval suite: golden set, retrieval/context/citation/abstention metrics, optional LLM judge | `evals/` |
| Eval as a CI gate | `tests/test_eval_suite.py`, `.github/workflows/ci.yml` |
| Pattern-based redaction before text leaves the service | `privacy.py` |

## Quick start (no API key needed)

By default the app uses a **fake model**, so you can install, test and call it with no key and no cost.

**Prerequisites:** Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) (a fast Python package manager;
install with `brew install uv` or see the uv docs). Developed on Python 3.14.

```bash
git clone <this-repo-url>
cd claims-rag-python

uv sync                                          # create .venv and install dependencies
uv run pytest                                    # unit tests, API tests and the offline eval gate
uv run uvicorn --factory claims_rag.api:app      # start the server on http://127.0.0.1:8000
```

In a second terminal:

```bash
curl -s localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question":"How many days do I have to file a first-level appeal?"}'

curl -s localhost:8000/api/usage
```

FastAPI also serves interactive docs at http://127.0.0.1:8000/docs.

With the fake model the answer is just the best-matching sentence from the retrieved text. That is enough to exercise
the retrieval, citation and cost-tracking code. For real answers, use Claude as described below.

## Get an Anthropic API key

To use real Claude you need an Anthropic API key. Keys are managed in the Anthropic Console.

1. Go to **https://console.anthropic.com** and sign in, or create an account.
2. Set up billing: open **Billing** (or **Plans & Billing**) and add a payment method or buy a small amount of credit.
   API usage is billed separately from any Claude chat subscription; check the Console for current billing options and any
   credits on your account. One full eval run of this project costs about **5 US cents**.
3. Open **Settings → API keys** (direct link: https://console.anthropic.com/settings/keys) and click **Create key**.
   Name it (for example `claims-rag-local`) and, if asked, choose a workspace.
4. **Copy the key immediately.** It starts with `sk-ant-` and is shown only once. If you lose it, create a new one.
5. Keep it secret: never commit it, paste it into a chat, or put it in a file tracked by git. If it leaks, delete it in the Console and create another.

> **User-scoped keys:** if your key starts with `sk-ant-usr-` instead of being created inside a workspace, the API will
> also require a workspace ID. Set `ANTHROPIC_WORKSPACE_ID` (it looks like `wrkspc_...` and is on the workspace's settings
> page in the Console). Keys created inside a workspace do not need this.

Pricing and available models change over time. Check the Console and Anthropic's pricing page before relying on the cost figures here.

## Run with real Claude

Put the key in an environment variable for your current terminal only. This prompt hides what you type and keeps the key out of your shell history:

```bash
read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
# only for sk-ant-usr-... keys:
# export ANTHROPIC_WORKSPACE_ID=wrkspc_...

ASSISTANT_PROVIDER=claude uv run uvicorn --factory claims_rag.api:app
```

(In zsh, `read -s` works as shown. In bash use `read -s ANTHROPIC_API_KEY; export ANTHROPIC_API_KEY`.)

Then call the same endpoints as above. The response now contains real token counts and a non-zero `cost_usd`, and
`GET /api/usage` shows the running totals per model.

The default model is `claude-opus-5-5` at `effort=low`. Change it with the `ASSISTANT_*` variables below.

## API reference

### `POST /api/ask`

Request body:

```json
{ "question": "How many days do I have to file a first-level appeal?" }
```

`question` is required, must not be blank, and is limited to 2000 characters (otherwise `422`).

Example response (from the offline fake model; the retrieved text is shortened here):

```json
{
  "answer": "A first-level appeal must be filed within 120 days of the date on the remittance advice that shows the denial. [appeals#0]",
  "answered": true,
  "citations": ["appeals#0"],
  "retrieved": [
    { "chunk_id": "appeals#2", "title": "Claim Appeals Process", "score": 0.22, "text": "Northwind Health decides first-level appeals within 30 calen..." },
    { "chunk_id": "appeals#0", "title": "Claim Appeals Process", "score": 0.17, "text": "Synthetic policy text for demonstration only. Not real payer..." },
    { "chunk_id": "appeals#3", "title": "Claim Appeals Process", "score": 0.16, "text": "A second-level appeal may be filed within 60 days of the fir..." }
  ],
  "usage": { "input_tokens": 303, "output_tokens": 30, "cache_read_tokens": 0, "cache_write_tokens": 0 },
  "cost_usd": 0.0,
  "latency_ms": 0
}
```

| Field | Meaning |
|---|---|
| `answered` | `false` when the system declined (nothing relevant retrieved, or the model replied `NOT_IN_CONTEXT`) |
| `citations` | chunk ids the answer cites, in `document#chunk` form |
| `retrieved` | the passages sent to the model, with similarity score |
| `cost_usd` | computed from token usage and the price table in `config.py` (0 for the fake model) |

If the Claude call fails, the endpoint returns `503` for retryable errors (rate limit, 5xx, connection) and `502` otherwise.

### `GET /api/usage`

Running token and cost totals per model since the process started.

## Evals

An **eval** measures whether the system's output is good, in a repeatable way. This project has a golden set of 15
questions in `src/claims_rag/evals/golden.json`: 12 with known answers and 3 that should be refused (two out of
scope and one prompt-injection attempt). For each question it checks:

| Metric | What it checks |
|---|---|
| Retrieval recall | a chunk from the expected document was retrieved |
| Context recall | the retrieved text actually contains the facts needed to answer |
| Fact coverage | the answer contains the required facts |
| Citation validity | the answer cites only chunks that were retrieved |
| Citation correctness | the answer cites the expected document |
| Abstain accuracy | unanswerable questions are declined |
| Judge groundedness (live only) | a second Claude model scores 1-5 how well the answer is supported by the retrieved text |

```bash
# offline pipeline eval (exits 1 by design: the fake model does not reach the 90% pass-rate gate)
uv run python -m claims_rag.evals

# live eval against Claude, with the judge (needs an API key, costs about $0.05)
ASSISTANT_PROVIDER=claude uv run python -m claims_rag.evals
```

The report prints to the console and is written to `eval-report/report.md`. The process exits with code `0` at or above a
90% pass rate, `1` below it, and `2` if the model call fails. What CI gates on is `uv run pytest`, which includes
`tests/test_eval_suite.py`.

### Results (live, 2026-10-05, `claude-opus-5-5` at low effort, judge `claude-haiku-4-5`)

| Metric | Value |
|---|---|
| Retrieval recall | 100% |
| Context recall | 100% |
| Fact coverage | 100% |
| Citation validity | 100% |
| Citation correctness | 100% |
| Abstain accuracy | 100% |
| Case pass rate | 100% |
| Judge groundedness (1-5) | 5.00 |
| Cost for the run | about $0.05 |

**How much to trust these numbers:** this is a single run, the golden set is small and written by the same person who wrote
the corpus, retrieval was tuned against that set, and the judge is a smaller model grading a larger one's short answers,
so a perfect judge score is weak evidence. The run shows the pipeline works end to end with Claude. It does not show how the system
would perform on questions it has not seen.

This version started with fixes found while building the Java sibling, where an earlier version scored 88%: IDF weighting,
light suffix stemming, an exact-match boost for denial codes such as `CO-27`, a context-recall metric, and a judge that is
given the retrieved chunk text.

### What the offline eval does and does not tell you

Offline, the "model" is a fake that extracts the best-matching sentence. It checks the plumbing: retrieval, prompt
assembly, citation parsing, abstention and cost accounting. The CI gate asserts only retrieval, context recall, citation
validity and abstention. Fact coverage and pass rate are reported offline but not gated, because the fake often picks the
wrong sentence from correct context.

## Configuration

Settings live in `src/claims_rag/config.py` and can be overridden with environment variables prefixed `ASSISTANT_`.

| Variable | Default | Meaning |
|---|---|---|
| `ASSISTANT_PROVIDER` | `fake` | `fake` (offline) or `claude` |
| `ASSISTANT_MODEL` | `claude-opus-5-5` | model used to answer |
| `ASSISTANT_JUDGE_MODEL` | `claude-haiku-4-5` | model used by the eval judge |
| `ASSISTANT_MAX_TOKENS` | `1024` | output cap per answer |
| `ASSISTANT_EFFORT` | `low` | reasoning effort; empty to omit (some models reject it) |
| `ASSISTANT_TOP_K` | `3` | passages retrieved per question |
| `ASSISTANT_MIN_SCORE` | `0.08` | minimum similarity for a passage to count |

The price table (USD per million tokens, per model) is in `config.py`. It was copied from published rates; verify before relying on it.

Other environment variables: `ANTHROPIC_API_KEY` (required for `ASSISTANT_PROVIDER=claude`) and `ANTHROPIC_WORKSPACE_ID` (only for `sk-ant-usr-` keys).

## Project layout

```
src/claims_rag/
  config.py          settings and price table
  llm.py             LlmClient protocol, Claude implementation, offline fake, usage types
  rag.py             chunker, hashing embedder, vector store, retriever
  service.py         redact -> retrieve -> prompt -> model -> cite -> cost
  cost.py            token usage to USD, per-model totals
  privacy.py         regex redaction of common identifiers
  api.py             FastAPI app and endpoints
  corpus/*.md        the invented policy documents
  evals/             golden set, runner, metrics, LLM judge, CLI (python -m claims_rag.evals)
tests/               unit tests, API tests, offline eval gate
.github/workflows/ci.yml   runs ruff and pytest on every push and pull request
```

## Known limits

- **Embeddings are lexical, not semantic.** The embedder hashes words and word pairs with IDF weighting, crude stemming and a boost for code-like tokens. It cannot match synonyms. The `Embedder` protocol is where a real embedding model would plug in.
- **The vector store is brute force and in memory.** Fine for hundreds of chunks, not for large corpora.
- **The golden set is small** and written by the same person who wrote the corpus, so scores are optimistic.
- **Redaction is regex-based.** `redact()` misses names, addresses and free text. It shows where redaction belongs; it is not HIPAA de-identification, and this project makes no compliance claims.
- **Not production-ready:** no authentication, rate limiting, streaming, persistence, retries beyond the SDK's own, or prompt caching (the system prompt is below the minimum cacheable size). Server-side refusal fallbacks are not configured.
- No license file is included yet, so default copyright applies.
