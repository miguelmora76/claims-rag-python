from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .config import Settings
from .cost import CostTracker
from .llm import ClaudeLlmClient, FakeLlmClient, LlmClient, LlmError
from .logging_config import configure_logging
from .rag import build_retriever
from .service import AskService


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, pattern=r".*\S.*")


def create_app(settings: Settings | None = None, llm: LlmClient | None = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings()
    if llm is None:
        llm = ClaudeLlmClient() if settings.provider.lower() == "claude" else FakeLlmClient()
    costs = CostTracker(settings.pricing)
    service = AskService(build_retriever(settings.top_k, settings.min_score), llm, costs, settings)

    app = FastAPI(title="claims-rag-python")

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        try:
            return asdict(service.ask(req.question))
        except LlmError as e:
            raise HTTPException(status_code=503 if e.retryable else 502, detail=str(e)) from e

    @app.get("/healthz")
    def liveness() -> dict:
        return {"status": "UP"}

    @app.get("/readyz")
    def readiness() -> dict:
        # The retriever and corpus are built before the app object exists, so being here means ready.
        return {"status": "UP"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(costs.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/usage")
    def usage() -> dict:
        return {m: asdict(t) for m, t in costs.snapshot().items()}

    return app


def app() -> FastAPI:  # uvicorn --factory claims_rag.api:app
    return create_app()
