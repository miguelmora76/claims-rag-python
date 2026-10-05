import pytest
from fastapi.testclient import TestClient

from claims_rag.api import create_app
from claims_rag.config import Settings
from claims_rag.llm import LlmError, LlmRequest, LlmResponse


@pytest.fixture
def client():
    return TestClient(create_app(Settings(provider="fake")))


def test_answers_with_citation_and_usage(client):
    r = client.post("/api/ask", json={"question": "How many days do I have to file a first-level appeal?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answered"] is True
    assert body["citations"][0].startswith("appeals#")
    assert isinstance(body["usage"]["input_tokens"], int)


def test_abstains_when_nothing_relevant(client):
    r = client.post("/api/ask", json={"question": "best pizza toppings"})
    assert r.status_code == 200
    assert r.json()["answered"] is False


def test_rejects_blank_question(client):
    assert client.post("/api/ask", json={"question": " "}).status_code == 422


def test_usage_endpoint_responds(client):
    assert client.get("/api/usage").status_code == 200


def test_llm_failure_maps_to_503_when_retryable():
    class Boom:
        def complete(self, req: LlmRequest) -> LlmResponse:
            raise LlmError("rate limited", retryable=True)

    c = TestClient(create_app(Settings(provider="fake"), llm=Boom()))
    assert c.post("/api/ask", json={"question": "How many days for a first-level appeal?"}).status_code == 503
