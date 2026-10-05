from claims_rag.config import Price
from claims_rag.cost import CostTracker
from claims_rag.llm import Usage
from claims_rag.privacy import redact
from claims_rag.rag import chunk_markdown


def test_chunker_uses_title_and_stable_ids():
    chunks = chunk_markdown(
        "doc", "# My Title\n\nFirst paragraph here.\n\nSecond paragraph is a bit longer than the first.\n", 40
    )
    assert [c.id for c in chunks] == ["doc#0", "doc#1"]
    assert chunks[0].title == "My Title"
    assert "# My Title" not in chunks[0].text


def test_redactor_masks_common_identifiers():
    out = redact("Patient 123-45-6789 call 555-123-4567 or a.b@example.com, DOB 1/2/1980")
    for token in ("[SSN]", "[PHONE]", "[EMAIL]", "[DATE]"):
        assert token in out
    for raw in ("123-45-6789", "555-123-4567", "example.com"):
        assert raw not in out


def test_cost_tracker_applies_price_table_and_accumulates():
    tracker = CostTracker({"m": Price(input=4.0, output=20.0, cache_read=0.2, cache_write=5.0)})
    assert tracker.record("m", Usage(1_000_000, 100_000)) == 6.0
    tracker.record("m", Usage(0, 50_000, 1_000_000, 0))
    t = tracker.snapshot()["m"]
    assert t.requests == 2
    assert abs(t.cost_usd - 7.2) < 1e-9
    assert tracker.cost_of("unknown-model", Usage(5, 5)) == 0.0
