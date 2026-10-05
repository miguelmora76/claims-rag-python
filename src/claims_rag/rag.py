from __future__ import annotations

import math
import re
from dataclasses import dataclass
from importlib import resources
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    """id looks like "timely-filing#2" (document id, chunk index)."""

    id: str
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def chunk_markdown(doc_id: str, markdown: str, max_chars: int = 300) -> list[Chunk]:
    """Paragraph-based chunks, merging short paragraphs up to max_chars."""
    title = doc_id
    paragraphs: list[str] = []
    for p in re.split(r"\n\s*\n", markdown):
        t = p.strip()
        if not t:
            continue
        if t.startswith("# "):
            first, _, rest = t.partition("\n")
            title = first[2:].strip()
            t = rest.strip()
            if not t:
                continue
        paragraphs.append(t)

    out: list[Chunk] = []
    cur = ""
    for p in paragraphs:
        if cur and len(cur) + len(p) + 1 > max_chars:
            out.append(Chunk(f"{doc_id}#{len(out)}", doc_id, title, cur))
            cur = ""
        cur = f"{cur}\n{p}" if cur else p
    if cur:
        out.append(Chunk(f"{doc_id}#{len(out)}", doc_id, title, cur))
    return out


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    root = resources.files("claims_rag") / "corpus"
    for f in sorted(root.iterdir(), key=lambda x: x.name):
        if f.name.endswith(".md"):
            chunks.extend(chunk_markdown(f.name[: -len(".md")], f.read_text(encoding="utf-8")))
    return chunks


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "is",
    "are",
    "for",
    "and",
    "in",
    "on",
    "what",
    "how",
    "does",
    "do",
    "my",
    "i",
    "it",
    "be",
    "can",
    "or",
    "by",
    "with",
    "that",
    "this",
    "when",
    "if",
}
_CODE = re.compile(r"[a-z]{1,3}-\d+")


def stem(t: str) -> str:
    """Crude suffix stripping so "decided", "decides" and "decision" land on the same feature."""
    for suffix in ("ions", "ion", "ing", "ed", "es", "s"):
        if len(t) > len(suffix) + 3 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def _bucket(feature: str, dims: int) -> tuple[int, float]:
    # Stable across runs (Python's str hash is randomised per process).
    h = 2166136261
    for ch in feature.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h % dims, (1.0 if (h >> 31) == 0 else -1.0)


class HashingEmbedder:
    """Local, dependency-free embedder: hashed unigrams and bigrams with IDF, L2-normalised.

    Lexical, not semantic. Swap a real embedding model in behind the Embedder protocol for semantic search.
    """

    def __init__(self, dims: int = 512):
        self.dims = dims
        self._idf: list[float] | None = None

    def fit(self, corpus: list[str]) -> None:
        df = [0] * self.dims
        for doc in corpus:
            for i, x in enumerate(self._raw(doc)):
                if x != 0:
                    df[i] += 1
        n = len(corpus)
        self._idf = [math.log(1.0 + n / (1 + d)) for d in df]

    def embed(self, text: str) -> list[float]:
        v = self._raw(text)
        if self._idf is not None:
            v = [x * w for x, w in zip(v, self._idf)]
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def _raw(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        prev: str | None = None
        for token in re.split(r"[^a-z0-9\-]+", text.lower()):
            if len(token) < 2 or token in _STOP:
                continue
            raw = stem(token)
            # denial codes like co-27 are strong exact-match signals
            self._add(v, raw, 3.0 if _CODE.fullmatch(raw) else 1.0)
            if prev is not None:
                self._add(v, f"{prev}_{raw}", 0.5)
            prev = raw
        return v

    def _add(self, v: list[float], feature: str, weight: float) -> None:
        idx, sign = _bucket(feature, self.dims)
        v[idx] += sign * weight


class VectorStore:
    """Brute-force in-memory cosine search. Fine for a few hundred chunks; use a real vector DB beyond that."""

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        self._items.append((chunk, vector))

    def __len__(self) -> int:
        return len(self._items)

    def search(self, query: list[float], k: int, min_score: float) -> list[ScoredChunk]:
        scored = [ScoredChunk(c, sum(a * b for a, b in zip(query, v))) for c, v in self._items]
        scored = [s for s in scored if s.score >= min_score]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int, min_score: float):
        self._embedder, self._store, self._k, self._min = embedder, store, top_k, min_score

    def retrieve(self, question: str) -> list[ScoredChunk]:
        return self._store.search(self._embedder.embed(question), self._k, self._min)


def build_retriever(top_k: int, min_score: float) -> Retriever:
    chunks = load_corpus()
    embedder = HashingEmbedder(512)
    embedder.fit([f"{c.title} {c.text}" for c in chunks])
    store = VectorStore()
    for c in chunks:
        store.add(c, embedder.embed(f"{c.title} {c.text}"))
    return Retriever(embedder, store, top_k, min_score)
