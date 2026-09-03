"""Vector similarity search over the pre-built index in data/embeddings.json.

A port of `rag/retriever.py` from the source repository. Two things changed and nothing else:
the query embedding comes from fastembed instead of the Gemini API, and the index is checked
against the model that built it before it is used.

Flat numpy cosine over the whole index, no FAISS. 76 vectors of 768 dimensions is one matrix
multiply of 58k floats, which is instant, and FAISS would be a dependency to install, explain,
and defend. See D22.

**Why the guard matters more than it looks.** multilingual e5 base produces 768 dimensions, the
same as the Gemini index it replaces. An index built by the wrong model has the right shape, loads
with no error, and turns retrieval into noise that nothing downstream can detect. The check covers
`model_file` too, since quantisation changes the vectors while the name and the dimension stay
identical. See D27.

`MIN_SCORE` is 0.0, and that is the calibrated answer rather than a placeholder. Measured over the
41 golden set questions, top1 spans 0.7824 to 0.8573 and the off-topic category outscores half the
factual one, so no absolute floor separates a real question from a question that should be refused.
The floor that mattered under Gemini, 0.4, passes the entire index here. See D57.
"""

import json
import os
from functools import lru_cache

import numpy as np

from .embed import DIM, MODEL_FILE, MODEL_NAME, embed_query

_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "embeddings.json")

TOP_K = 8
# 16 was measured and rejected: it fixed fact-education but broke 5 cases, 33 of 41 against 36.
# Calibrated to no floor. Off-topic questions score as high as factual ones. See D57.
MIN_SCORE = 0.0


@lru_cache(maxsize=1)
def _load_index():
    """Load the index once per warm process, refusing anything the current model did not build."""
    with open(_INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    if not isinstance(index, dict) or "chunks" not in index:
        raise ValueError(
            "embeddings.json has no metadata block. The bare list format came from the Gemini "
            "build script and cannot be verified. Rebuild with scripts/build_index.py."
        )

    for field, expected in (("model", MODEL_NAME), ("dim", DIM), ("model_file", MODEL_FILE)):
        actual = index.get(field)
        if actual != expected:
            raise ValueError(
                f"index {field} is {actual!r}, this process embeds with {expected!r}. "
                "Rebuild with scripts/build_index.py."
            )

    entries = index["chunks"]
    sources = [e["source"] for e in entries]
    texts = [e["text"] for e in entries]
    vectors = np.array([e["vector"] for e in entries], dtype=np.float32)

    # Normalised again on load, so cosine is a dot product even if a future writer forgets.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return sources, texts, vectors / norms


# The widget's suggestion chips send identical questions repeatedly. This caches the vector, never
# the answer, so it cannot affect what the model says. The answer cache is separate, see D36.
_query_cache: dict[str, np.ndarray] = {}
_QUERY_CACHE_MAX = 64


def _embed_cached(query: str) -> np.ndarray:
    key = query.strip().lower()
    hit = _query_cache.get(key)
    if hit is not None:
        return hit
    vec = embed_query(query)
    if len(_query_cache) >= _QUERY_CACHE_MAX:
        _query_cache.clear()
    _query_cache[key] = vec
    return vec


def retrieve(query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> list[dict]:
    """Return up to `top_k` chunks ranked by cosine similarity, each above `min_score`.

    Each hit is `{"source": str, "text": str, "score": float}`. `top_k` and `min_score` are
    arguments rather than only constants because Phase 5 sweeps both.
    """
    sources, texts, vectors = _load_index()
    if not texts:
        return []

    qv = _embed_cached(query)
    qv = qv / (np.linalg.norm(qv) or 1.0)

    scores = vectors @ qv
    order = np.argsort(scores)[::-1][:top_k]
    return [
        {"source": sources[i], "text": texts[i], "score": float(scores[i])}
        for i in order
        if scores[i] >= min_score
    ]


def format_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a context block.

    The block is appended to the visitor's latest message, never to the system prompt. The system
    prompt stays static so that retrieved text, which can contain anything, cannot reach the place
    where instructions are trusted. The 6 `injection` cases in the golden set test exactly this.
    See D30.
    """
    if not hits:
        return "(No relevant information found in the portfolio.)"
    return "\n\n---\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)


if __name__ == "__main__":
    sources, texts, vectors = _load_index()
    assert len(sources) == len(texts) == len(vectors), "index arrays out of sync"
    assert vectors.shape == (76, DIM), vectors.shape
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4), "vectors not normalised"
    assert len(set(sources)) == 22, len(set(sources))

    # A known query must return its own source first. This is the whole point of Phase 1.
    hits = retrieve("Where does Firza work right now?")
    assert hits, "no hits at all"
    assert "Hypefast" in hits[0]["source"] or "Hypefast" in hits[0]["text"], hits[0]["source"]

    # Scores must descend, otherwise the ranking is broken and every metric later is meaningless.
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), scores

    # The vector cache must return the same vector, not merely a similar one.
    a = _embed_cached("Where does Firza work right now?")
    b = _embed_cached("  WHERE DOES FIRZA WORK RIGHT NOW?  ")
    assert np.array_equal(a, b), "the query cache missed on a case and whitespace variant"

    print(f"{len(sources)} chunks, {len(set(sources))} sources, dim {vectors.shape[1]}, unit norms")
    print(f"query: Where does Firza work right now?  ->  {len(hits)} hits")
    for h in hits:
        print(f"  {h['score']:.4f}  {h['source']}")
    print("\nformat_context, first 200 characters:")
    print(format_context(hits)[:200])
