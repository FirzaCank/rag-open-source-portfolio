"""Hit Rate and MRR over scripts/retrieval_set.json, the only denominator for both, see D49.

A hit means the query's own source appears in the top k, matched per source and never per
chunk id, see D28. Run it after any change to chunking, the embedding model, or the corpus.
Numbers move only for those three reasons, so a shift with none of them means a bug.

MRR is cut off at k=8 because production retrieves 8. A rank of 21 is a miss a visitor never
sees, so it earns 0 here, and the full index rank is printed separately as a diagnosis. An
earlier ad hoc run scored MRR over the whole index and read 0.7577 against the same data,
which is the same set measured a different way, not a better result.

    python scripts/eval_retrieval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.retriever import retrieve  # noqa: E402

K_VALUES = (1, 3, 8)


def rank_of(query, gold, k=8):
    """1-based rank of the first hit from the gold source, or None if it is not in the top k."""
    for i, hit in enumerate(retrieve(query, top_k=k), start=1):
        if hit["source"] == gold:
            return i
    return None


def full_rank(query, gold):
    """Rank over the entire index. Diagnosis only, never a metric, see the note above."""
    return rank_of(query, gold, k=10_000)


def report(name, ranks):
    n = len(ranks)
    if not n:
        return
    row = [f"{name:<12} n={n:<3}"]
    for k in K_VALUES:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        row.append(f"HR@{k} {hits / n * 100:5.1f}")
    mrr = sum(1 / r for r in ranks if r is not None) / n
    row.append(f"MRR {mrr:.4f}")
    print("  ".join(row))


def main():
    data = json.loads((ROOT / "scripts/retrieval_set.json").read_text())
    queries = data["queries"]
    assert len(queries) == 66, f"the denominator is 66 queries, found {len(queries)}"

    results = []
    for q in queries:
        r = rank_of(q["query"], q["source"])
        results.append((q, r))

    ranks = [r for _, r in results]
    print()
    report("all", ranks)
    report("english", [r for q, r in results if q["lang"] == "en"])
    report("indonesian", [r for q, r in results if q["lang"] == "id"])

    misses = [(q, full_rank(q["query"], q["source"])) for q, r in results if r is None]
    print(f"\nnot retrieved in the top 8: {len(misses)} of {len(queries)}, full index rank shown")
    for q, fr in sorted(misses, key=lambda x: x[1] or 9999):
        print(f"  rank {fr:<4} {q['id']:<32} {q['query'][:48]}")
    worst = max((fr for _, fr in misses if fr), default=None)
    print(f"worst rank in the whole index: {worst}")

    deep = sorted(((q, r) for q, r in results if r and r > 3), key=lambda x: -x[1])
    print(f"\nfound but ranked below 3: {len(deep)}")
    for q, r in deep:
        print(f"  rank {r}  {q['id']:<28} {q['query'][:52]}")


if __name__ == "__main__":
    main()
