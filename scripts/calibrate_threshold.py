"""Measure the retrieval score spread over the 41 golden set questions, and pick MIN_SCORE from it.

Phase 1 Step 3. The old threshold, 0.4, was calibrated against Gemini score spread and carries no
meaning under multilingual e5. This script produces the numbers the new threshold is chosen from,
and it is the reason MIN_SCORE starts at 0.0: a floor is only defensible once the spread is known.

It reads the golden set questions and nothing else from that file. Nothing here scores an answer,
because no LLM exists in the project until Phase 2.

Run: python scripts/calibrate_threshold.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag.retriever import TOP_K, _embed_cached, _load_index

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")


def query_of(case: dict) -> str:
    """The text retrieval actually runs on.

    35 cases carry a single `question`. The 6 multi-turn and send-message cases carry `messages`
    instead, and retrieval sees only the latest user turn, which is what production does too.
    """
    if "question" in case:
        return case["question"]
    return [m["content"] for m in case["messages"] if m["role"] == "user"][-1]


def scores_for(query: str, V: np.ndarray) -> np.ndarray:
    qv = _embed_cached(query)
    return V @ (qv / (np.linalg.norm(qv) or 1.0))


def main() -> None:
    sources, _, V = _load_index()
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    rows = []
    for c in cases:
        s = np.sort(scores_for(query_of(c), V))[::-1]
        rows.append({
            "id": c["id"],
            "category": c["category"],
            "top1": float(s[0]),
            "topk": float(s[TOP_K - 1]),
            "worst": float(s[-1]),
        })

    print(f"{'id':28s} {'category':13s} {'top1':>7s} {'top8':>7s} {'worst':>7s} {'1-8 gap':>8s}")
    for r in rows:
        print(f"{r['id']:28s} {r['category']:13s} {r['top1']:7.4f} {r['topk']:7.4f} "
              f"{r['worst']:7.4f} {r['top1'] - r['topk']:8.4f}")

    top1 = np.array([r["top1"] for r in rows])
    topk = np.array([r["topk"] for r in rows])
    worst = np.array([r["worst"] for r in rows])

    print(f"\n{len(rows)} questions over {len(sources)} chunks")
    print(f"top1: min {top1.min():.4f}, median {np.median(top1):.4f}, max {top1.max():.4f}")
    print(f"top8: min {topk.min():.4f}, median {np.median(topk):.4f}, max {topk.max():.4f}")
    print(f"worst in index: min {worst.min():.4f}, max {worst.max():.4f}")
    print(f"top1 minus top8: min {(top1 - topk).min():.4f}, median {np.median(top1 - topk):.4f}")

    # The question a threshold has to answer: does any single number keep the 8th hit of a good
    # question while dropping the 8th hit of an off-topic one? Overlap here means the answer is no.
    print("\nwhat an absolute floor would do, chunks kept per question out of 8:")
    for t in (0.0, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82):
        kept = []
        empty = 0
        for c in cases:
            s = np.sort(scores_for(query_of(c), V))[::-1][:TOP_K]
            n = int((s >= t).sum())
            kept.append(n)
            empty += n == 0
        print(f"  floor {t:.2f}: mean {np.mean(kept):4.1f}, min {min(kept)}, "
              f"questions left with nothing {empty}")

    # A relative floor asks a different question: keep a hit only if it is close to the best hit for
    # this query. It survives the compression because it never compares across queries.
    print("\nwhat a relative floor would do, keep hits within X of that query's top1:")
    for frac in (0.02, 0.03, 0.05, 0.08, 0.12):
        kept = []
        for c in cases:
            s = np.sort(scores_for(query_of(c), V))[::-1][:TOP_K]
            kept.append(int((s >= s[0] - frac).sum()))
        print(f"  within {frac:.2f}: mean {np.mean(kept):4.1f}, min {min(kept)}, max {max(kept)}")

    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["top1"])
    print("\ntop1 by category, the off-topic and injection rows are the interesting ones:")
    for cat in sorted(by_cat, key=lambda c: -np.median(by_cat[c])):
        v = np.array(by_cat[cat])
        print(f"  {cat:13s} n={len(v):2d}  median {np.median(v):.4f}  min {v.min():.4f}  max {v.max():.4f}")


if __name__ == "__main__":
    main()
