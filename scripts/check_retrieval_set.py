"""Validate scripts/retrieval_set.json before it is frozen.

This set is the only denominator for Hit Rate and MRR in Phase 5, see D49. Three ways it can look
finished while measuring nothing, and one check each:

1. Unbalanced coverage. 3 queries per source across all 22 sources, or one well covered source
   dominates the average.
2. Copied phrasing. A query that reuses its chunk's words measures string matching, not retrieval,
   and it will score well no matter which strategy runs. Any 6 word sequence shared with the source
   text fails the file.
3. Silent editing. Once `frozen` is true, this refuses to pass a file whose query text has changed,
   using a recorded checksum. A set editable after seeing results measures nothing, same reason as
   D41.

Run: python scripts/check_retrieval_set.py
"""

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag.sources import get_docs

SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_set.json")

QUERIES_PER_SOURCE = 3
INDONESIAN_TARGET = 12
NGRAM = 6
CONTAINMENT_WARN = 0.7

# Too common to be evidence of copying. Names stay in: repeating a company name is not reuse.
STOP = {
    "a", "an", "and", "the", "of", "in", "on", "at", "for", "to", "with", "by", "from", "as", "is",
    "was", "were", "are", "his", "he", "it", "that", "this", "what", "which", "who", "how", "did",
    "does", "do", "apa", "yang", "di", "ke", "dari", "dan", "untuk", "dengan", "itu", "ini", "saja",
    "siapa", "berapa", "bagaimana", "kapan", "adalah", "pada", "nya",
}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: list[str], n: int = NGRAM) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def main() -> int:
    with open(SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = data["queries"]

    docs = get_docs()
    corpus_sources = {d["source"] for d in docs}
    text_by_source: dict[str, str] = {}
    for d in docs:
        text_by_source[d["source"]] = text_by_source.get(d["source"], "") + "\n" + d["text"]

    errors: list[str] = []
    warnings: list[str] = []

    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        errors.append(f"duplicate ids: {sorted(dupes)}")

    blank = [e["id"] for e in entries if not e["query"].strip()]
    if blank:
        errors.append(f"{len(blank)} queries still empty, first few: {blank[:5]}")

    per_source: dict[str, int] = {}
    for e in entries:
        per_source[e["source"]] = per_source.get(e["source"], 0) + 1
    for src in sorted(corpus_sources):
        n = per_source.get(src, 0)
        if n != QUERIES_PER_SOURCE:
            errors.append(f"source has {n} queries, expected {QUERIES_PER_SOURCE}: {src}")
    for src in sorted(set(per_source) - corpus_sources):
        errors.append(f"source label is not in the corpus: {src!r}")

    n_id = sum(1 for e in entries if e.get("lang") == "id")
    if n_id != INDONESIAN_TARGET:
        errors.append(f"{n_id} Indonesian queries, expected {INDONESIAN_TARGET}")

    texts = [e["query"].strip().lower() for e in entries if e["query"].strip()]
    if len(texts) != len(set(texts)):
        errors.append("two queries have identical text")

    for e in entries:
        q = e["query"].strip()
        if not q:
            continue
        source_text = text_by_source.get(e["source"], "")
        shared = ngrams(words(q)) & ngrams(words(source_text))
        if shared:
            sample = " ".join(sorted(shared)[0])
            errors.append(f"{e['id']}: reuses {NGRAM} words from its own source: \"{sample}\"")

        content = [w for w in words(q) if w not in STOP]
        if content:
            src_words = set(words(source_text))
            ratio = sum(1 for w in content if w in src_words) / len(content)
            if ratio > CONTAINMENT_WARN:
                warnings.append(f"{e['id']}: {ratio:.0%} of content words appear in the source, "
                                f"consider rewording: {q!r}")

    if data.get("frozen"):
        digest = hashlib.sha256(
            "\n".join(f"{e['id']}\t{e['query']}" for e in entries).encode("utf-8")
        ).hexdigest()
        recorded = data.get("checksum")
        if not recorded:
            print(f"frozen is true but no checksum recorded. Add:\n  \"checksum\": \"{digest}\"")
            return 1
        if recorded != digest:
            errors.append(f"file is frozen but the queries changed. recorded {recorded[:12]}, "
                          f"actual {digest[:12]}")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"FAIL  {e}")

    filled = len(entries) - len(blank)
    print(f"\n{filled}/{len(entries)} queries written, {n_id} Indonesian, "
          f"{len(per_source)} sources covered")
    if errors:
        print(f"{len(errors)} problems. Not ready to freeze.")
        return 1
    print("all checks passed" + (f", {len(warnings)} warnings to look at" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
