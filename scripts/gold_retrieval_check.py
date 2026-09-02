"""For each labeled golden set case, did the gold source actually reach the model?

This separates a retrieval miss from a generation failure. `fact-education` is the case that
forced it into existence: the 7b answer said the context did not mention any education, which is
correct grounding behaviour, and the chunk containing ITB was never in the top 8. Blaming the
model there would have been wrong. See D63.

**Two levels, and the second is the one that matters.** Source level asks whether the right
document was retrieved. Text level asks whether the sentence carrying the fact was in the chunks
that actually reached the model. They disagree: `About Firza` is one document split across about
ten chunks, so `fact-japanese` can retrieve the source at rank 3 while the JLPT sentence sits in a
chunk that never came back. Only the text level separates a retrieval miss from a model failure,
and it is possible because `golden_sources.json` stores the evidence quote. See D63.

A source in `also_answers` counts as a hit. `fact-current-job` is why: its gold source ranked 17,
but Career Timeline ranked 1 and states the current employer outright, so the model had the fact.
An earlier version of this script counted only `gold_sources` and reported 3 misses where there
was 1.

Retrieval is model independent and runs on CPU, so this costs nothing and never needs the GPU
service. Labels come from scripts/golden_sources.json, see D60.

    python scripts/gold_retrieval_check.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.llm import retrieval_query  # noqa: E402
from rag.retriever import retrieve  # noqa: E402

DEPTH = 30
TOP_K = 8

# Evidence for these two is a note about how the case works, not a quote from the corpus.
PROSE = {"id-ai-projects", "multi-followup"}


def words(text):
    clean = re.sub(r"[^a-z0-9 ]+", " ", text.replace("\u00d7", "x").lower())
    return [w for w in clean.split() if len(w) > 3]


def evidence_present(evidence, hits):
    """Fraction of the evidence's content words found in the chunks that reached the model."""
    ev = words(evidence)
    if not ev:
        return None
    seen = words(" ".join(h["text"] for h in hits))
    pool = set(seen)
    return sum(1 for w in ev if w in pool) / len(ev)


def query_of(case):
    """The text the retriever actually sees, follow-up prefixing included."""
    messages = case.get("messages") or [{"role": "user", "content": case["question"]}]
    return retrieval_query([{"role": m["role"], "content": m["content"]} for m in messages])


def main():
    labels = {l["id"]: l for l in json.loads((ROOT / "scripts/golden_sources.json").read_text())["labels"]}
    cases = {c["id"]: c for c in json.loads((ROOT / "scripts/golden_set.json").read_text())["cases"]}

    print(f"{'case':<24} {'gold':>5} {'any':>5}  {'src':>4}  {'fact':>5}  answered by")
    misses = []
    fact_missing = []
    for cid, lab in labels.items():
        q = query_of(cases[cid])
        hits = retrieve(q, top_k=DEPTH)
        accept = list(lab["gold_sources"]) + list(lab["also_answers"])

        def first_rank(sources):
            for i, h in enumerate(hits, 1):
                if h["source"] in sources:
                    return i, h["source"]
            return None, None

        gold_rank, _ = first_rank(lab["gold_sources"])
        any_rank, any_src = first_rank(accept)
        in_top8 = any_rank is not None and any_rank <= 8
        if not in_top8:
            misses.append((cid, gold_rank))
        # The text level check runs on exactly the chunks the model would have seen.
        top = hits[:TOP_K]
        frac = None if cid in PROSE else evidence_present(lab["evidence"], top)
        if frac is not None and frac < 0.7:
            fact_missing.append((cid, frac))

        g = str(gold_rank) if gold_rank else f">{DEPTH}"
        a = str(any_rank) if any_rank else f">{DEPTH}"
        f = "n/a" if frac is None else f"{frac * 100:.0f}%"
        print(f"{cid:<24} {g:>5} {a:>5}  {'yes' if in_top8 else 'NO':>4}  {f:>5}  {(any_src or 'nothing')[:40]}")

    print(f"\n{len(labels) - len(misses)} of {len(labels)} labeled cases had an answering source in the top 8")
    for cid, rank in misses:
        print(f"  MISS {cid}: no answering source in the top 8, gold ranked {rank or f'below {DEPTH}'}")
    print(f"\n{len(fact_missing)} case(s) below the evidence threshold:")
    for cid, frac in fact_missing:
        print(f"  THIN {cid}: {frac * 100:.0f} percent of the evidence words were in the top {TOP_K}")

    print("""
How to read this:
  MISS  no answering source retrieved. A failure here is retrieval, and no model fixes it.
  THIN  a suspicion, not a verdict. The 70 percent threshold is a heuristic, and it flags
        id-current-job, which answers correctly through Career Timeline while the evidence words
        come from the Hypefast document. Read the case before concluding.
  fact 100 percent with a wrong answer is the opposite finding: the model had the sentence and
  still got it wrong. That is a model failure, see D62 and D63.""")


if __name__ == "__main__":
    main()
