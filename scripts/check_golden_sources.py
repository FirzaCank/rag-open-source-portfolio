"""Check scripts/golden_sources.json against the golden set and the corpus.

Fails if a case id does not exist, a gold source is not a real corpus source,
or the evidence text does not actually appear in the source it points at.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Evidence for these two is a note about how the case works, not a corpus quote.
PROSE = {"id-ai-projects", "multi-followup"}


def norm(s):
    s = s.replace("×", "x").replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower())


def words(s):
    return [w for w in norm(s).split() if len(w) > 2]


def main():
    labels = json.loads((ROOT / "scripts/golden_sources.json").read_text())
    golden = json.loads((ROOT / "scripts/golden_set.json").read_text())
    index = json.loads((ROOT / "data/embeddings.json").read_text())

    case_ids = {c["id"] for c in golden["cases"]}
    by_source = {}
    for c in index["chunks"]:
        by_source.setdefault(c["source"], []).append(c["text"])
    corpus = {s: " ".join(t) for s, t in by_source.items()}

    fail = []
    for lab in labels["labels"]:
        cid = lab["id"]
        if cid not in case_ids:
            fail.append(f"{cid}: not a case in golden_set.json")
            continue
        for src in lab["gold_sources"] + lab["also_answers"]:
            if src not in corpus:
                fail.append(f"{cid}: unknown source {src!r}")
        if cid in PROSE or not lab["gold_sources"]:
            continue
        # Evidence must live in the first gold source, allowing for "..." elisions.
        text = norm(corpus[lab["gold_sources"][0]])
        ev = words(lab["evidence"])
        missing = [w for w in ev if w not in text]
        if len(missing) > len(ev) * 0.2:
            fail.append(f"{cid}: evidence not in source, missing {missing[:6]}")

    n = len(labels["labels"])
    assert n == labels["cases_labeled"], f"cases_labeled says {labels['cases_labeled']}, file has {n}"
    assert len(golden["cases"]) == labels["cases_total"] == 41
    assert len({l["id"] for l in labels["labels"]}) == n, "duplicate case id"

    if fail:
        print("FAIL")
        for f in fail:
            print("  " + f)
        sys.exit(1)
    print(f"all checks passed: {n} labeled, {41 - n} unlabeled, sources and evidence verified")


if __name__ == "__main__":
    main()
