"""Build data/train_pairs.json: template generated (query, chunk) pairs for the Phase 5c fine-tune.

Templates, not a generative model. The pairs are deterministic, so all of them can be read by hand,
and a bad generator cannot quietly teach the model something wrong.

**The templates deliberately avoid the words in the chunk they point at.** The failure Phase 5c
exists to fix is exactly that gap: `fact-education` fails because "Where did Firza study?" scores
0.7298 against a chunk that opens with `Education: Bachelor's Degree`, losing to `Career Timeline`
at 0.8386 which contains no education text at all. A template that asked "What is Firza's
education?" would train the model on the one phrasing it already handles. See D89.

**Pairs come from variant B chunking, the index stays variant A.** A positive has to be a chunk
the label actually owns. Under variant A the 255 character `Education:` paragraph sits inside a
1200 character chunk whose other 921 characters are career achievements, so it owns 0.21 of the
chunk and training against it would teach that career text answers an education query. Variant B
isolates it and the share reaches 1.00. Variant B was separately evaluated as a production index
and rejected, HR@1 68.2 to 59.1, so it never reaches `data/embeddings.json`. That split is sound:
the `Education:` sentence is present in the variant A index too, and a model trained to pull
"Where did Firza study?" toward that sentence retrieves the variant A chunk containing it. See D90.

**One positive per query, and no ambiguous label.** A project query first matched all 14 chunks of
its own document, which pulls the query vector toward 14 targets at once. A source name query now
takes that document's first chunk only. Separately, `Languages:` labels two different things here,
spoken languages under `About Firza` and `Languages: Python, SQL, JavaScript` inside the skills
block, so `LABEL_MUST_CONTAIN` requires a word only the spoken sense carries. `Languages:` is also
77 characters and merges forward under `chunk.MIN_PARA_CHARS`, owning 0.06 of the chunk it lands in,
so `SHORT_LABEL_CHARS` lets a paragraph too short to ever reach `MIN_LABEL_SHARE` keep that chunk.
Without it the 4 Languages queries get no positive and `Does he know any Japanese?`, one of the 66
eval queries, is untouched by the fine-tune. All three are asserted in the self check. See D90.

**27 of the 134 chunks carry a pair, and the other 107 are meant to have none.** A source name
query takes its document's first chunk, which is where the name, the framing and the stack live.
Every phrase the 9 failing eval queries need was checked against that: `1B+ daily records`,
`Snowflake`, `blockchain`, `Terraform` and `Business Development` all appear in a covered chunk,
and the uncovered hits are second mentions inside the same document. Adding pairs for body chunks
would mean inventing a query per chunk, which nothing measured yet asks for. The count is asserted
so a silent drop gets caught.

`retrieval_set.json` is not touched. Those 66 hand-written queries are the only denominator for
Hit Rate and MRR, so training on them would make the metric measure itself. Synthetic pairs are
training only and are never reported as an evaluation number. See D49.

    python scripts/make_train_pairs.py            # writes data/train_pairs.json
    python scripts/make_train_pairs.py --sample 20  # prints 20 pairs to read by hand
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunk import chunk_docs  # noqa: E402
from rag.sources import get_docs  # noqa: E402

OUT_PATH = ROOT / "data" / "train_pairs.json"

# Pairs are built on variant B, the production index is variant A. See the module docstring.
PAIR_VARIANT = "B"

# Both languages: 12 of the 66 eval queries are Indonesian and D47 makes a drop there a kill.
LABEL_QUERIES = {
    "Education": [
        "Where did Firza study?",
        "Which university did Firza attend?",
        "What is Firza's degree?",
        "Did Firza go to college?",
        "What did Firza major in?",
        "Di mana Firza kuliah?",
        "Apa latar belakang pendidikan Firza?",
        "Firza lulusan mana?",
        "Firza kuliah jurusan apa?",
    ],
    "Certifications": [
        "What courses has Firza taken?",
        "Is Firza certified in anything?",
        "What training has Firza completed?",
        "Sertifikat apa yang dimiliki Firza?",
        "Firza pernah ikut kursus apa?",
    ],
    "Languages": [
        "What languages does Firza speak?",
        "Can Firza speak English?",
        "Does Firza know Japanese?",
        "Firza bisa bahasa apa saja?",
    ],
    "Skills": [
        "What tools does Firza use?",
        "What is Firza good at?",
        "Which technologies does Firza know?",
        "Does Firza know SQL?",
        "Keahlian Firza apa saja?",
        "Firza bisa pakai tools apa?",
    ],
    "Leadership": [
        "Has Firza led a team?",
        "What organisations was Firza part of?",
        "Does Firza have leadership experience?",
        "Apakah Firza pernah memimpin tim?",
    ],
    "Achievements": [
        "What has Firza won?",
        "Any awards or scholarships?",
        "What is Firza proudest of?",
        "Prestasi Firza apa saja?",
    ],
}

# Queries for a whole document, keyed by how its source name starts.
EXPERIENCE_QUERIES = [
    "What did Firza do at {company}?",
    "Tell me about Firza's work at {company}.",
    "What was Firza's role at {company}?",
    "What did Firza build at {company}?",
    "Firza ngerjain apa di {company}?",
    "Apa tanggung jawab Firza di {company}?",
]

PROJECT_QUERIES = [
    "Tell me about {name}.",
    "What was {name} about?",
    "How did Firza build {name}?",
    "What problem did {name} solve?",
    "Ceritakan tentang {name}.",
    "Apa itu {name}?",
]

TIMELINE_QUERIES = [
    "Where does Firza work now?",
    "What is Firza's current role?",
    "Walk me through Firza's career.",
    "How many years of experience does Firza have?",
    "Firza sekarang kerja di mana?",
    "Berapa lama Firza sudah bekerja?",
]

ABOUT_QUERIES = [
    "Who is Firza?",
    "Tell me about Firza.",
    "What does Firza do?",
    "Firza itu siapa?",
    "Apa pekerjaan Firza?",
]


# A word only the intended sense of an ambiguous label carries. See the docstring and D90.
LABEL_MUST_CONTAIN = {"Languages": "Indonesia"}


def _labels_of(text):
    """Return every LABEL_QUERIES label present in a chunk, paired with how much of it that
    label's own paragraph occupies. A chunk is a positive for a label only if the label owns
    enough of it, or the pair teaches the model that a career chunk answers an education query.
    """
    out = []
    for label in LABEL_QUERIES:
        m = re.search(rf"\b{label}:", text)
        if not m:
            continue
        need = LABEL_MUST_CONTAIN.get(label)
        if need and need not in text[m.start():m.start() + 200]:
            continue
        # The label's paragraph runs to the next blank line or the next label, whichever comes first.
        rest = text[m.start():]
        nxt = min([p for p in (rest.find("\n\n"), *(rest.find(f"{l}:", 1) for l in LABEL_QUERIES)) if p > 0]
                  or [len(rest)])
        out.append((label, nxt / len(text)))
    return out


# How much of a chunk a label must own to be its positive. See the docstring and D89.
MIN_LABEL_SHARE = 0.30

# Below this a paragraph cannot reach MIN_LABEL_SHARE, so it keeps its chunk. See the docstring.
SHORT_LABEL_CHARS = 200


def _company_of(source):
    """Pull the company out of `Experience: <role> at <company>`."""
    return source.split(" at ", 1)[1] if " at " in source else None


def build_pairs():
    """Return [{query, text, source, template}] over every chunk the index will hold."""
    chunks = chunk_docs(get_docs(), variant=PAIR_VARIANT)
    pairs = []
    seen_sources = set()

    def add(query, chunk, template):
        pairs.append({"query": query, "text": chunk["text"], "source": chunk["source"],
                      "template": template})

        seen_sources = set()

    for c in chunks:
        source, text = c["source"], c["text"]

        # Labels beat the source name: education and skills share one document. See D89.
        labels = [(l, share) for l, share in _labels_of(text)
                  if share >= MIN_LABEL_SHARE or share * len(text) < SHORT_LABEL_CHARS]
        if labels:
            for label, _ in labels:
                for q in LABEL_QUERIES[label]:
                    add(q, c, f"label:{label}")
            continue

        # A source name query names the document, so only its first chunk may claim it. See D90.
        if source in seen_sources:
            continue
        seen_sources.add(source)

        if source.startswith("Experience:"):
            company = _company_of(source)
            if company:
                for t in EXPERIENCE_QUERIES:
                    add(t.format(company=company), c, "experience")
        elif source.startswith("Project:"):
            name = source.split(": ", 1)[1]
            for t in PROJECT_QUERIES:
                add(t.format(name=name), c, "project")
        elif source == "Career Timeline":
            for q in TIMELINE_QUERIES:
                add(q, c, "timeline")
        elif source == "About Firza":
            for q in ABOUT_QUERIES:
                add(q, c, "about")

    return pairs


def _overlap_with_eval(pairs):
    """Count pairs whose query also appears in retrieval_set.json. Must be zero. See D49."""
    path = ROOT / "scripts" / "retrieval_set.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw["queries"] if isinstance(raw, dict) and "queries" in raw else raw
    eval_qs = {(c.get("query") or c.get("question") or "").strip().lower() for c in cases}
    return [p for p in pairs if p["query"].strip().lower() in eval_qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="print N random pairs instead of writing")
    args = ap.parse_args()

    pairs = build_pairs()

    # Training on an eval query makes Hit Rate measure itself. Refuse rather than warn. See D49.
    leaked = _overlap_with_eval(pairs)
    if leaked:
        print(f"ERROR: {len(leaked)} training queries also appear in retrieval_set.json:")
        for p in leaked[:10]:
            print(f"  {p['query']}")
        sys.exit(1)

    if args.sample:
        random.seed(1)
        for p in random.sample(pairs, min(args.sample, len(pairs))):
            print(f"\n[{p['template']}]  {p['query']}")
            print(f"  -> {p['source']}: {p['text'][:120]}...")
        return

    by_template = Counter(p["template"] for p in pairs)
    covered = {p["source"] for p in pairs}
    all_sources = {c["source"] for c in chunk_docs(get_docs(), variant=PAIR_VARIANT)}

    OUT_PATH.write_text(json.dumps({
        "_comment": "Phase 5c training pairs, template generated. Training only, never an eval number. See D49.",
        "pairs": pairs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(pairs)} pairs over {len(covered)} of {len(all_sources)} sources")
    for t, n in by_template.most_common():
        print(f"  {n:4d}  {t}")
    missing = all_sources - covered
    if missing:
        print(f"\nWARNING: {len(missing)} sources got no pairs:")
        for m in sorted(missing):
            print(f"  {m}")
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    # A template must not contain its own label, or it trains phrasing the model already handles.
    for q in LABEL_QUERIES["Education"]:
        assert "education" not in q.lower(), q
    for q in LABEL_QUERIES["Skills"]:
        assert "skills" not in q.lower(), q

    pairs = build_pairs()
    assert pairs, "no pairs generated"
    assert all(p["query"] and p["text"] for p in pairs), "an empty query or text was produced"

    # The education chunk must actually be reachable, or the whole fine-tune targets nothing.
    edu = [p for p in pairs if p["template"] == "label:Education"]
    assert edu, "no education pairs, the label regex missed"
    assert any("Bandung Institute of Technology" in p["text"] for p in edu), "education pairs miss ITB"

    # Variant A packs education into a 1200 char career chunk, so a clean positive proves B ran.
    assert all(len(p["text"]) < 600 for p in edu), max(len(p["text"]) for p in edu)
    assert all(p["text"].startswith("Education:") for p in edu), edu[0]["text"][:60]

    # Every query must be unique per chunk, since a duplicate pair is a silent weight on one example.
    seen = {(p["query"], p["text"]) for p in pairs}
    assert len(seen) == len(pairs), f"{len(pairs) - len(seen)} duplicate pairs"

    # One positive per query: a project query once matched 14 chunks of its document. See D90.
    counts = Counter(p["query"] for p in pairs)
    worst = counts.most_common(1)[0]
    assert worst[1] == 1, f"{worst[0]!r} has {worst[1]} positives"

    # Spoken languages, never `Languages: Python, SQL` from the skills block. See D90.
    langs = [p for p in pairs if p["template"] == "label:Languages"]
    assert langs, "no Languages pairs, the disambiguation rejected both senses"
    assert all("JLPT" in p["text"] for p in langs), langs[0]["text"][:80]

    # 27 of 134 chunks carry a pair, deliberately. See the docstring.
    assert len({p["text"] for p in pairs}) == 27, len({p["text"] for p in pairs})

    # Every label must reach training, or its queries leave the fine-tune silently. See D90.
    got = {t.split(":", 1)[1] for t in (p["template"] for p in pairs) if t.startswith("label:")}
    assert got == set(LABEL_QUERIES), f"labels with no pairs: {sorted(set(LABEL_QUERIES) - got)}"

    main()
