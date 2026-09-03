"""Split documents into overlapping chunks for embedding.

A port of `lib/rag/chunk.ts` from the source repository, faithful to what that file actually does
rather than to what its comments claim. The numbers are inherited, not chosen: 1200 characters is
roughly 300 tokens, and 180 characters of overlap is about 15 percent.

This is variant A in the Phase 5 chunking comparison, and it is fixed size chunking. The
TypeScript opens with `raw.replace(/\\s+\\n/g, "\\n")`, and `\\s+` consumes newlines too, so every
blank line collapses to a single newline before the paragraph split ever runs. With no `\\n\\n` left
in the text, every long document falls through to the hard split branch. Its own comment says
"preferring paragraph boundaries", and that is not what happens.

Verified against the production index, `data/embeddings.json` in the source repo: 37 of its 76
chunks are exactly 1200 characters and none of them contains a blank line. Paragraph aware
chunking is therefore a real variant to compare against in Phase 5, not a tweak.

Chunks carry no id. The production index stores `{source, text, vector}`, because
`build-embeddings.ts` drops the id after chunking. Nothing reads it, and D28 measures per source
precisely because chunk ids shift the moment the chunk size changes.

Run `python3 rag/chunk.py` for the counts and the size distribution.
"""

import os
import re

TARGET_CHARS = 1200
OVERLAP_CHARS = 180


def chunk_text(source: str, raw: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[dict]:
    """Split one document into overlapping windows of at most `target` characters.

    The first substitution is the one that decides the behaviour of this whole function, so it is
    ported exactly: `\\s+\\n` matches any run of whitespace ending in a newline, newlines included,
    which flattens paragraph breaks. Keeping it means variant A here behaves like variant A in
    production. Fixing it would be a different variant, and Phase 5 is where variants get compared,
    not where they get quietly swapped.
    """
    text = re.sub(r"\s+\n", "\n", raw)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) <= target:
        return [{"source": source, "text": text}] if text else []

    chunks: list[dict] = []
    buf = ""

    def flush():
        nonlocal buf
        t = buf.strip()
        if t:
            chunks.append({"source": source, "text": t})

    for para in [p for p in re.split(r"\n\n+", text) if p != ""]:
        if buf and len(buf) + len(para) + 2 > target:
            flush()
            # The tail of the flushed chunk becomes the head of the next one. That is the overlap.
            buf = buf[-overlap:] + "\n\n" + para
        else:
            buf = buf + "\n\n" + para if buf else para

        while len(buf) > target:
            chunks.append({"source": source, "text": buf[:target]})
            buf = buf[target - overlap:]

    flush()
    return chunks


# Below this a paragraph is merged forward: `Languages:` is 77 chars, too thin to answer from.
MIN_PARA_CHARS = 200


def chunk_text_b(source: str, raw: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[dict]:
    """Variant B, paragraph aware. One topic per chunk instead of packing to fill `target`.

    Two changes from variant A. The whitespace substitution is `[ \\t]+\\n` rather than `\\s+\\n`, so
    blank lines survive and the paragraph split actually runs. And a paragraph long enough to stand
    alone is never merged with the next one just to reach `target`.

    **Why it exists, measured.** Variant A puts the 255 character `Education:` paragraph into a 1200
    character chunk whose other 921 characters are career achievements, so the paragraph owns 0.21 of
    the chunk it lives in. Under variant B it owns 1.00. That share is what decides whether a
    Phase 5c training pair has a clean positive to point at: training a query about education
    against a chunk that is 79 percent career text teaches the wrong thing. Certifications goes 0.20
    to 1.00, Leadership 0.49 to 1.00, Achievements 0.37 to 1.00, Skills 0.61 to 0.91. See D89.

    **Rejected as a production index, measured.** Built and evaluated over the 66 query retrieval
    set: HR@1 falls 68.2 to 59.1 and MRR 0.7497 to 0.7005, which fires the D47 kill criterion. The
    English split takes the damage, 68.5 to 53.7, while Indonesian rises 66.7 to 83.3. HR@8 rises
    slightly, 86.4 to 87.9, so the correct chunk is still found and only its rank slips: 76 chunks
    become 134, and a query's top hit now loses to a sibling chunk from the same document.

    **Kept for training pairs only.** `scripts/make_train_pairs.py` calls this with `variant="B"`
    because a positive has to be a clean one, and the index stays variant A. The `Education:`
    sentence exists in the variant A index too, inside a 1200 character chunk, so a fine-tune that
    pulls "Where did Firza study?" toward that sentence works without changing the index. See D90.
    """
    text = re.sub(r"[ \t]+\n", "\n", raw)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= target:
        return [{"source": source, "text": text}]

    out: list[str] = []
    buf = ""
    for para in [p for p in re.split(r"\n\n+", text) if p.strip()]:
        if buf and len(buf) < MIN_PARA_CHARS:
            buf = buf + "\n\n" + para
        else:
            if buf:
                out.append(buf)
            buf = para

        while len(buf) > target:
            # Cut on a space so no chunk starts mid word. Variant A split `Institute` in half.
            cut = buf.rfind(" ", target - 150, target)
            cut = cut if cut > 0 else target
            out.append(buf[:cut].strip())
            buf = buf[max(0, cut - overlap):].lstrip()

    if buf:
        out.append(buf)
    return [{"source": source, "text": t.strip()} for t in out if t.strip()]


# Which chunker builds the index, off the environment so an arm is a flag not an edit. See D89.
VARIANT = os.environ.get("CHUNK_VARIANT", "A").strip().upper()
_CHUNKERS = {"A": chunk_text, "B": chunk_text_b}


def chunk_docs(docs: list[dict], target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS,
               variant: str | None = None) -> list[dict]:
    """Chunk a whole corpus, preserving document order. `variant` defaults to CHUNK_VARIANT."""
    key = (variant or VARIANT).upper()
    if key not in _CHUNKERS:
        raise ValueError(f"CHUNK_VARIANT is {key!r}, expected one of {sorted(_CHUNKERS)}")
    fn = _CHUNKERS[key]
    out: list[dict] = []
    for d in docs:
        out.extend(fn(d["source"], d["text"], target, overlap))
    return out


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sources import get_docs

    # Behaviour checks on text small enough to verify by hand.
    assert chunk_text("S", "") == []
    assert chunk_text("S", "   \n\n  ") == []
    assert chunk_text("S", "short text") == [{"source": "S", "text": "short text"}]

    # Hard split with overlap: without it a sentence on a seam is lost from both sides.
    hard = chunk_text("S", "x" * 2500, target=1000, overlap=100)
    assert len(hard) == 3, len(hard)
    assert all(len(c["text"]) <= 1000 for c in hard)
    seam = chunk_text("S", "a" * 700 + "\n\n" + "b" * 700, target=1000, overlap=100)
    assert len(seam) == 2, len(seam)
    # The next chunk opens with the previous one's tail, since the split lands mid paragraph.
    assert seam[1]["text"][:100] == seam[0]["text"][-100:], "the overlap did not carry across the seam"

    # Blank lines must be gone, so a future regex fix becomes a deliberate variant, not a silent one.
    flattened = chunk_text("S", "p1\n\np2\n\np3")
    assert "\n\n" not in flattened[0]["text"], "blank lines survived, this is no longer variant A"

    # Variant B keeps blank lines, or its paragraph split never runs and it is variant A again.
    kept = chunk_text_b("S", "x" * 900 + "\n\n" + "y" * 900, target=1200, overlap=180)
    assert len(kept) == 2, len(kept)
    assert kept[0]["text"].startswith("x") and kept[1]["text"].startswith("y"), "paragraphs merged"

    # No chunk may start mid word. Variant A split `Institute` across a boundary. See D89.
    b_docs = chunk_docs(get_docs(), variant="B")
    edu = [c for c in b_docs if c["text"].startswith("Education:")]
    assert edu, "no chunk starts at the Education paragraph, variant B did not split on it"
    assert "Bandung Institute of Technology" in edu[0]["text"], edu[0]["text"][:80]

    # An unknown variant must fail loudly rather than silently building the wrong index.
    try:
        chunk_docs(get_docs(), variant="Z")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown variant built an index instead of raising")

    chunks = chunk_docs(get_docs(), variant="A")
    sources = sorted({c["source"] for c in chunks})
    sizes = sorted(len(c["text"]) for c in chunks)
    full = sum(1 for s in sizes if s == TARGET_CHARS)

    # Matched against the original index: equal numbers are the evidence this reproduces it.
    assert len(chunks) == 76, f"expected 76 chunks, got {len(chunks)}"
    assert len(sources) == 22, f"expected 22 sources, got {len(sources)}"
    assert full == 37, f"expected 37 chunks of exactly {TARGET_CHARS} chars, got {full}"
    assert sizes[0] == 220, f"expected shortest chunk 220 chars, got {sizes[0]}"
    assert all(c["text"] for c in chunks), "an empty chunk was produced"

    print(f"{len(chunks)} chunks across {len(sources)} sources")
    print(f"sizes: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}, "
          f"{full} at exactly {TARGET_CHARS}")
    print(f"total {sum(sizes)} characters, {sum(sizes) / len(sizes):.0f} average")
    print("\nchunks per source:")
    for s in sources:
        n = sum(1 for c in chunks if c["source"] == s)
        print(f"  {n:2d}  {s}")
