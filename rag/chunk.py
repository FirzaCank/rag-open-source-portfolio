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


def chunk_docs(docs: list[dict], target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[dict]:
    """Chunk a whole corpus, preserving document order."""
    out: list[dict] = []
    for d in docs:
        out.extend(chunk_text(d["source"], d["text"], target, overlap))
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

    chunks = chunk_docs(get_docs())
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
