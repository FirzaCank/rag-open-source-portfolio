"""Print what each of the 22 sources actually contains, for writing the retrieval eval set by hand.

Phase 1 Step 4 asks for 3 queries per source, written without copying the chunk's own phrasing. To
do that you need to know what a source covers without reading its raw text closely enough to start
echoing it. This prints a digest: how many chunks, how long, and the opening of each document.

Reused in Phase 5b when the case set is extended.

Run: python scripts/corpus_digest.py
     python scripts/corpus_digest.py --full    to print every document in full
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag.chunk import chunk_docs
from rag.sources import get_docs


def slug(source: str) -> str:
    """A short handle for a source, used as the id prefix in retrieval_set.json.

    For an experience source the company is the distinguishing part, not the job title: three of
    the five roles are some flavour of data engineer. For a project it is the title.
    """
    s = source
    if s.startswith("Experience:") and " at " in s:
        s = s.split(" at ", 1)[1]
    elif s.startswith("Project:"):
        s = s.split(":", 1)[1]
    words = ["".join(c for c in w.lower() if c.isalnum()) for w in s.split()]
    return "-".join(w for w in words if w)[:28].strip("-")


def main() -> None:
    full = "--full" in sys.argv
    docs = get_docs()
    chunks = chunk_docs(docs)

    order: list[str] = []
    for d in docs:
        if d["source"] not in order:
            order.append(d["source"])

    print(f"{len(docs)} documents, {len(chunks)} chunks, {len(order)} sources\n")
    for i, source in enumerate(order, 1):
        mine = [d for d in docs if d["source"] == source]
        n_chunks = sum(1 for c in chunks if c["source"] == source)
        chars = sum(len(d["text"]) for d in mine)
        print("=" * 100)
        print(f"[{i:2d}/{len(order)}]  {source}")
        print(f"        slug {slug(source)}  |  {len(mine)} doc, {n_chunks} chunk, {chars} chars")
        print("=" * 100)
        for j, d in enumerate(mine, 1):
            body = d["text"] if full else d["text"][:400]
            tail = "" if full or len(d["text"]) <= 400 else f"\n        ... +{len(d['text']) - 400} chars"
            print(f"  doc {j}:")
            for line in (body + tail).split("\n"):
                print(f"        {line}")
            print()


if __name__ == "__main__":
    main()
