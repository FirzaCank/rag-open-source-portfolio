"""Generate scripts/retrieval_worksheet.md: one file to write the 66 queries in.

Editing a 66 entry JSON while reading the corpus in a second window is the slow part of Phase 1
Step 4, and a stray comma breaks the file. This puts each source's content and its three empty
slots in the same place, top to bottom, with no punctuation that can break.

Fill the `Q1:` `Q2:` `Q3:` lines, then run scripts/worksheet_to_json.py.

Run: python scripts/make_worksheet.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag.chunk import chunk_docs
from rag.sources import get_docs
from scripts.corpus_digest import slug

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_worksheet.md")
PREVIEW_CHARS = 700

# One Indonesian slot in 12 of the 22 sources, spread over roles and projects. See the skeleton.
ID_SLOT_SOURCES_COUNT = 12


def main() -> None:
    docs = get_docs()
    chunks = chunk_docs(docs)
    order: list[str] = []
    for d in docs:
        if d["source"] not in order:
            order.append(d["source"])

    id_sources = set(order[:7]) | {order[7], order[13], order[16], order[19], order[21]}
    assert len(id_sources) == ID_SLOT_SOURCES_COUNT, len(id_sources)

    lines = [
        "# Retrieval set worksheet, 66 queries",
        "",
        "Fill every `Q1:` `Q2:` `Q3:` line, then run:",
        "",
        "```bash",
        "python scripts/worksheet_to_json.py",
        "```",
        "",
        "That converts this file into `scripts/retrieval_set.json` and runs the checker.",
        "",
        "## Rules",
        "",
        "- Write like a visitor who has never read the portfolio. That is the real query population.",
        "- Never reuse the source's own phrasing. A query that quotes its chunk measures string",
        "  matching, and it wins under every retrieval strategy, so it separates nothing.",
        "- Three different angles per source. One direct, one indirect, one about a number or a skill.",
        "- `(ID)` slots are Bahasa Indonesia. 12 of them, one in each of 12 sources.",
        "",
        "Real example of why the phrasing rule matters: a visitor writes \"right now\", the corpus",
        "writes \"current\", and that single difference pushed the Hypefast chunks to rank 17.",
        "",
        "---",
        "",
    ]

    for i, source in enumerate(order, 1):
        mine = [d for d in docs if d["source"] == source]
        n_chunks = sum(1 for c in chunks if c["source"] == source)
        chars = sum(len(d["text"]) for d in mine)
        preview = mine[0]["text"][:PREVIEW_CHARS].replace("\n", " ")
        more = "" if len(mine[0]["text"]) <= PREVIEW_CHARS else " ..."

        lines += [
            f"## {i}. {source}",
            "",
            f"`{slug(source)}` | {len(mine)} doc, {n_chunks} chunk, {chars} chars",
            "",
            f"> {preview}{more}",
            "",
        ]
        if len(mine) > 1:
            lines += [f"Other documents under this source: {len(mine) - 1}, "
                      f"one per highlight plus an overview.", ""]
        lines += [
            "```",
            "Q1: ",
            "Q2: ",
            f"Q3{' (ID)' if source in id_sources else ''}: ",
            "```",
            "",
        ]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    n_id = sum(1 for s in order if s in id_sources)
    print(f"wrote scripts/retrieval_worksheet.md: {len(order)} sources, {len(order) * 3} slots, "
          f"{n_id} of them Indonesian")


if __name__ == "__main__":
    main()
