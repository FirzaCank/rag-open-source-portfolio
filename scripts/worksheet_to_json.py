"""Convert scripts/retrieval_worksheet.md into scripts/retrieval_set.json, then run the checker.

The worksheet is the file a human edits. This is the only thing that writes the JSON, so no hand
editing of 66 entries and no broken commas. Rerun it as often as you like: it overwrites the JSON
from the worksheet every time.

Refuses to overwrite a frozen JSON, since freezing is the point after which the set stops changing.

Run: python scripts/worksheet_to_json.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag.sources import get_docs
from scripts.corpus_digest import slug

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSHEET = os.path.join(HERE, "retrieval_worksheet.md")
OUT = os.path.join(HERE, "retrieval_set.json")

COMMENT = (
    "66 hand written retrieval queries, 3 per source across 22 sources, 12 of them in Indonesian. "
    "The only denominator for Hit Rate and MRR, see D49. Labeled per source, never per chunk id, "
    "see D28. Generated from retrieval_worksheet.md by scripts/worksheet_to_json.py, never edited "
    "by hand. Frozen once complete, see D41."
)


def parse() -> list[dict]:
    """Read the worksheet and return one entry per filled or empty Q slot."""
    with open(WORKSHEET, encoding="utf-8") as f:
        lines = f.read().split("\n")

    source_names = []
    for d in get_docs():
        if d["source"] not in source_names:
            source_names.append(d["source"])
    by_heading = {f"{i}. {s}": s for i, s in enumerate(source_names, 1)}

    entries: list[dict] = []
    current: str | None = None
    for raw in lines:
        line = raw.strip()

        heading = re.match(r"^##\s+(\d+\.\s+.+)$", line)
        if heading:
            key = heading.group(1).strip()
            current = by_heading.get(key)
            if key not in by_heading and re.match(r"^\d+\.", key):
                raise SystemExit(f"heading does not match any corpus source: {key!r}")
            continue

        slot = re.match(r"^Q([123])(\s*\(ID\))?\s*:\s*(.*)$", line)
        if slot and current:
            n, is_id, text = slot.group(1), bool(slot.group(2)), slot.group(3).strip()
            entries.append({
                "id": f"{slug(current)}-{n}",
                "query": text,
                "source": current,
                "lang": "id" if is_id else "en",
            })

    return entries


def main() -> int:
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            if json.load(f).get("frozen"):
                print("retrieval_set.json is frozen. Refusing to overwrite it from the worksheet.")
                print("If a change is genuinely needed, set frozen to false first and say why in "
                      "docs/DECISIONS.md.")
                return 1

    entries = parse()
    if len(entries) != 66:
        print(f"found {len(entries)} Q slots in the worksheet, expected 66. "
              "A heading or a Q line was probably edited or deleted.")
        return 1

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"_comment": COMMENT, "frozen": False, "queries": entries}, f,
                  ensure_ascii=False, indent=2)

    filled = sum(1 for e in entries if e["query"])
    print(f"wrote scripts/retrieval_set.json: {filled}/66 filled, "
          f"{sum(1 for e in entries if e['lang'] == 'id')} Indonesian\n")

    from scripts.check_retrieval_set import main as check
    return check()


if __name__ == "__main__":
    sys.exit(main())
