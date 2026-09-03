"""Turn data/portfolio.json into labeled documents, ready for chunking.

This is a port of `lib/rag/sources.ts` from the source repository. The document boundaries it
produces must match that file exactly, because the Gemini comparison in D12 only means something if
both systems retrieve over the same units of text.

A "document" here is one piece of portfolio content plus a human readable `source` label, for
example `Experience: Data Engineer at Hypefast`. The label is what the model cites, and it is the
key the retrieval eval set in Phase 1 Step 4 labels its queries with. Labels are stable across
chunking variants, which is why D28 measures per source and never per chunk id.

Why Python reads the JSON instead of running the TypeScript: `scripts/export-portfolio.ts` in the
source repo exports the raw data, not the document boundaries, so the boundaries had to be
reproduced somewhere. Doing it here removes Node from this project entirely. Everything
`sources.ts` reads is present in `portfolio.json`, verified field by field before this file was
written.

Run `python3 rag/sources.py` to see the counts and a preview.
"""

import json
import os
import re

# Resolved against this file, not the working directory, so imports and direct runs behave alike.
_HERE = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_PATH = os.path.join(_HERE, "..", "data", "portfolio.json")


def strip_mdx(s: str) -> str:
    """Remove MDX markup so the embedder sees prose, not syntax.

    Ported step for step from `stripMdx` in sources.ts, including the order of the replacements.
    The order matters: HTML-like tags go first, so a fence sitting inside a JSX block is already
    partly flattened by the time the fence pattern runs. Reproducing the quirk is the point, since
    a different cleanup produces different chunk text and quietly breaks the comparison.
    """
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)  # [\s\S] in JS, DOTALL here
    s = re.sub(r"[#*_`>|]", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def load_portfolio(path: str = PORTFOLIO_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_docs(data: dict | None = None) -> list[dict]:
    """Return `[{"source": str, "text": str}, ...]` in the same order as sources.ts.

    Four kinds of document, and the shape of each is a retrieval decision, not a formatting one:

    1. One `About Firza` document holding bio, education, certifications, languages, skills,
       leadership, and achievements.
    2. One `Career Timeline` document listing every role oldest to newest. It exists so that
       questions like "what was his first job" have something to hit. No single role document can
       answer a question about the sequence of roles.
    3. Per role: one document per highlight, so retrieval can land on a specific achievement
       instead of a wall of text, plus one overview document for broad questions about the role.
    4. One document per project, carrying the case study body.
    """
    data = data if data is not None else load_portfolio()
    about = data["about"]
    roles = data["experience"]
    projects = data["projects"]

    docs: list[dict] = []

    education = "; ".join(
        f"{e['degree']}, {e['faculty']}, {e['institution']} ({e['period']}). "
        f"Thesis: {(e.get('thesis') or {}).get('title', '')}"
        for e in about["education"]
    )
    certifications = "; ".join(f"{c['name']} ({c['issuer']})" for c in about["certifications"])
    languages = "; ".join(f"{l['name']}: {l['level']}" for l in about["languages"])
    skills = ". ".join(f"{g['group']}: {', '.join(g['items'])}" for g in about["skills"])
    leadership = " ".join(
        f"{l['role']} at {l['organization']} ({l['context']}): {' '.join(l['highlights'])}"
        for l in about["leadership"]
    )
    achievements = "; ".join(
        f"{a['title']}, {a['context']} ({a['issuer']})" for a in about["achievements"]
    )

    docs.append({
        "source": "About Firza",
        "text": "\n\n".join([
            "\n".join(about["bio"]),
            "Education: " + education,
            "Certifications: " + certifications,
            "Languages: " + languages,
            "Skills: " + skills,
            "Leadership: " + leadership,
            "Achievements: " + achievements,
        ]),
    })

    # Oldest first. The JSON lists roles newest first, matching how the website renders them.
    timeline = "\n".join(
        f"{i + 1}. {r['title']} at {r['company']}, {r['period']}"
        f"{' (internship)' if r.get('internship') else ''}"
        f"{' (current)' if r.get('current') else ''}."
        for i, r in enumerate(reversed(roles))
    )
    docs.append({
        "source": "Career Timeline",
        "text": "Firza's complete career history in chronological order (earliest to latest):\n" + timeline,
    })

    for r in roles:
        placement = f" ({r['placement']})" if r.get("placement") else ""
        current = " (current role)" if r.get("current") else ""
        header = (
            f"{r['title']} at {r['company']}{placement}, {r['period']}{current}. "
            f"Location: {r['location']}."
        )
        source = f"Experience: {r['title']} at {r['company']}"
        stack = ", ".join(r["stack"])
        for highlight in r["highlights"]:
            docs.append({
                "source": source,
                "text": f"{header}\n{r['summary']}\n{highlight}\nStack: {stack}.",
            })
        docs.append({
            "source": source,
            "text": f"{header}\n{r['summary']}\nStack: {stack}.",
        })

    for p in projects:
        parts = [
            f"{p['title']} ({p['year']}) for {p['client']}. {p['subtitle']}",
            f"Categories: {', '.join(p['categories'])}. Stack: {', '.join(p['stack'])}.",
            strip_mdx(p["detail"]),
        ]
        docs.append({
            "source": f"Project: {p['title']}",
            "text": "\n\n".join(part for part in parts if part),
        })

    return docs


if __name__ == "__main__":
    docs = get_docs()
    sources = sorted({d["source"] for d in docs})

    # 39 docs over 22 sources, asserted so a corpus edit that drops content fails here.
    assert len(docs) == 39, f"expected 39 documents, got {len(docs)}"
    assert len(sources) == 22, f"expected 22 sources, got {len(sources)}"
    assert all(d["text"].strip() for d in docs), "a document came out empty"
    assert not any("```" in d["text"] for d in docs), "an unstripped code fence survived"

    chars = sum(len(d["text"]) for d in docs)
    print(f"{len(docs)} documents, {len(sources)} sources, {chars} characters total")
    print(f"shortest {min(len(d['text']) for d in docs)} chars, "
          f"longest {max(len(d['text']) for d in docs)} chars")
    print("\nsources:")
    for s in sources:
        n = sum(1 for d in docs if d["source"] == s)
        print(f"  {n:2d} doc  {s}")
    print("\nfirst 300 characters of the About document:\n")
    print(docs[0]["text"][:300])
