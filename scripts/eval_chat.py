"""Golden set eval for the RAG chat pipeline.

Runs every case in scripts/golden_set.json through rag.llm.run_chat, no HTTP, and checks the
answer against the frozen regexes. Exit code 1 if anything fails.

    python scripts/eval_chat.py                     # all 41 cases
    python scripts/eval_chat.py --category injection
    python scripts/eval_chat.py --only sens-salary
    python scripts/eval_chat.py --verbose
    python scripts/eval_chat.py --label "3b, not baseline"

Ported from the source repository. Five changes:

1. No API key and no `--sleep`. The model is local, so there is no quota to pace against.
2. No send stub. The original patched `_post_contact` so an eval run could not email Firza. Now
   `CONTACT_ENDPOINT` does that for every caller, and this script refuses to run when it is set.
   An env var guards 5 callers, a stub guarded 1. See D29.
3. Per case wall time and time to first token, which Phase 3b needs. See D33.
4. Per case `prompt_tokens` and `truncation_risk` from the Ollama response. A truncated prompt
   answers blank with no error, so it has to be a number in the output. See D61.
5. Every run writes a JSON file under `eval_runs/`, labeled. A 3b run on this laptop is never a
   baseline and never compared with Gemini, so the label travels with the numbers. See D24.

The regexes are frozen. They were written for Gemini's phrasing, so a correct Qwen answer can fail
on wording. Do not loosen them, classify the failure by hand as wrong-fact or wrong-wording
instead. See D25.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.llm import MODEL, NUM_CTX, run_chat  # noqa: E402

GOLDEN_PATH = ROOT / "scripts/golden_set.json"
OUT_DIR = ROOT / "eval_runs"


def check(case, answer):
    """Return a list of failure descriptions. Empty means the case passed."""
    failures = []
    for pattern in case.get("must_match", []):
        if not re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"must_match missed: /{pattern}/")
    any_patterns = case.get("any_match", [])
    if any_patterns and not any(re.search(p, answer, re.IGNORECASE) for p in any_patterns):
        failures.append(f"any_match: none of {any_patterns} matched")
    for pattern in case.get("must_not_match", []):
        if re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"must_not_match hit: /{pattern}/")
    max_chars = case.get("max_chars")
    if max_chars and len(answer) > max_chars:
        failures.append(f"too long: {len(answer)} chars > {max_chars}")
    return failures


def run_case(case):
    """Run one case and return its record: answer, verdict, timings, and model stats."""
    messages = case.get("messages") or [{"role": "user", "content": case["question"]}]
    stats = {}
    t0 = time.time()
    ttft = None
    parts = []
    try:
        for chunk in run_chat(messages, stats):
            if ttft is None:
                ttft = time.time() - t0
            parts.append(chunk)
    except Exception as e:
        return {"id": case["id"], "category": case["category"], "error": str(e)[:300],
                "wall_s": round(time.time() - t0, 2)}

    answer = "".join(parts)
    return {
        "id": case["id"],
        "category": case["category"],
        "answer": answer,
        "failures": check(case, answer),
        "wall_s": round(time.time() - t0, 2),
        "ttft_s": round(ttft, 2) if ttft is not None else None,
        "chars": len(answer),
        "chunks": stats.get("chunks"),
        "top_score": stats.get("top_score"),
        "rounds": stats.get("rounds"),
        "tools": stats.get("tools"),
        "sent": stats.get("sent"),
        "prompt_tokens": stats.get("prompt_tokens"),
        "output_tokens": stats.get("output_tokens"),
        "truncation_risk": bool(stats.get("truncation_risk")),
    }


def summarise(records, label, elapsed):
    """Print the run summary and return the object written to disk."""
    passed = [r for r in records if not r.get("error") and not r["failures"]]
    failed = [r for r in records if not r.get("error") and r["failures"]]
    errored = [r for r in records if r.get("error")]

    by_cat = {}
    for r in records:
        cat = by_cat.setdefault(r["category"], {"n": 0, "pass": 0})
        cat["n"] += 1
        if not r.get("error") and not r["failures"]:
            cat["pass"] += 1

    walls = [r["wall_s"] for r in records if r.get("wall_s")]
    ttfts = [r["ttft_s"] for r in records if r.get("ttft_s")]
    prompts = [r["prompt_tokens"] for r in records if r.get("prompt_tokens")]

    print(f"\n{len(passed)}/{len(records)} passed" + (f", {len(errored)} errored" if errored else ""))
    print(f"label: {label}  model: {MODEL}  num_ctx: {NUM_CTX}  total: {elapsed / 60:.1f} min\n")

    print(f"{'category':<14} {'pass':>5} {'of':>4}")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"{cat:<14} {c['pass']:>5} {c['n']:>4}")

    if walls:
        print(f"\nwall seconds   median {statistics.median(walls):.1f}  max {max(walls):.1f}")
    if ttfts:
        print(f"first token    median {statistics.median(ttfts):.1f}  max {max(ttfts):.1f}")
    if prompts:
        print(f"prompt tokens  median {statistics.median(prompts):.0f}  max {max(prompts)} of {NUM_CTX}")

    risky = [r["id"] for r in records if r.get("truncation_risk")]
    if risky:
        print(f"\nTRUNCATION RISK, prompt near the context ceiling: {', '.join(risky)}")

    if failed:
        print("\nfailed: " + ", ".join(r["id"] for r in failed))
    if errored:
        print("errored: " + ", ".join(r["id"] for r in errored))

    return {
        "label": label,
        "model": MODEL,
        "num_ctx": NUM_CTX,
        "cases": len(records),
        "passed": len(passed),
        "failed": [r["id"] for r in failed],
        "errored": [r["id"] for r in errored],
        "by_category": by_cat,
        "elapsed_s": round(elapsed, 1),
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser(description="Run the chat golden set eval.")
    ap.add_argument("--only", help="run a single case by id")
    ap.add_argument("--category", help="run only cases in this category")
    ap.add_argument("--verbose", action="store_true", help="print full answers")
    ap.add_argument("--label", default="3b, not baseline", help="label recorded with the run")
    ap.add_argument("--out", help="output path, defaults to eval_runs/<epoch>_<model>.json")
    args = ap.parse_args()

    # A configured endpoint means a real POST to the contact route, and the send cases feed the
    # model real looking recruiter details. Refuse rather than risk mail in Firza's inbox.
    if os.environ.get("CONTACT_ENDPOINT", "").strip():
        print("ERROR: CONTACT_ENDPOINT is set. Unset it before running the eval, see D29.")
        sys.exit(1)

    cases = json.loads(GOLDEN_PATH.read_text())["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if not cases:
        print("No cases matched the filter.")
        sys.exit(1)

    t_start = time.time()
    records = []
    for case in cases:
        rec = run_case(case)
        records.append(rec)
        label = f"[{rec['category']}] {rec['id']}"
        if rec.get("error"):
            print(f"ERROR {label}: {rec['error']}")
        elif rec["failures"]:
            print(f"FAIL  {label}  {rec['wall_s']}s")
            for desc in rec["failures"]:
                print(f"      {desc}")
            print(f"      answer: {rec['answer'] if args.verbose else rec['answer'][:300].replace(chr(10), ' ')}")
        else:
            print(f"pass  {label}  {rec['wall_s']}s")
            if args.verbose:
                print(f"      answer: {rec['answer']}")

    run = summarise(records, args.label, time.time() - t_start)

    OUT_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"{int(t_start)}_{MODEL.replace(':', '-')}.json"
    out.write_text(json.dumps(run, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")

    sys.exit(1 if (run["failed"] or run["errored"]) else 0)


if __name__ == "__main__":
    main()
