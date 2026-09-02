"""Compare two eval runs: score, which cases moved, and how many answers are byte identical.

Written after the third run of the same config scored 28, 23, and 24 of 41. A score on its own
cannot tell a real change from a different sample, so every comparison from here reports the
failure set diff and the identical-answer count next to the score.

The identical-answer count is what proves a fixed seed works. Two runs at the same seed and the
same config should be identical on every case. Two arms that differ only in MAX_TOOL_ROUNDS should
be identical on every case that never calls a tool.

    python scripts/compare_runs.py eval_runs/<a>.json eval_runs/<b>.json
"""
import json
import sys
from pathlib import Path


def load(path):
    d = json.loads(Path(path).read_text())
    return d, {r["id"]: r for r in d["records"]}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    a, ra = load(sys.argv[1])
    b, rb = load(sys.argv[2])

    print(f"A  {a['passed']}/{a['cases']}  {a['label']}")
    print(f"B  {b['passed']}/{b['cases']}  {b['label']}")
    print(f"   model A {a.get('model')}  model B {b.get('model')}")

    fa, fb = set(a["failed"]), set(b["failed"])
    print(f"\nfailed in both: {len(fa & fb)}")
    if fb - fa:
        print(f"B broke {len(fb - fa)}: {', '.join(sorted(fb - fa))}")
    if fa - fb:
        print(f"B fixed {len(fa - fb)}: {', '.join(sorted(fa - fb))}")

    shared = sorted(set(ra) & set(rb))
    same = [i for i in shared if ra[i].get("answer") == rb[i].get("answer")]
    print(f"\nbyte identical answers: {len(same)}/{len(shared)}")
    diff = [i for i in shared if i not in set(same)]
    if diff:
        print("differing: " + ", ".join(diff))

    # A pass count that moved while every answer stayed identical would mean the grader is unstable,
    # which is a different bug from the model being unstable. Worth knowing which one you have.
    if len(same) == len(shared) and a["passed"] != b["passed"]:
        print("\nWARNING: identical answers but different scores. The grader moved, not the model.")


if __name__ == "__main__":
    main()
