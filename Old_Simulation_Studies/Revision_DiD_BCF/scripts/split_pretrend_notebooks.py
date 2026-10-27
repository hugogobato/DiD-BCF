"""Split each Pre-trend notebook into fixed replication blocks for Colab.

A Colab session is capped at roughly 8 hours and one diagnostic fit is ~2
minutes, so a 200-replication run does not fit in one sitting.  Replications are
seeded by index, so ``rep_start``/``rep_end`` blocks concatenate into exactly the
undivided run.  This writes one notebook per block, each landing in its own
summary CSV.

The base notebooks are the source of truth: run ``scaffold_suite.py`` first,
then this.  Blocks are regenerated from the base every time, so a change to the
templates propagates -- which means a block notebook that already carries
execution outputs is **skipped**, since overwriting it would destroy the record
of that run.

Run: ``python scripts/split_pretrend_notebooks.py [--block 50] [--reps 200]``
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT_DIR = os.path.join(ROOT, "Pretrend")
_RUN_LINE = re.compile(r"^REP_START, REP_END = .*$", re.M)


def _blocks(reps: int, size: int):
    return [(s, min(s + size, reps)) for s in range(0, reps, size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    bases = [f for f in sorted(glob.glob(os.path.join(PT_DIR, "*.ipynb")))
             if "_reps" not in os.path.basename(f)]
    if not bases:
        raise SystemExit("no base notebooks in Pretrend/ -- run scaffold_suite.py")

    written = skipped = 0
    for base in bases:
        stem = os.path.basename(base)[:-len(".ipynb")]
        src = json.load(open(base))
        for start, end in _blocks(args.reps, args.block):
            out = os.path.join(PT_DIR, f"{stem}_reps{start}-{end}.ipynb")
            if os.path.exists(out):
                nb = json.load(open(out))
                if any(c.get("outputs") for c in nb["cells"]):
                    print(f"  SKIP (has outputs): {os.path.basename(out)}")
                    skipped += 1
                    continue
            nb = json.loads(json.dumps(src))          # deep copy
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                text = "".join(cell["source"])
                if "REP_START, REP_END" not in text:
                    continue
                text = _RUN_LINE.sub(f"REP_START, REP_END = {start}, {end}", text)
                cell["source"] = text.splitlines(keepends=True)
            with open(out, "w") as f:
                json.dump(nb, f, indent=1)
            written += 1

    print(f"\nbase notebooks: {len(bases)}")
    print(f"blocks written: {written}   skipped (already run): {skipped}")


if __name__ == "__main__":
    raise SystemExit(main())
