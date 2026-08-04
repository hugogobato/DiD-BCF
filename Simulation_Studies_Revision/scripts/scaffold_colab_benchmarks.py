#!/usr/bin/env python3
"""Generate the Colab benchmark notebooks (DoubleML, grf-DiD) for new scenarios.

``DoubleML_Colab/`` and ``CFFE_Wang_Colab/`` hold one self-contained notebook per
scenario.  They were written by hand, and comparing any two of them shows they
differ in exactly two places: the scenario name and the scenario's ``note``
string from ``config.py``.  So rather than hand-editing a notebook per new
scenario -- which is how the R scripts drifted out of sync with
``scaffold_suite.py`` -- this retargets a checked-in **reference notebook** for
the matching DGP, substituting those two strings and the replication count.

The reference notebook is never rewritten, so running this cannot damage the
source it copies from.

Why these two estimators live on Colab: DoubleML fits random-forest nuisances and
takes roughly an hour per scenario locally (measured on ``D_ramp_nt40``: 200
replications x 3 linearity degrees, still running at 67 minutes), and grf-DiD
needs an R toolchain the notebooks install themselves.  The three fast R
estimators (Callaway--Sant'Anna, Gardner, synthdid) stay local via
``scripts/run_r_benchmarks.py --scripts did_dr_new did2s synthdid``.

Examples
--------
    python scripts/scaffold_colab_benchmarks.py --scenarios D_ramp_nt40 D_ramp_nt05
    python scripts/scaffold_colab_benchmarks.py --workstream D      # every D scenario
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from did_bcf_revision import config as cfg

# One reference notebook per (folder, dgp).  These are checked in and are the
# source of truth for their folder's layout.
FAMILIES = {
    "DoubleML_Colab": {
        "prefix": "DoubleML_",
        "reference": {"canonical": "B1_baseline", "staggered": "D_staggered"},
    },
    "CFFE_Wang_Colab": {
        "prefix": "CFFE_Wang_",
        "reference": {"canonical": "B1_baseline", "staggered": "D_staggered"},
    },
}


def _reps_line(src: str, reps: int) -> str:
    """Rewrite the ``REPS = <n>`` assignment, preserving the column alignment."""
    out = []
    for line in src.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("REPS") and "=" in line:
            head, _, tail = line.partition("=")
            comment = tail.split("#", 1)
            value = f" {reps}"
            if len(comment) == 2:
                # Keep the trailing comment roughly where it was.
                pad = max(1, len(comment[0]) - len(value))
                line = f"{head}={value}{' ' * pad}#{comment[1]}"
            else:
                line = f"{head}={value}"
        out.append(line)
    return "\n".join(out)


def retarget(nb: dict, ref_name: str, ref_note: str,
             scen: str, note: str, reps: int) -> dict:
    nb = copy.deepcopy(nb)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if ref_note and ref_note in src:
            src = src.replace(ref_note, note)
        src = src.replace(ref_name, scen)
        if cell["cell_type"] == "code" and "REPS" in src:
            src = _reps_line(src, reps)
        cell["source"] = src
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="scenario names (default: every non-PT scenario that "
                         "has no notebook yet)")
    ap.add_argument("--workstream", default=None,
                    help="generate for a whole workstream, e.g. D")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--force", action="store_true",
                    help="overwrite notebooks that already exist")
    args = ap.parse_args()

    exps = {e.name: e for e in cfg.all_experiments() if e.workstream != "PT"}
    if args.scenarios:
        unknown = set(args.scenarios) - set(exps)
        if unknown:
            ap.error(f"unknown scenario(s): {sorted(unknown)}")
        names = list(args.scenarios)
    elif args.workstream:
        names = [n for n, e in exps.items() if e.workstream == args.workstream]
    else:
        names = list(exps)

    written = skipped = 0
    for family in args.families:
        spec = FAMILIES[family]
        folder = os.path.join(ROOT, family)
        os.makedirs(folder, exist_ok=True)
        cache: dict = {}
        for name in names:
            e = exps[name]
            ref_name = spec["reference"][e.dgp]
            if name == ref_name:
                continue
            out_path = os.path.join(folder, f"{spec['prefix']}{name}.ipynb")
            if os.path.exists(out_path) and not args.force:
                skipped += 1
                continue
            if ref_name not in cache:
                ref_path = os.path.join(folder, f"{spec['prefix']}{ref_name}.ipynb")
                if not os.path.exists(ref_path):
                    sys.exit(f"Missing reference notebook {ref_path}")
                with open(ref_path) as fh:
                    cache[ref_name] = json.load(fh)
            nb = retarget(cache[ref_name], ref_name, exps[ref_name].note,
                          name, e.note, e.reps)
            with open(out_path, "w") as fh:
                json.dump(nb, fh, indent=1)
            print(f"  -> {os.path.relpath(out_path, ROOT)}  (reps={e.reps})")
            written += 1

    print(f"written: {written}, skipped (already exist): {skipped}")
    if skipped and not args.force:
        print("pass --force to regenerate existing notebooks")


if __name__ == "__main__":
    main()
