#!/usr/bin/env python3
"""Retarget the B2 sample-size-sweep Colab notebooks to a different N.

``DoubleML_Colab/`` and ``CFFE_Wang_Colab/`` hold **one notebook per (sweep
scenario, N)** -- DoubleML's random-forest nuisances and grf's forests are slow
enough at the top of the sweep that a single notebook covering every N times out
on Colab.  Those notebooks are identical apart from the sample size, so a new N
should never be produced by hand-editing a copy (that is how the N-tag in the
output filenames drifted between the two families in the first place).

This is the N-axis counterpart of ``scaffold_colab_benchmarks.py``, which
retargets the *scenario* axis.  It reads a checked-in reference notebook, swaps
every N-dependent string, and writes ``<prefix><scenario>_N<N>.ipynb``.

Two things are normalised while retargeting, so generated notebooks agree with
the result files already on disk:

* the sample-size hint comment is rewritten from the reference's stale
  ``(200/400/800/1600)`` to the live ``config.N_SWEEP``;
* the grf zip is named ``Wang_<scenario>_results_N<N>.zip``.  The reference
  notebook writes an *untagged* ``Wang_<scenario>_results.zip``, which collides
  across N -- the archives in ``CFFE_Wang_Colab/`` carry the N tag because they
  were renamed after the fact.

``CFFE_Wang_*`` notebooks embed ``wang_grf.R`` via ``%%writefile`` so they stay
self-contained on Colab; the embedded copy is refreshed from
``CFFE_Wang_Colab/wang_grf.R`` here, which keeps the two from diverging.

Examples
--------
    python scripts/scaffold_sweep_N_notebooks.py --N 50 100
    python scripts/scaffold_sweep_N_notebooks.py --N 50 --families DoubleML_Colab
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from did_bcf_revision import config as cfg

# Reference notebook per family, expressed as the N it was written for.  N=400
# is the reference for both because it is the only N whose notebooks exist in
# both families for both sweep scenarios.
FAMILIES = {
    "DoubleML_Colab": {"prefix": "DoubleML_", "ref_N": 400},
    "CFFE_Wang_Colab": {"prefix": "CFFE_Wang_", "ref_N": 400},
}
SCENARIOS = ("B2_sweep", "B2_sweep_serial")

_HINT_RE = re.compile(r"\(\s*\d+(?:\s*/\s*\d+)+\s*\)")


def _retarget(src: str, ref_N: int, N: int, hint: str) -> str:
    """Swap every N-dependent token in one cell's source."""
    # The bare reference N appears as `N=400`, `N = 400`, `_N400_`, `N400`.
    src = src.replace(f"N={ref_N}", f"N={N}")
    src = src.replace(f"_N{ref_N}", f"_N{N}")
    src = re.sub(rf"(\bN\s*=\s*){ref_N}\b", rf"\g<1>{N}", src)
    # `N          = 400        # ... (200/400/800/1600)` -> the live sweep.
    src = _HINT_RE.sub(hint, src)
    # The reference text says "the four N CSVs" / "the four sweeps"; the sweep
    # length is not four any more.
    src = re.sub(r"\b(the )(four|five|six|seven)( N CSVs| sweeps)",
                 rf"\g<1>{_count(len(cfg.N_SWEEP))}\g<3>", src)
    return src


_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def _count(n: int) -> str:
    return _WORDS.get(n, str(n))


def _fix_wang(nb: dict, wang_src: str) -> None:
    """Refresh the embedded wang_grf.R and give the grf zip an N tag."""
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if src.startswith("%%writefile wang_grf.R"):
            cell["source"] = "%%writefile wang_grf.R\n" + wang_src
            continue
        if 'zipname = f"Wang_{SCENARIO}_results.zip"' in src:
            cell["source"] = src.replace(
                'zipname = f"Wang_{SCENARIO}_results.zip"',
                'zipname = f"Wang_{SCENARIO}_results_N{N}.zip"')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", required=True,
                    help="sample size(s) to generate notebooks for")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                    choices=list(SCENARIOS))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--force", action="store_true",
                    help="overwrite notebooks that already exist")
    args = ap.parse_args()

    hint = "(" + "/".join(str(n) for n in cfg.N_SWEEP) + ")"
    wang_src = open(os.path.join(ROOT, "CFFE_Wang_Colab", "wang_grf.R")).read()

    written = skipped = 0
    for family in args.families:
        spec = FAMILIES[family]
        folder = os.path.join(ROOT, family)
        for scen in args.scenarios:
            ref_path = os.path.join(
                folder, f"{spec['prefix']}{scen}_N{spec['ref_N']}.ipynb")
            if not os.path.exists(ref_path):
                sys.exit(f"Missing reference notebook {ref_path}")
            with open(ref_path) as fh:
                ref = json.load(fh)
            for N in args.N:
                if N == spec["ref_N"]:
                    continue
                out_path = os.path.join(folder, f"{spec['prefix']}{scen}_N{N}.ipynb")
                if os.path.exists(out_path) and not args.force:
                    skipped += 1
                    continue
                nb = copy.deepcopy(ref)
                for cell in nb["cells"]:
                    cell["source"] = _retarget("".join(cell["source"]),
                                               spec["ref_N"], N, hint)
                if family == "CFFE_Wang_Colab":
                    _fix_wang(nb, wang_src)
                with open(out_path, "w") as fh:
                    json.dump(nb, fh, indent=1)
                print(f"  -> {os.path.relpath(out_path, ROOT)}")
                written += 1

    print(f"written: {written}, skipped (already exist): {skipped}")
    if skipped and not args.force:
        print("pass --force to regenerate existing notebooks")


if __name__ == "__main__":
    main()
