#!/usr/bin/env python3
"""Aggregate the R-benchmark summaries into metric tables *the same way DiD-BCF
does* -- by feeding them through the engine's own
``did_bcf_revision.metrics.{compute_metrics, surface_metrics}``.

Each R benchmark script writes
``R_code/<scenario>_datasets/summaries_<method>_<scenario>_lin_<d>.csv`` in the
exact schema of ``DiD_BCF/summaries_<scenario>_lin_<d>.csv``.  This script, for
every (scenario, linearity_degree):

  * stacks all available benchmark summaries with the DiD-BCF summaries
    (``plain``/``corrected``) when present, and
  * runs the identical metric functions, writing
        Benchmark_Results/metrics_<scenario>_lin_<d>.csv   (decomposed scalar metrics)
        Benchmark_Results/surface_<scenario>_lin_<d>.csv   (CATT-surface RMSE/MAE/MAPE)
    with one row per (estimand cell|surface) x method -- so DiD-BCF and every
    benchmark sit side by side on identical definitions.

The benchmark methods carry no individual CATT, so their CATT-surface rows use
their scalar estimate (GATT(g,t)/ES(k)/ATT) broadcast to each treated obs.

Run:  python3 benchmark_metrics.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from did_bcf_revision.metrics import compute_metrics, surface_metrics  # noqa: E402

RCODE = os.path.join(HERE, "R_code")
BCF_DIR = os.path.join(HERE, "DiD_BCF")
OUT_DIR = os.path.join(HERE, "Benchmark_Results")

KNOWN_METHODS = ["did_dr", "doubleml", "did2s", "synthdid", "wang"]


def _parse(fname: str):
    """summaries_<method>_<scenario>_lin_<d>.csv -> (method, scenario, lin)."""
    body = os.path.basename(fname)[len("summaries_"):-len(".csv")]
    m = re.match(r"(.+)_lin_(\d+)$", body)
    if not m:
        return None
    rest, lin = m.group(1), int(m.group(2))
    for meth in KNOWN_METHODS:
        if rest.startswith(meth + "_"):
            return meth, rest[len(meth) + 1:], lin
    return None


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    # recursive: also picks up the B2 sample-size sweep written under
    # <scenario>_datasets/N=<N>/ (one summary per N; the N column distinguishes them).
    bench = glob.glob(os.path.join(RCODE, "*_datasets", "**", "summaries_*_lin_*.csv"),
                      recursive=True)
    groups: dict[tuple[str, int], list[str]] = {}
    for f in bench:
        p = _parse(f)
        if p is None:
            print("  ?? skip unparseable:", f)
            continue
        _, scenario, lin = p
        groups.setdefault((scenario, lin), []).append(f)

    if not groups:
        print("No benchmark summaries found under", RCODE)
        return

    for (scenario, lin) in sorted(groups):
        frames = []
        bcf = os.path.join(BCF_DIR, f"summaries_{scenario}_lin_{lin}.csv")
        if os.path.isfile(bcf):
            frames.append(pd.read_csv(bcf))
        for f in sorted(groups[(scenario, lin)]):
            frames.append(pd.read_csv(f))
        summ = pd.concat(frames, ignore_index=True).drop_duplicates()

        met = compute_metrics(summ)
        surf = surface_metrics(summ)
        mpath = os.path.join(OUT_DIR, f"metrics_{scenario}_lin_{lin}.csv")
        spath = os.path.join(OUT_DIR, f"surface_{scenario}_lin_{lin}.csv")
        met.to_csv(mpath, index=False)
        surf.to_csv(spath, index=False)
        methods = sorted(summ["method"].unique())
        print(f"[{scenario} lin{lin}] methods={methods} "
              f"-> {os.path.basename(mpath)} ({len(met)} rows), "
              f"{os.path.basename(spath)} ({len(surf)} rows)")


if __name__ == "__main__":
    main()
