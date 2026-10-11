#!/usr/bin/env python3
"""Estimate production waves, runtime and storage for candidate budgets.

Uses the missing-phase manifest, the measured main-phase durations (scaled by
the per-chain sweep ratio) and conservative kind factors.  Estimates are
scheduling inputs only; the accepted budget is frozen by the E2 gate and the
first returned production waves replace these estimates with measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recovery_common import BUDGETS, INVENTORY, MANIFESTS, RECOVERY, \
    sweeps_per_chain, write_json_atomic
from recovery_plan import ATOMIC_KINDS, make_units


def raw_bytes(job: dict, budget: dict) -> float:
    """Uncompressed per-unit draw estimate (storage planning only)."""
    n_obs = 1600 if not job["kind"].startswith("application") else 2500
    n_mcmc = budget["num_mcmc"]
    if job["kind"] in ATOMIC_KINDS:
        return n_obs * n_mcmc * budget["num_chains"] * 8.0
    return (n_obs * n_mcmc * 8.0 if job["kind"] == "application"
            else 20 * n_mcmc * 8.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=RECOVERY / "production")
    parser.add_argument("--target-hours", type=float, default=7.0)
    parser.add_argument("--wave-size", type=int, default=48)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    missing = json.loads((MANIFESTS / "manifest_missing_phases.json").read_text())
    rows = []
    for key in ("B0_historical", "B1_warm2500_ret1000", "B2_warm5000_ret1000",
                "B3_warm2500_ret2000", "B4_warm10000_ret2000"):
        manifest = {**missing, "budgets": {key: BUDGETS[key]},
                    "budget_keys": [key], "trace": False}
        units = make_units(manifest)
        hours = sum(unit["est_seconds"] for unit in units) / 3600
        notebooks = max(1, int(-(-hours // args.target_hours)))
        waves = int(-(-notebooks // args.wave_size))
        storage_gib = sum(raw_bytes(unit["job"], unit["budget"])
                          for unit in units) / 2 ** 30
        rows.append({"budget_key": key, "units": len(units),
                     "per_chain_sweeps": sweeps_per_chain(BUDGETS[key]),
                     "estimated_sampling_hours": round(hours, 1),
                     "notebooks_at_target": notebooks, "waves_of_48": waves,
                     "raw_draw_storage_gib": round(storage_gib, 1),
                     "compressed_storage_gib_estimate": round(storage_gib * 0.55, 1)})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "production_estimates.csv", index=False)

    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None))
    table = "\n".join([header, divider, body])

    lines = [
        "# Production compute and storage schedule (estimates before the E2 gate)",
        "",
        "No production notebook is generated until the pilot gate decision is "
        "recorded. These are conservative scheduling estimates from measured "
        "main-phase durations scaled by the per-chain sweep ratio and kind "
        "factors; they are not promises and not evidence.",
        "",
        "Per-chain sweeps: B0=2,550, B1=7,550, B2=10,050, B3=12,550, B4=20,050. "
        "The archived main phase ran 106/107 fits per shard in 4.45-7.56 hours.",
        "",
        table,
        "",
        "Notes",
        "",
        "1. Notebook count assumes no shard exceeds the target sampling hours "
        f"({args.target_hours:g} h) and one worker per runtime; it does not yet "
        "use measured production chain times.",
        "2. Raw storage is the uncompressed full-surface draws for diagnostic "
        "and application units; compressed estimates use a 0.55 factor. "
        "Diagnostic jobs dominate: storing full tau arrays is required to "
        "reconstruct subgroup and joint statistics and must not be thinned "
        "away.",
        "3. Waves are groups of at most 48 independent notebooks, matching the "
        "user's 16 accounts and 48-shard envelope. Run the application batch "
        "first (application and application_sensitivity phases are scheduled "
        "in the earliest waves by the production generator).",
        "4. A single structured chain is small; pretrend diagnostic jobs are "
        "atomic because stochtree.BCFModel does not expose per-chain state. "
        "Any unit whose conservative estimate exceeds the session window is "
        "reported and not started.",
        "5. After each wave, rerun `recovery_collect.py`; the collector writes "
        "`observed_seconds.csv`, and regenerating assignments then uses the "
        "measured times.",
    ]
    (args.out / "SCHEDULING.md").write_text("\n".join(lines) + "\n")
    write_json_atomic(args.out / "production_estimates.json",
                      frame.to_dict(orient="records"))
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
