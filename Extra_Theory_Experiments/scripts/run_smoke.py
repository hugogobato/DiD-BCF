#!/usr/bin/env python3
"""Cheap end-to-end smoke path; no stochtree fits are performed."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from extra_theory_experiments.manifest import build_manifest, manifest_frame
from extra_theory_experiments.runner import run_tasks, write_provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(HERE / "results" / "smoke"))
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # One raw/current/reference correction task plus one information task per
    # family keeps this path under seconds while exercising every architecture.
    a = build_manifest("correction_audit", reps=args.reps, n_shards=1)
    b = build_manifest("information_ablation", reps=args.reps, n_shards=1)
    wanted = ("raw_structured", "current_hybrid_logit",
              "reference_fold_convolution_logit")
    tasks = [x for x in a if x.estimator in wanted][:3]
    tasks += [x for x in b if x.estimator in
              ("full_panel_raw", "reduced_cell", "pooled_full_panel")][:3]
    manifest_frame(tasks).to_csv(out / "smoke_manifest.csv", index=False)
    result = run_tasks(tasks, out_dir=out / "checkpoints", smoke=True, resume=True)
    result.to_csv(out / "smoke_summary.csv", index=False)
    write_provenance(out / "provenance.json", tasks=tasks, repo_root=HERE.parent,
                     smoke=True)
    print("smoke tasks:", len(tasks), "| output rows:", len(result))
    print(result[["family", "design", "task_estimator"]].drop_duplicates().to_string(index=False))


if __name__ == "__main__":
    main()
