#!/usr/bin/env python3
"""Cheap end-to-end smoke path; no stochtree fits are performed.

Default mode exercises one raw/current/reference correction task plus one
information task per family.  ``--family correction_completion`` runs one smoke
replication of every completion design and both estimators (including the
pre-trend designs), then aggregates the checkpoints through the same
downloaded-archive path used for Colab results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from extra_theory_experiments.manifest import build_manifest, manifest_frame
from extra_theory_experiments.runner import run_tasks, write_provenance


def _completion_check(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tasks = build_manifest("correction_completion", reps=1, n_shards=1)
    designs = sorted({task.design for task in tasks})
    assert len(designs) == 21, designs
    manifest_path = out / "manifest.csv"
    manifest_frame(tasks).to_csv(manifest_path, index=False)
    result = run_tasks(tasks, out_dir=out / "checkpoints", smoke=True, resume=True)
    summary_path = out / "summary.csv"
    result.to_csv(summary_path, index=False)
    write_provenance(out / "provenance.json", tasks=tasks,
                     repo_root=HERE.parents[1], smoke=True)
    archive_path = out / "correction_completion_smoke.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, arcname="manifest.csv")
        archive.write(summary_path, arcname="summary.csv")
    from aggregate_archives import aggregate_archives
    report, _, metrics = aggregate_archives(
        [archive_path], output_dir=out / "aggregated",
        family="correction_completion", n_shards=1)
    print("completion smoke tasks:", len(tasks), "| output rows:", len(result))
    print("designs:", len(designs), "| estimators:",
          sorted({task.estimator for task in tasks}))
    print(summary_path.read_text(encoding="utf-8").splitlines()[0])
    print("missing shards:", report["missing_shards"],
          "| missing tasks:", len(report["missing_task_bundles"]),
          "| metrics rows:", len(metrics))
    if report["missing_task_bundles"]:
        raise SystemExit("smoke aggregation reports missing task bundles")
    if not (out / "aggregated" / "aggregation_report.json").exists():
        raise SystemExit("smoke aggregation did not write its report")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--family", default=None,
                        choices=[None, "correction_completion"])
    args = parser.parse_args()
    out = (Path(args.out_dir) if args.out_dir
           else HERE / "results" / ("smoke_completion" if args.family
                                    else "smoke"))
    out.mkdir(parents=True, exist_ok=True)
    if args.family == "correction_completion":
        _completion_check(out)
        return
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
    write_provenance(out / "provenance.json", tasks=tasks,
                     repo_root=HERE.parents[1], smoke=True)
    print("smoke tasks:", len(tasks), "| output rows:", len(result))
    print(result[["family", "design", "task_estimator"]].drop_duplicates().to_string(index=False))


if __name__ == "__main__":
    main()
