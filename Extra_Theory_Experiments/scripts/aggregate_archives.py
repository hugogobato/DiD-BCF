#!/usr/bin/env python3
"""Safely aggregate downloaded shard archives and compute paired metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extra_theory_experiments.metrics import summarise_replications


TASK_COLS = ["family", "design", "degree", "N", "rep", "task_estimator"]
ESTIMAND_COLS = TASK_COLS + ["estimand_type", "estimand_id", "method"]


def _normalise_task_columns(frame):
    if "task_estimator" not in frame and "estimator" in frame:
        frame = frame.rename(columns={"estimator": "task_estimator"})
    return frame


def _task_key(frame):
    return frame[TASK_COLS].astype(str).agg("|".join, axis=1)


def _read_zip(path: Path):
    manifests, summaries = [], []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = Path(info.filename)
            # Reject traversal before reading, even though only known basenames
            # are selected.  Archives are downloaded artifacts, not trusted input.
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive member: {info.filename}")
            if name.name not in {"manifest.csv", "summary.csv"}:
                continue
            data = archive.read(info)
            from io import BytesIO
            frame = pd.read_csv(BytesIO(data))
            (manifests if name.name == "manifest.csv" else summaries).append(frame)
    return manifests, summaries


def _input_archives(inputs):
    paths = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.zip")))
        elif p.suffix.lower() == ".zip":
            paths.append(p)
        else:
            raise ValueError(f"input is neither a directory nor zip: {p}")
    return sorted(set(paths))


def aggregate_archives(inputs, *, output_dir="aggregated", family=None, n_shards=48):
    paths = _input_archives(inputs)
    if not paths:
        raise ValueError("no shard zip archives found")
    manifests, summaries = [], []
    for path in paths:
        ms, ss = _read_zip(path)
        manifests.extend(ms); summaries.extend(ss)
    if not manifests:
        raise ValueError("no manifest.csv found in downloaded archives")
    manifest = _normalise_task_columns(pd.concat(manifests, ignore_index=True))
    if family is not None:
        manifest = manifest[manifest["family"] == family].copy()
    required = [x for x in TASK_COLS + ["shard", "n_shards", "config_hash"] if x not in manifest]
    if required:
        raise ValueError(f"manifest missing columns: {required}")
    observed_shards = sorted(manifest["shard"].astype(int).unique().tolist())
    expected_shards = list(range(int(n_shards)))
    missing_shards = sorted(set(expected_shards) - set(observed_shards))
    duplicate_tasks = int(manifest.duplicated(TASK_COLS, keep=False).sum())
    manifest = manifest.drop_duplicates(TASK_COLS, keep="first")
    config_hashes = sorted(manifest["config_hash"].dropna().astype(str).unique())
    inconsistent_n = sorted(manifest["n_shards"].astype(int).unique().tolist())
    if len(config_hashes) > 1:
        raise ValueError(f"config hash mismatch across archives: {config_hashes}")
    if any(int(x) != int(n_shards) for x in inconsistent_n):
        raise ValueError(f"unexpected n_shards values {inconsistent_n}; expected {n_shards}")
    summary = (_normalise_task_columns(pd.concat(summaries, ignore_index=True))
               if summaries else pd.DataFrame())
    if family is not None and not summary.empty and "family" in summary:
        summary = summary[summary["family"] == family].copy()
    duplicate_rows = 0
    if not summary.empty:
        missing = [x for x in ESTIMAND_COLS if x not in summary]
        if missing:
            # Unavailable rows have no estimand schema and are retained separately.
            available = summary.get("status", pd.Series(index=summary.index)).ne("unavailable")
            missing = [x for x in missing if available.any()]
        if not missing:
            duplicate_rows = int(summary.duplicated(ESTIMAND_COLS, keep=False).sum())
            summary = summary.drop_duplicates(ESTIMAND_COLS, keep="first")
    manifest_keys = set(_task_key(manifest))
    if summary.empty:
        summary_keys = set()
    else:
        summary_keys = set(_task_key(summary[summary.get("status", pd.Series(
            "ok", index=summary.index)).ne("unavailable")]))
    missing_tasks = sorted(manifest_keys - summary_keys)
    unavailable_tasks = set()
    if not summary.empty and "status" in summary:
        unavailable_tasks = set(_task_key(summary[summary["status"] == "unavailable"]))
    missing_tasks = [x for x in missing_tasks if x not in unavailable_tasks]
    metrics_input = summary.copy()
    if "status" in metrics_input:
        metrics_input = metrics_input[metrics_input["status"] != "unavailable"]
    metrics = summarise_replications(metrics_input) if not metrics_input.empty else pd.DataFrame()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "combined_manifest.csv", index=False)
    summary.to_csv(out / "combined_per_replication.csv", index=False)
    try:
        summary.to_parquet(out / "combined_per_replication.parquet", index=False)
        metrics.to_parquet(out / "metrics_mean_median.parquet", index=False)
    except Exception:
        pass
    metrics.to_csv(out / "metrics_mean_median.csv", index=False)
    report = {
        "archives_read": [str(x) for x in paths],
        "family": family,
        "expected_shards": expected_shards,
        "observed_shards": observed_shards,
        "missing_shards": missing_shards,
        "inconsistent_n_shards": inconsistent_n,
        "config_hashes": config_hashes,
        "duplicate_manifest_tasks": duplicate_tasks,
        "duplicate_summary_rows": duplicate_rows,
        "manifest_tasks": len(manifest_keys),
        "summary_task_bundles": len(summary_keys),
        "missing_task_bundles": missing_tasks,
        "unavailable_task_bundles": sorted(unavailable_tasks),
        "oracle_unavailable_not_failed": True,
        "metrics_rows": len(metrics),
    }
    (out / "aggregation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, summary, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="downloaded zip files or directories")
    parser.add_argument("--output-dir", default="aggregated")
    parser.add_argument("--family", default=None)
    parser.add_argument("--n-shards", type=int, default=48)
    args = parser.parse_args()
    report, _, _ = aggregate_archives(args.inputs, output_dir=args.output_dir,
                                       family=args.family, n_shards=args.n_shards)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
