#!/usr/bin/env python3
"""Safely aggregate downloaded shard archives and compute paired metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
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
            # ``design="null"`` is a legitimate design name; pandas' default NA
            # handling would silently turn it into a missing value.
            frame = pd.read_csv(BytesIO(data), keep_default_na=False, na_values=[])
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


def aggregate_archives(inputs, *, output_dir="aggregated", family=None,
                       n_shards=48, n_waves=None):
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
    wave_columns = {"wave_id", "n_waves"}
    present_wave_columns = wave_columns.intersection(manifest.columns)
    if present_wave_columns and present_wave_columns != wave_columns:
        raise ValueError("manifest must contain both wave_id and n_waves")
    if present_wave_columns == wave_columns:
        try:
            manifest["wave_id"] = manifest["wave_id"].astype(int)
            manifest["n_waves"] = manifest["n_waves"].astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest wave fields must be integer-valued") from exc
        declared_waves = sorted(manifest["n_waves"].unique().tolist())
        if any(x < 1 for x in declared_waves):
            raise ValueError(f"manifest n_waves values must be positive: {declared_waves}")
        if any(x < 0 or x >= n for x, n in
               zip(manifest["wave_id"], manifest["n_waves"])):
            raise ValueError("manifest wave_id must satisfy 0 <= wave_id < n_waves")
        if len(declared_waves) > 1:
            raise ValueError(f"inconsistent n_waves values: {declared_waves}")
        manifest_n_waves = int(declared_waves[0]) if declared_waves else 1
        if n_waves is not None and int(n_waves) != manifest_n_waves:
            raise ValueError(
                f"manifest n_waves={manifest_n_waves} does not match requested {n_waves}")
        expected_n_waves = manifest_n_waves
        wave_fields_present = True
    else:
        # Archives from the original one-wave runner predate wave columns.
        # Treat them as wave 0 of 1 unless the caller asks to audit a larger
        # multi-wave collection, in which case missing waves are reported.
        expected_n_waves = int(n_waves) if n_waves is not None else 1
        if expected_n_waves < 1:
            raise ValueError("n_waves must be positive")
        wave_fields_present = False
    observed_waves = ([int(x) for x in sorted(manifest["wave_id"].unique())]
                      if wave_fields_present and not manifest.empty else
                      ([0] if not manifest.empty else []))
    expected_waves = list(range(expected_n_waves))
    missing_waves = sorted(set(expected_waves) - set(observed_waves))
    observed_wave_shards = sorted({
        (int(row.wave_id), int(row.shard))
        for row in manifest[["wave_id", "shard"]].itertuples(index=False)
    }) if wave_fields_present and not manifest.empty else sorted(
        {(0, int(x)) for x in observed_shards})
    expected_wave_shards = [
        {"wave_id": wave, "shard": shard}
        for wave in expected_waves for shard in expected_shards
    ]
    missing_wave_shards = [
        item for item in expected_wave_shards
        if (item["wave_id"], item["shard"]) not in set(observed_wave_shards)
    ]
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
    unavailable_rows_without_task_key = 0
    if not summary.empty:
        status = (summary["status"].astype(str)
                  if "status" in summary else
                  pd.Series("ok", index=summary.index, dtype="object"))
        available_mask = status.ne("unavailable")
        available = summary.loc[available_mask].copy()
        unavailable = summary.loc[~available_mask].copy()
        # Validate and deduplicate only actual estimates.  Some shard runners
        # emit a deliberately compact unavailable-only row with just status and
        # reason, so requiring the estimand schema on that row would be wrong.
        if not available.empty:
            missing = [x for x in ESTIMAND_COLS if x not in available]
            if missing:
                raise ValueError(f"available summary missing columns: {missing}")
            duplicate_rows = int(available.duplicated(ESTIMAND_COLS, keep=False).sum())
            available = available.drop_duplicates(ESTIMAND_COLS, keep="first")
        if (not unavailable.empty and
                not set(TASK_COLS).issubset(unavailable.columns)):
            unavailable_rows_without_task_key = int(len(unavailable))
        summary = pd.concat([available, unavailable], ignore_index=True)
    manifest_keys = set(_task_key(manifest))
    if summary.empty:
        summary_keys = set()
    else:
        status = (summary["status"].astype(str)
                  if "status" in summary else
                  pd.Series("ok", index=summary.index, dtype="object"))
        available = summary.loc[status.ne("unavailable")]
        summary_keys = (set(_task_key(available))
                        if not available.empty else set())
    missing_tasks = sorted(manifest_keys - summary_keys)
    unavailable_tasks = set()
    if not summary.empty and "status" in summary:
        unavailable = summary.loc[summary["status"].astype(str) == "unavailable"]
        if not unavailable.empty and set(TASK_COLS).issubset(unavailable.columns):
            unavailable_tasks = set(_task_key(unavailable))
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
        "wave_fields_present": wave_fields_present,
        "expected_n_waves": expected_n_waves,
        "expected_waves": expected_waves,
        "observed_waves": observed_waves,
        "missing_waves": missing_waves,
        "observed_wave_shards": [
            {"wave_id": wave, "shard": shard}
            for wave, shard in observed_wave_shards
        ],
        "missing_wave_shards": missing_wave_shards,
        "backward_compatible_one_wave": not wave_fields_present and expected_n_waves == 1,
        "inconsistent_n_shards": inconsistent_n,
        "config_hashes": config_hashes,
        "duplicate_manifest_tasks": duplicate_tasks,
        "duplicate_summary_rows": duplicate_rows,
        "manifest_tasks": len(manifest_keys),
        "summary_task_bundles": len(summary_keys),
        "missing_task_bundles": missing_tasks,
        "unavailable_task_bundles": sorted(unavailable_tasks),
        "unavailable_rows_without_task_key": unavailable_rows_without_task_key,
        "oracle_unavailable_not_failed": True,
        "metrics_rows": len(metrics),
    }
    (out / "aggregation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, summary, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="downloaded zip files or directories")
    parser.add_argument("--output-dir", default=None,
                        help="default: 'aggregated', or the family results folder "
                             "for correction_completion")
    parser.add_argument("--family", default=None,
                        choices=["correction_audit", "information_ablation",
                                 "correction_completion"])
    parser.add_argument("--n-shards", type=int, default=48)
    parser.add_argument("--n-waves", type=int, default=None,
                        help="expected waves; inferred from wave-aware manifests")
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (str(HERE / "results" / "aggregated_completion")
                      if args.family == "correction_completion" else "aggregated")
    report, _, _ = aggregate_archives(args.inputs, output_dir=output_dir,
                                       family=args.family, n_shards=args.n_shards,
                                       n_waves=args.n_waves)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
