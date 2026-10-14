"""Collect compact non-main campaign results and report every missing fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import tempfile
import zipfile

from common import PACKAGE, atomic, assignment, bootstrap, read, sha, verify_fit, verify_package


# These counts are part of the non-main campaign definition. Keeping them here
# makes a manifest with an accidentally omitted or renamed phase fail closed.
PHASE_EXPECTED = {
    "diagnostic": 4000,
    "pt_impact": 4000,
    "sensitivity": 1200,
    "application": 9,
    "application_sensitivity": 18,
}
REUSED_EXPECTED = 8
APPLICATION_PHASES = {"application", "application_sensitivity"}
SIMULATION_PHASES = set(PHASE_EXPECTED) - APPLICATION_PHASES
MAX_MEMBER_BYTES = 16 * 2**20


def _read_csv(path):
    """Read compact CSVs without allowing pandas to coerce hexadecimal IDs."""
    import pandas as pd

    path = Path(path)
    try:
        frame = pd.read_csv(path, dtype={"job_id": str}, float_precision="round_trip")
    except Exception as exc:
        raise ValueError(f"Cannot read compact CSV {path}: {exc}") from exc
    if "job_id" in frame and frame["job_id"].isna().any():
        raise ValueError(f"Missing job_id in compact CSV {path}")
    return frame


def _same_value(value, expected):
    import pandas as pd

    if expected is None:
        return bool(pd.isna(value))
    if pd.isna(expected):
        return bool(pd.isna(value))
    if isinstance(expected, bool):
        return value is expected or value == expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(value) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(value) == str(expected)


def _check_context(frame, expected, fields, label, fill_missing=False):
    import pandas as pd

    for field in fields:
        if field not in frame.columns:
            continue
        value = expected.get(field)
        # Some frozen rows include derived empirical labels (for example,
        # dgp="empirical" and N=500) that are intentionally absent from the
        # manifest job. Validate only authoritative, non-null job fields.
        if value is None:
            continue
        present = frame[field].notna()
        if not all(_same_value(item, value) for item in frame.loc[present, field].array):
            raise ValueError(f"{label} has inconsistent {field}")
        if fill_missing and not bool(present.all()):
            frame[field] = frame[field].where(present, value)


def _validate_manifest(manifest):
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Unknown manifest schema: jobs must be a list")
    job_ids = [job.get("id") for job in jobs]
    if any(not isinstance(job_id, str) or not job_id for job_id in job_ids):
        raise ValueError("Unknown manifest schema: every job id must be a nonempty string")
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Manifest contains duplicate job ids")
    counts = {phase: 0 for phase in PHASE_EXPECTED}
    for job in jobs:
        phase = job.get("phase")
        if phase not in counts:
            raise ValueError(f"Unknown non-main phase in manifest: {phase!r}")
        counts[phase] += 1
    if counts != PHASE_EXPECTED:
        raise ValueError(f"Manifest phase counts differ from the frozen campaign: {counts}")
    reused_ids = manifest.get("reused_job_ids")
    if (not isinstance(reused_ids, list)
            or any(not isinstance(job_id, str) for job_id in reused_ids)
            or len(reused_ids) != len(set(reused_ids))
            or len(reused_ids) != REUSED_EXPECTED
            or not set(reused_ids).issubset(set(job_ids))):
        raise ValueError("Unknown reused-fit registry or reused job IDs in manifest")
    return jobs, set(job_ids), set(reused_ids)


def _extract_zip(source, root):
    """Extract only small CSV/JSON metrics members from a worker checkpoint."""
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid worker ZIP {source}: {exc}") from exc
    with archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP entries in {source}")
        for entry in entries:
            name = entry.filename
            parts = Path(name).parts
            target = (root / name).resolve()
            mode = (entry.external_attr >> 16) & 0o170000
            if ("\\" in name or not target.is_relative_to(root)
                    or ".." in parts or name.startswith("/")
                    or mode == stat.S_IFLNK):
                raise ValueError(f"Unsafe ZIP member in {source}: {name}")
            if entry.is_dir():
                continue
            if (not (name.startswith("fits/") or name.startswith("bundle/"))
                    or target.suffix not in {".csv", ".json"}):
                raise ValueError(f"Unexpected worker ZIP member: {name}")
            if entry.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"Oversized compact ZIP member: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(entry) as src, target.open("wb") as dst:
                    dst.write(src.read(MAX_MEMBER_BYTES + 1))
            except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
                raise ValueError(f"Cannot extract worker ZIP member {name}: {exc}") from exc
            if target.stat().st_size != entry.file_size:
                raise ValueError(f"Truncated or oversized ZIP member: {name}")


def _fit_folders(root):
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"Expected fits directory, found {root}")
    folders = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            raise ValueError(f"Unexpected file under fits/: {path}")
        files = list(path.iterdir())
        if not (path / "complete.json").is_file():
            # A worker can be interrupted after writing compact CSVs but
            # before publishing complete.json. Treat that fit as missing.
            continue
        folders.append(path)
    return folders


def _check_fit_files(folder, record):
    expected = set(record.get("outputs", {})) | {"complete.json"}
    actual = set()
    for path in folder.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Unexpected nested or linked fit artifact: {path}")
        actual.add(path.name)
    if actual != expected:
        extra, missing = sorted(actual - expected), sorted(expected - actual)
        raise ValueError(f"Fit artifact list mismatch at {folder}; extra={extra}, missing={missing}")


def _validate_per_replication(frame, record, job, folder):
    required = {
        "job_id", "budget_key", "phase", "estimand_type", "estimand_id", "method",
        "sigma2_shape", "sigma2_rate",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Unknown per_replication.csv schema in {folder}: missing {sorted(required - set(frame.columns))}")
    if len(frame) != record.get("per_replication_rows") or frame.empty:
        raise ValueError(f"Per-fit CSV row count mismatch in {folder}")
    if set(frame["job_id"].astype(str)) != {record["job_id"]}:
        raise ValueError(f"Per-fit CSV job_id mismatch in {folder}")
    if set(frame["budget_key"].astype(str)) != {record["budget_key"]}:
        raise ValueError(f"Per-fit CSV budget_key mismatch in {folder}")
    if set(frame["phase"].astype(str)) != {job["phase"]}:
        raise ValueError(f"Per-fit CSV phase mismatch in {folder}")
    for field in ("sigma2_shape", "sigma2_rate"):
        if (frame[field].isna().any()
                or not all(_same_value(value, job[field]) for value in frame[field].array)):
            raise ValueError(f"Per-fit CSV has missing or inconsistent {field} in {folder}")
    _check_context(
        frame, job,
        ("kind", "dgp", "setting", "N", "linearity_degree", "rep", "spec",
         "sigma2_shape", "sigma2_rate"),
        f"Per-fit CSV {folder}", fill_missing=True,
    )
    key = ["job_id", "budget_key", "estimand_type", "estimand_id", "method"]
    if frame.duplicated(key).any():
        raise ValueError(f"Repeated per-fit metric in {folder}")


def _validate_diagnostics(frame, record, job, folder):
    required = {"job_id", "budget_key"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unknown diagnostics.csv schema in {folder}: missing {sorted(required - set(frame.columns))}")
    if len(frame) != record.get("diagnostic_rows"):
        raise ValueError(f"Diagnostic row count mismatch in {folder}")
    if len(frame):
        if set(frame["job_id"].astype(str)) != {record["job_id"]}:
            raise ValueError(f"Diagnostic job_id mismatch in {folder}")
        if set(frame["budget_key"].astype(str)) != {record["budget_key"]}:
            raise ValueError(f"Diagnostic budget_key mismatch in {folder}")
    context = {**job, "job_id": record["job_id"], "budget_key": record["budget_key"]}
    tagged = ("job_id", "budget_key", "phase", "kind", "dgp", "setting", "N",
              "linearity_degree", "rep", "spec", "sigma2_shape", "sigma2_rate")
    _check_context(frame, context, tagged, f"Diagnostic CSV {folder}", fill_missing=True)
    # Diagnostics are shared output tables. Add job context so pooled rows
    # remain attributable even when the frozen scalar diagnostics omit it.
    for field in tagged:
        if field not in frame.columns:
            frame[field] = context.get(field)


def _validate_application_tables(folder, record, job, app, read_csv):
    from paper_metrics import _APPLICATION_COLUMNS

    if job.get("phase") not in APPLICATION_PHASES or job.get("kind") != "application":
        raise ValueError(f"Application summaries appear on a non-application fit: {folder}")
    row_counts = record.get("application_output_rows")
    if not isinstance(row_counts, dict):
        raise ValueError(f"Application fit lacks output row counts: {folder}")
    context = {**job, "job_id": record["job_id"], "budget_key": record["budget_key"]}
    expected_columns = {
        "job_id", "budget_key", "rep",
    }
    for kind in app:
        path = folder / f"application_{kind}.csv"
        if not path.is_file() or f"application_{kind}.csv" not in record["outputs"]:
            raise ValueError(f"Missing application_{kind}.csv for {folder}")
        frame = read_csv(path)
        expected = set(_APPLICATION_COLUMNS[kind]) | expected_columns
        if set(frame.columns) != expected:
            raise ValueError(f"Unknown application_{kind}.csv schema in {folder}")
        if len(frame) != row_counts.get(kind):
            raise ValueError(f"Application {kind} row count mismatch in {folder}")
        if not len(frame):
            raise ValueError(f"Empty application_{kind}.csv in {folder}")
        _check_context(frame, context, ("job_id", "budget_key", "rep", "spec"),
                       f"Application {kind} CSV {folder}")
        if {"true", "error", "absolute_error", "squared_error"} & set(frame.columns):
            raise ValueError(f"Application table contains unsupported truth/error fields: {path}")
        app[kind].append(frame)


def collect(sources, out, require_complete=False, require_application=False):
    """Collect valid compact fits; write partial results unless completeness is requested."""
    bootstrap()
    import pandas as pd
    from paper_metrics import (
        _EMPTY_DIAGNOSTIC_COLUMNS,
        _EMPTY_PER_REPLICATION_COLUMNS,
        _write_csv,
        aggregate_metrics,
    )

    package, manifest = verify_package()
    jobs, all_ids, reused_ids = _validate_manifest(manifest)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, diagnostics, metadata = [], [], {}
    application = {kind: [] for kind in ("gate", "catt", "contrasts")}
    duplicates = []
    reuse_seen = set()

    def accept(folder, source, allowed_ids=None, is_reuse=False):
        folder = Path(folder)
        record = verify_fit(folder, manifest)
        if not record:
            raise ValueError(f"Incomplete compact fit: {folder}")
        job_id = record.get("job_id")
        if not isinstance(job_id, str) or job_id not in all_ids:
            raise ValueError(f"Unknown job_id in compact fit: {folder}")
        if folder.name != job_id:
            raise ValueError(f"Fit directory does not match job_id {job_id}: {folder}")
        if allowed_ids is not None and job_id not in allowed_ids:
            raise ValueError(f"Fit outside registered source assignment: {job_id}")
        if is_reuse and job_id not in reused_ids:
            raise ValueError(f"Unregistered reused fit: {job_id}")
        if not is_reuse and job_id in reused_ids:
            # A worker checkpoint can repeat a reused fit only when byte and
            # record identity agree; ordinary execution must not create one.
            if job_id not in metadata:
                raise ValueError(f"Reused fit arrived outside reuse/fits: {job_id}")
        if job_id in metadata:
            if metadata[job_id]["record_hash"] != record.get("record_hash"):
                raise ValueError(f"Conflicting repeated fit {job_id}")
            duplicates.append({"job_id": job_id, "source": str(source)})
            if is_reuse:
                reuse_seen.add(job_id)
            return

        _check_fit_files(folder, record)
        job = next(item for item in jobs if item["id"] == job_id)
        per = _read_csv(folder / "per_replication.csv")
        _validate_per_replication(per, record, job, folder)
        diag = _read_csv(folder / "diagnostics.csv")
        _validate_diagnostics(diag, record, job, folder)

        app_output_names = {f"application_{kind}.csv" for kind in application}
        output_app = set(record.get("outputs", {})) & app_output_names
        if job["phase"] in APPLICATION_PHASES and job["kind"] == "application":
            if output_app != app_output_names:
                raise ValueError(f"Application fit has an incomplete table set: {folder}")
            _validate_application_tables(folder, record, job, application, _read_csv)
        elif job["phase"] in APPLICATION_PHASES and job["kind"] == "application_pretrend":
            if output_app:
                raise ValueError(f"Application pretrend fit unexpectedly contains application tables: {folder}")
        elif output_app:
            raise ValueError(f"Non-application fit contains application summaries: {folder}")
        elif job["phase"] in APPLICATION_PHASES:
            raise ValueError(f"Unknown application phase job kind {job['kind']!r}: {folder}")

        metadata[job_id] = {**record, "source": str(source)}
        summaries.append(per)
        diagnostics.append(diag)
        if is_reuse:
            reuse_seen.add(job_id)

    reuse_root = PACKAGE / "reuse" / "fits"
    for folder in sorted(reuse_root.iterdir()) if reuse_root.exists() else []:
        if not folder.is_dir():
            raise ValueError(f"Unexpected file in verified reuse directory: {folder}")
        accept(folder, "verified_reuse", is_reuse=True)
    if reuse_seen != reused_ids:
        missing_reuse = sorted(reused_ids - reuse_seen)
        unexpected_reuse = sorted(reuse_seen - reused_ids)
        raise ValueError(f"Verified reuse set mismatch; missing={missing_reuse}, unexpected={unexpected_reuse}")

    for source in sources:
        source = Path(source)
        if source.is_dir():
            assigned = assignment(source / "bundle" / "assignment.json", manifest)
            fits_root = source / "fits"
            for folder in _fit_folders(fits_root):
                accept(folder, source, set(assigned["job_ids"]))
            continue
        if not source.is_file() or source.suffix.lower() != ".zip":
            raise ValueError(f"Source must be a worker directory or ZIP archive: {source}")
        with tempfile.TemporaryDirectory(prefix="didbcf-nonmain-collect-") as temp:
            root = Path(temp).resolve()
            _extract_zip(source, root)
            assigned = assignment(root / "bundle" / "assignment.json", manifest)
            for folder in _fit_folders(root / "fits"):
                accept(folder, source, set(assigned["job_ids"]))

    frame = (pd.concat(summaries, ignore_index=True) if summaries
             else pd.DataFrame(columns=_EMPTY_PER_REPLICATION_COLUMNS))
    diagnostic = (pd.concat(diagnostics, ignore_index=True) if diagnostics
                  else pd.DataFrame(columns=_EMPTY_DIAGNOSTIC_COLUMNS))
    _write_csv(frame, out / "per_replication.csv")
    _write_csv(diagnostic, out / "diagnostics.csv")

    scalar_tables, surfaces = [], []
    # Empirical application rows have no observed truth. Preserve their raw
    # tables and per-fit rows, while keeping truth-based metrics simulation-only.
    if len(frame):
        simulation = frame.loc[frame["phase"].isin(SIMULATION_PHASES)]
        group_fields = ["phase", "budget_key", "sigma2_shape", "sigma2_rate"]
        if len(simulation):
            for key, group in simulation.groupby(group_fields, dropna=False):
                scalar, surface = aggregate_metrics(group)
                for table, target in ((scalar, scalar_tables), (surface, surfaces)):
                    if len(table):
                        for field, value in zip(group_fields, key):
                            table[field] = value
                        target.append(table)
    scalar = pd.concat(scalar_tables, ignore_index=True) if scalar_tables else pd.DataFrame()
    surface = pd.concat(surfaces, ignore_index=True) if surfaces else pd.DataFrame()
    _write_csv(scalar, out / "scalar_metrics.csv")
    _write_csv(surface, out / "surface_metrics.csv")
    for kind, tables in application.items():
        if tables:
            _write_csv(pd.concat(tables, ignore_index=True), out / f"application_{kind}.csv")
        else:
            from paper_metrics import _APPLICATION_COLUMNS
            empty = pd.DataFrame(columns=(
                *_APPLICATION_COLUMNS[kind], "job_id", "budget_key", "rep"
            ))
            _write_csv(empty, out / f"application_{kind}.csv")

    job_by_id = {job["id"]: job for job in jobs}
    missing_ids = sorted(all_ids - set(metadata))
    missing = [job_by_id[job_id] for job_id in missing_ids]
    missing_columns = ["id", "phase", "kind", "setting", "N", "linearity_degree",
                       "rep", "spec", "sigma2_shape", "sigma2_rate"]
    _write_csv(pd.DataFrame(
        [{field: job.get(field) for field in missing_columns} for job in missing],
        columns=missing_columns,
    ), out / "missing_jobs.csv")

    phase_completed = {
        phase: sum(1 for job_id in metadata if job_by_id[job_id]["phase"] == phase)
        for phase in PHASE_EXPECTED
    }
    status = {
        "manifest_hash": manifest["manifest_hash"],
        "phase_expected": dict(PHASE_EXPECTED),
        "phase_completed": phase_completed,
        "completed_total": len(metadata),
        "expected_total": len(jobs),
        "missing_total": len(missing_ids),
        "missing_ids": missing_ids,
        "missing_job_ids": missing_ids,
        "duplicates_identical": duplicates,
        "diagnostic_filtering": False,
        "application_truth_or_error_metrics_created": False,
        "posterior_arrays_needed_for_collection": False,
    }
    atomic(out / "status.json", status)
    atomic(out / "provenance.json", {
        "package_hash": package["package_hash"],
        "sources": [
            {"path": str(source), "sha256": sha(source) if Path(source).is_file() else None}
            for source in sources
        ],
        "verified_reuse_job_ids": sorted(reuse_seen),
        "fit_record_hashes": {
            job_id: metadata[job_id]["record_hash"] for job_id in sorted(metadata)
        },
    })
    print({key: value for key, value in status.items()
           if key not in {"missing_ids", "missing_job_ids", "duplicates_identical"}})

    incomplete = status["missing_total"] > 0
    application_incomplete = any(
        phase_completed[phase] != PHASE_EXPECTED[phase]
        for phase in APPLICATION_PHASES
    )
    if require_complete and incomplete:
        raise RuntimeError("Non-main campaign incomplete; status and partial tables were saved")
    if require_application and application_incomplete:
        raise RuntimeError("Application phases incomplete; status and collected tables were saved")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-application", action="store_true")
    args = parser.parse_args()
    collect(args.sources, args.out, args.require_complete, args.require_application)
