"""Compact, paper-facing postprocessing for the frozen recovery collector.

Posterior draw arrays and sweep traces are consumed in memory by the pinned
``recovery_collect`` helpers and are never copied into this export.  The
per-replication CSV contains only posterior summaries and evaluation metrics;
``diagnostics.csv`` contains scalar chain diagnostics.  Application summaries
are kept in three small tables (gate, CATT, and contrasts).

Aggregate scalar metrics are truth-based: rows without an observed truth (the
mpdta application) remain ``NaN`` in ``per_replication.csv`` and do not receive
invented errors, coverage, MAPE, or power summaries.  Frozen MAPE is ``NaN``
when all usable truths in a group are within its zero tolerance, since a
percentage error has no defined nonzero denominator in that case.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_EMPTY_PER_REPLICATION_COLUMNS = [
    "job_id", "budget_key", "phase", "kind", "dgp", "setting", "N",
    "linearity_degree", "rep", "spec", "sigma2_shape", "sigma2_rate",
    "estimand_type", "estimand_id", "method", "g", "t", "k",
    "post_mean", "sd", "q025", "q05", "q95", "q975", "p_bayes",
    "true", "surf_rmse", "surf_mae", "surf_mape", "surf_cover90",
    "surf_cover95", "surf_len90", "surf_len95", "surf_n",
]

_EMPTY_DIAGNOSTIC_COLUMNS = [
    "job_id", "budget_key", "parameter", "n_chains", "n_draws_per_chain",
    "post_mean", "post_sd", "rhat", "rhat_second_half", "bulk_ess",
    "tail_ess", "mean_ess", "mean_mcse", "sd_mcse", "batch_mcse",
    "chain_mean_range", "second_half_shift_in_sd", "q025", "q025_mcse",
    "q05", "q05_mcse", "q50", "q50_mcse", "q95", "q95_mcse",
    "q975", "q975_mcse", "chain_set_complete",
]

_APPLICATION_COLUMNS = {
    "gate": [
        "spec", "group", "n_obs", "att_log", "post_sd", "lo95_log",
        "hi95_log", "att_pct", "lo95_pct", "hi95_pct", "p_bayes",
    ],
    "catt": [
        "unit_id", "time", "lpop", "lpop_group", "spec", "catt_log",
        "catt_pct",
    ],
    "contrasts": [
        "spec", "contrast", "diff_log", "post_sd", "lo95_log", "hi95_log",
        "p_bayes",
    ],
}

_DIAGNOSTIC_CONTAINER_TYPES = (list, tuple, dict, set, np.ndarray)
_WILSON_Z_95 = 1.959963984540054


def _contains_container(value: Any) -> bool:
    return isinstance(value, _DIAGNOSTIC_CONTAINER_TYPES)


def _assert_scalar_cells(frame: pd.DataFrame, label: str) -> None:
    """Reject accidental draw arrays or nested payloads before CSV export."""
    for column in frame.columns:
        for value in frame[column].array:
            if _contains_container(value):
                raise ValueError(
                    f"{label}.{column} contains a nested/array value; "
                    "paper exports may contain scalar summaries only"
                )


def _check_numeric_roundtrip(frame: pd.DataFrame, path: Path) -> None:
    """Check that 17-digit CSV formatting preserves every numeric value."""
    if not len(frame.columns):
        if len(frame):
            raise ValueError("Rows without a CSV schema")
        return
    restored = pd.read_csv(path, keep_default_na=True, float_precision="round_trip")
    if list(restored.columns) != list(frame.columns) or len(restored) != len(frame):
        raise IOError(f"CSV schema/row-count roundtrip failed for {path.name}")
    for column in frame.columns:
        if (not pd.api.types.is_numeric_dtype(frame[column].dtype)
                or pd.api.types.is_bool_dtype(frame[column].dtype)):
            continue
        expected = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        actual = pd.to_numeric(restored[column], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        if not np.array_equal(expected, actual, equal_nan=True):
            raise IOError(
                f"Numeric CSV roundtrip changed values in {path.name}:{column}"
            )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a compact CSV and verify its numeric roundtrip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_scalar_cells(frame, path.name)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            float_format="%.17g",
            na_rep="NaN",
            lineterminator="\n",
        )
        _check_numeric_roundtrip(frame, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _diagnostics_frame(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    # Frozen scalar diagnostics include a list of chain seeds. It is provenance
    # rather than a diagnostic and is already available in the run records, so
    # omit that nested field to keep this CSV scalar-only and small.
    keep = []
    for column in frame.columns:
        if not any(_contains_container(value) for value in frame[column].array):
            keep.append(column)
    frame = frame.loc[:, keep]
    if frame.empty and not len(frame.columns):
        return pd.DataFrame(columns=_EMPTY_DIAGNOSTIC_COLUMNS)
    return frame


def _per_replication_frame(value: list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    if frame.empty and not len(frame.columns):
        return pd.DataFrame(columns=_EMPTY_PER_REPLICATION_COLUMNS)
    return frame


def _application_frame(value: pd.DataFrame | None, kind: str,
                       job: dict, budget_key: str) -> pd.DataFrame:
    if value is None:
        frame = pd.DataFrame(columns=_APPLICATION_COLUMNS[kind])
    elif not isinstance(value, pd.DataFrame):
        raise TypeError(f"Frozen application output {kind!r} is not a DataFrame")
    else:
        frame = value.copy()
    frame["job_id"] = job["id"]
    frame["budget_key"] = budget_key
    frame["rep"] = job["rep"]
    return frame


def summarize_job(job: dict, budget_key: str, records: dict,
                  out_dir: Path) -> dict:
    """Write one completed job/budget's paper metrics and scalar diagnostics.

    ``records`` is the complete-sibling-chain mapping returned by the frozen
    recovery collector.  The caller must put that pinned collector and the
    frozen ``did_bcf_revision`` package on ``sys.path`` before calling here.
    Raw posterior arrays are loaded only inside those collector functions and
    are never written by this adapter.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collector = importlib.import_module("recovery_collect")

    if job.get("kind") in {"pretrend", "application_pretrend"}:
        per_rep_value, diagnostics_value = collector.pretrend_rows_for_job(
            job, records, budget_key
        )
    else:
        rows, diagnostics_value = collector.scalar_rows_for_job(
            job, records, budget_key
        )
        per_rep_value = rows

    per_replication = _per_replication_frame(per_rep_value)
    diagnostics = _diagnostics_frame(diagnostics_value)
    _assert_scalar_cells(per_replication, "per_replication")
    _assert_scalar_cells(diagnostics, "diagnostics")

    files: dict[str, Path] = {
        "per_replication.csv": out_dir / "per_replication.csv",
        "diagnostics.csv": out_dir / "diagnostics.csv",
    }
    _write_csv(per_replication, files["per_replication.csv"])
    _write_csv(diagnostics, files["diagnostics.csv"])

    application_row_counts: dict[str, int] = {}
    if job.get("kind") == "application":
        application = None
        if len(per_replication):
            application = collector.application_outputs(job, records, budget_key)
            if application is None:
                raise ValueError(
                    "Application per-replication rows exist but frozen "
                    "application_outputs returned no summaries"
                )
            missing = {"gate", "catt", "contrasts"} - set(application)
            if missing:
                raise ValueError(
                    "Frozen application outputs omitted required tables: "
                    + ", ".join(sorted(missing))
                )
        for kind in ("gate", "catt", "contrasts"):
            output_name = f"application_{kind}.csv"
            app_frame = _application_frame(
                application[kind] if application is not None else None,
                kind, job, budget_key,
            )
            files[output_name] = out_dir / output_name
            _write_csv(app_frame, files[output_name])
            application_row_counts[kind] = int(len(app_frame))

    outputs = {name: _sha256(path) for name, path in files.items()}
    return {
        "outputs": outputs,
        "per_replication_rows": int(len(per_replication)),
        "diagnostic_rows": int(len(diagnostics)),
        "application_output_rows": application_row_counts,
    }


def _key_component(value: Any) -> Any:
    if pd.isna(value):
        return ("<NA>",)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _group_key(values: Any) -> tuple:
    if not isinstance(values, tuple):
        values = (values,)
    return tuple(_key_component(value) for value in values)


def _wilson_interval(successes: int, n: int,
                     z: float = _WILSON_Z_95) -> tuple[float, float]:
    """Two-sided 95% Wilson interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def _finite_coverage(group: pd.DataFrame, lower: str,
                     upper: str) -> tuple[int, int, float, float, float]:
    if not {"true", lower, upper}.issubset(group.columns):
        return 0, 0, np.nan, np.nan, np.nan
    truth = pd.to_numeric(group["true"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(group[lower], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(group[upper], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(truth) & np.isfinite(lo) & np.isfinite(hi)
    n = int(valid.sum())
    if not n:
        return 0, 0, np.nan, np.nan, np.nan
    successes = int(np.sum((lo[valid] <= truth[valid]) &
                           (truth[valid] <= hi[valid])))
    estimate = successes / n
    lower_wilson, upper_wilson = _wilson_interval(successes, n)
    return successes, n, float(estimate), lower_wilson, upper_wilson


def aggregate_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate truth-based scalar and CATT-surface paper metrics.

    This delegates the definitions and estimand grouping to the frozen
    ``did_bcf_revision.metrics`` functions.  It then replaces frozen
    ``mcse_bias`` with sample SD(error)/sqrt(n), retaining the original estimate
    SD-based value as ``legacy_mcse_bias_estimate_sd``.  Scalar interval
    coverage includes success counts and 95% Wilson intervals; its denominator
    is the number of replications with finite truth and finite bounds for that
    interval level, which may differ across 90% and 95% intervals.

    Rows with unavailable truth are excluded from scalar evaluation metrics and
    stay unavailable in the per-replication table.  MAPE remains ``NaN`` for a
    group with no nonzero truth, matching the frozen zero-denominator rule.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    metric_module = importlib.import_module("did_bcf_revision.metrics")

    if frame.empty or "estimand_type" not in frame.columns:
        return pd.DataFrame(), pd.DataFrame()

    surface = metric_module.surface_metrics(frame)
    if "true" not in frame.columns:
        return pd.DataFrame(), surface
    truth = pd.to_numeric(frame["true"], errors="coerce")
    scalar_input = frame.loc[truth.notna()].copy()
    if scalar_input.empty:
        return pd.DataFrame(), surface

    scalar = metric_module.compute_metrics(scalar_input)
    if scalar.empty:
        return scalar, surface

    keys = [key for key in metric_module.GROUP_KEYS
            if key in scalar_input.columns and key in scalar.columns]
    if not keys:
        raise ValueError("Frozen scalar metrics exposed no grouping columns")
    source_groups = {
        _group_key(group_key): group
        for group_key, group in scalar_input.groupby(keys, dropna=False)
    }

    for index, result in scalar.iterrows():
        group = source_groups.get(_group_key(tuple(result[key] for key in keys)))
        if group is None:
            raise ValueError("Could not match frozen metric row to its replications")

        legacy_mcse = result.get("mcse_bias", np.nan)
        estimand_id = str(group["estimand_id"].iloc[0]) if "estimand_id" in group else ""
        if metric_module.is_decision_estimand(estimand_id):
            errors = np.empty(0, dtype=float)
        elif {"post_mean", "true"}.issubset(group.columns):
            estimate = pd.to_numeric(group["post_mean"], errors="coerce").to_numpy(
                dtype=float
            )
            actual_truth = pd.to_numeric(group["true"], errors="coerce").to_numpy(
                dtype=float
            )
            usable = np.isfinite(estimate) & np.isfinite(actual_truth)
            errors = estimate[usable] - actual_truth[usable]
        else:
            errors = np.empty(0, dtype=float)

        error_n = int(errors.size)
        error_sd_sample = (float(np.std(errors, ddof=1))
                           if error_n > 1 else np.nan)
        error_sd_population = (float(np.std(errors, ddof=0))
                               if error_n else np.nan)
        corrected_mcse = (error_sd_sample / np.sqrt(error_n)
                          if error_n > 1 else np.nan)
        scalar.loc[index, "legacy_mcse_bias_estimate_sd"] = legacy_mcse
        scalar.loc[index, "error_sd_sample"] = error_sd_sample
        scalar.loc[index, "error_sd_population"] = error_sd_population
        scalar.loc[index, "mcse_bias"] = corrected_mcse

        if metric_module.is_decision_estimand(estimand_id):
            continue

        for level, lower, upper in (
            (90, "q05", "q95"), (95, "q025", "q975"),
        ):
            count, n_coverage, coverage, wilson_lo, wilson_hi = _finite_coverage(
                group, lower, upper
            )
            scalar.loc[index, f"cover{level}_count"] = count
            scalar.loc[index, f"cover{level}_n"] = n_coverage
            scalar.loc[index, f"cover{level}"] = coverage
            scalar.loc[index, f"cover{level}_wilson95_lower"] = wilson_lo
            scalar.loc[index, f"cover{level}_wilson95_upper"] = wilson_hi
            scalar.loc[index, f"mcse_cover{level}"] = (
                float(np.sqrt(coverage * (1.0 - coverage) / n_coverage))
                if n_coverage else np.nan
            )

    # Convert count columns back to integers. If a metric family has no finite
    # intervals, its denominator/count are still exactly zero rather than NaN.
    for column in ("cover90_count", "cover90_n", "cover95_count", "cover95_n"):
        if column in scalar:
            scalar[column] = pd.to_numeric(scalar[column], errors="coerce").astype("Int64")
    return scalar, surface


__all__ = ["summarize_job", "aggregate_metrics"]
