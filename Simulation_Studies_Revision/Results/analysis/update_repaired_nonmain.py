#!/usr/bin/env python3
"""Build the versioned non-main repaired-results overlay.

This is a deterministic metrics-only adapter. It leaves the historical
aggregates and the earlier repaired-main overlay untouched, retains old
non-main DiD-BCF and comparator rows under explicit historical method labels,
and adds the completed 2026-10-26 diagnostic and PT-impact results. It also
creates paired prior-sensitivity tables from saved replication summaries.

No model is sampled here and no posterior arrays are read.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
REVISION = RESULTS.parent
REPO = REVISION.parent
PACKAGE = REVISION / "DiD_BCF" / "Nonmain_Rerun_20261014"
COLLECTED = PACKAGE / "Results" / "collected_20261026"
OUT = RESULTS / "aggregated_repaired_20261026"
MAIN = RESULTS / "aggregated_repaired_20261014"
COMPLETION = REVISION / "Theory_Calibration" / "results" / "aggregated_completion"

EXPECTED_MANIFEST_HASH = "d4c6fe4d5369638885b5d3a9a4f5077e5776990be6aa73d2e000188eb20bddca"
EXPECTED_PACKAGE_HASH = "99c37cf5731cf9d9a10096aefa3df1932c94902ac705f803c350cd2fee5f4487"
EXPECTED_PHASES = {
    "diagnostic": 4000,
    "pt_impact": 4000,
    "sensitivity": 1200,
    "application": 9,
    "application_sensitivity": 18,
}
PT_SETTINGS = (
    "PT_conditional", "PT_hold", "PT_violation_a20", "PT_violation_g05",
    "PT_violation_g10", "PT_violation_g20", "PT_violation_g40",
    "PT_violation_het10", "PT_violation_het20", "PT_violation_het40",
)
PT_DEGREES = (1, 3)
SENSITIVITY_SETTINGS = ("B2_sweep", "B2_sweep_serial")
SENSITIVITY_NS = (50, 200, 800)
SENSITIVITY_DEGREE = 1
SENSITIVITY_REPS = tuple(range(100))
SENSITIVITY_RATES = (0.1, 1.0, 10.0)
BOOTSTRAP_REPS = 5000
BOOTSTRAP_BASE_SEED = 20261026

METRIC_KEY = ["setting", "dgp", "N", "linearity_degree", "estimand_type",
              "estimand_id", "method"]
SURFACE_KEY = ["setting", "dgp", "N", "linearity_degree", "method"]
PT_CELL_KEY = ["setting", "dgp", "N", "linearity_degree"]
TARGET_SENSITIVITY_KEY = ["setting", "dgp", "N", "linearity_degree", "rep",
                          "estimand_type", "estimand_id", "method"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"job_id": str},
                       float_precision="round_trip")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g", na_rep="NaN",
                 lineterminator="\n")
    temporary.replace(path)


def write_json_last(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _single_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one {label} row, found {len(rows)}")
    return rows.iloc[0]


def _wilson95(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate_record(p_values: pd.Series | np.ndarray) -> tuple[float, int, int, float, float]:
    values = pd.to_numeric(pd.Series(p_values), errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return np.nan, 0, 0, np.nan, np.nan
    count = int(np.sum(values < 0.025))
    lower, upper = _wilson95(count, n)
    return count / n, count, n, lower, upper


def _assert_rejection_row(row: pd.Series, label: str) -> tuple[float, int, int, float, float]:
    n = int(row["n_reps"])
    rate = float(row["reject05"])
    if n <= 0 or not np.isfinite(rate):
        raise ValueError(f"Invalid historical reject05 row for {label}")
    count = int(round(n * rate))
    if not math.isclose(count / n, rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"reject05 does not map to an integer count for {label}")
    lower, upper = _wilson95(count, n)
    return rate, count, n, lower, upper


def _p_tail(frame: pd.DataFrame, label: str) -> pd.Series:
    """Read one-tail posterior masses and validate the capped two-sided form."""
    require_columns(frame, ["rep"], label)
    has_tail = "p_bayes_tail_min" in frame
    has_two_sided = "p_bayes_two_sided" in frame
    if not has_tail and not has_two_sided:
        raise ValueError(f"No posterior tail column for {label}")
    tail = (pd.to_numeric(frame["p_bayes_tail_min"], errors="coerce")
            if has_tail else None)
    two_sided = (pd.to_numeric(frame["p_bayes_two_sided"], errors="coerce")
                 if has_two_sided else None)
    if tail is not None and two_sided is not None:
        common = tail.notna() & two_sided.notna()
        # The stored two-sided measure is min(1, 2 * tail), so above 0.5
        # dividing by two does not recover the one-tail mass. Decisions use
        # the stored tail directly whenever it is available.
        expected = (2.0 * tail[common]).clip(upper=1.0)
        if common.any() and not np.allclose(expected, two_sided[common],
                                            rtol=0, atol=1e-12):
            raise ValueError(f"Tail and capped two-sided columns disagree for {label}")
    if tail is not None:
        return tail
    return two_sided / 2.0


def verify_inputs() -> tuple[dict, dict, dict, dict]:
    """Check the frozen non-main identity and completed collector outputs."""
    for path in (PACKAGE / "manifest.json", PACKAGE / "package.json",
                 COLLECTED / "status.json", COLLECTED / "provenance.json",
                 COLLECTED / "per_replication.csv", COLLECTED / "scalar_metrics.csv",
                 COLLECTED / "surface_metrics.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    package = json.loads((PACKAGE / "package.json").read_text())
    status = json.loads((COLLECTED / "status.json").read_text())
    collected_provenance = json.loads((COLLECTED / "provenance.json").read_text())

    if _canonical_digest({k: v for k, v in manifest.items()
                          if k != "manifest_hash"}) != manifest.get("manifest_hash"):
        raise ValueError("Frozen non-main manifest hash is invalid")
    if _canonical_digest({k: v for k, v in package.items()
                          if k != "package_hash"}) != package.get("package_hash"):
        raise ValueError("Frozen non-main package hash is invalid")
    if manifest.get("manifest_hash") != EXPECTED_MANIFEST_HASH:
        raise ValueError("Unexpected frozen non-main manifest hash")
    if package.get("package_hash") != EXPECTED_PACKAGE_HASH:
        raise ValueError("Unexpected frozen non-main package hash")
    if package.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("Package and manifest identities differ")
    if collected_provenance.get("package_hash") != package["package_hash"]:
        raise ValueError("Collected provenance package identity differs")
    if status.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("Collected status manifest identity differs")

    if status.get("expected_total") != 9227 or status.get("completed_total") != 9227:
        raise ValueError("Non-main collection does not contain all 9,227 jobs")
    if status.get("missing_total") != 0 or status.get("missing_ids"):
        raise ValueError("Non-main collection has missing jobs")
    if status.get("phase_expected") != EXPECTED_PHASES:
        raise ValueError("Collector expected phase counts differ from frozen campaign")
    if status.get("phase_completed") != EXPECTED_PHASES:
        raise ValueError("Collector completed phase counts differ from frozen campaign")
    if manifest.get("phase_expected") != EXPECTED_PHASES:
        raise ValueError("Manifest phase counts differ from frozen campaign")
    if status.get("diagnostic_filtering") is not False:
        raise ValueError("Collector diagnostic-filtering status is not false")
    if not (MAIN / "provenance.json").is_file():
        raise FileNotFoundError(MAIN / "provenance.json")

    main_provenance = json.loads((MAIN / "provenance.json").read_text())
    if main_provenance.get("status") != "PASS" or main_provenance.get("new_fits") != 0:
        raise ValueError("Prior repaired-main overlay is not a completed immutable input")
    for name, expected in main_provenance.get("outputs", {}).items():
        path = MAIN / name
        if path.is_file() and sha256(path) != expected:
            raise ValueError(f"Prior repaired-main output hash changed: {name}")
    return manifest, package, status, collected_provenance


def _target_filter(frame: pd.DataFrame, phase: str | None = None,
                   rate: float | None = None) -> pd.Series:
    mask = (frame["setting"].isin(PT_SETTINGS)
            & frame["N"].eq(200)
            & frame["linearity_degree"].isin(PT_DEGREES))
    if phase is not None:
        mask &= frame["phase"].eq(phase)
    if rate is not None:
        mask &= pd.to_numeric(frame["sigma2_rate"], errors="coerce").eq(rate)
    return mask


def _check_nonmain_shapes(manifest: dict, per: pd.DataFrame,
                          scalar: pd.DataFrame, surface: pd.DataFrame) -> None:
    expected_jobs = pd.DataFrame(manifest["jobs"])
    require_columns(per, ["job_id", "budget_key", "phase", "dgp", "setting", "N",
                          "linearity_degree", "rep", "spec", "sigma2_shape",
                          "sigma2_rate", "estimand_type", "estimand_id", "method"],
                    "nonmain per_replication")
    require_columns(scalar, ["phase", "budget_key", "sigma2_shape", "sigma2_rate",
                             "dgp", "setting", "N", "linearity_degree", "spec",
                             "estimand_type", "estimand_id", "method", "n_reps"],
                    "nonmain scalar_metrics")
    require_columns(surface, ["phase", "budget_key", "sigma2_shape", "sigma2_rate",
                              "dgp", "setting", "N", "linearity_degree", "spec",
                              "method", "n_reps"], "nonmain surface_metrics")
    if set(per["job_id"].astype(str)) != set(expected_jobs["id"].astype(str)):
        raise ValueError("Per-replication job IDs do not match the frozen manifest")
    if per.duplicated(["job_id", "budget_key", "estimand_type", "estimand_id", "method"]).any():
        raise ValueError("Repeated nonmain job/estimand/method rows")
    if not set(per["phase"].astype(str)).issubset(set(EXPECTED_PHASES)):
        raise ValueError("Unexpected phase in nonmain per-replication source")

    diagnostic_jobs = expected_jobs[expected_jobs.phase.eq("diagnostic")]
    impact_jobs = expected_jobs[expected_jobs.phase.eq("pt_impact")]
    for label, jobs in (("diagnostic", diagnostic_jobs), ("pt_impact", impact_jobs)):
        rows = jobs[(jobs.setting.isin(PT_SETTINGS)) & jobs.N.eq(200)
                    & jobs.linearity_degree.isin(PT_DEGREES)]
        counts = rows.groupby(["setting", "linearity_degree"]).rep.nunique()
        if len(counts) != len(PT_SETTINGS) * len(PT_DEGREES) or not counts.eq(200).all():
            raise ValueError(f"Frozen manifest has unexpected {label} pretrend replication counts")

    # All current diagnostic/impact design cells must be complete even when
    # the larger collection is otherwise complete.
    for label, phase in (("diagnostic", "diagnostic"), ("pt_impact", "pt_impact")):
        subset = per.loc[_target_filter(per, phase=phase, rate=1.0)]
        if subset.empty:
            raise ValueError(f"Missing {label} replications")
        counts = subset.groupby(["setting", "linearity_degree"]).rep.nunique()
        if len(counts) != len(PT_SETTINGS) * len(PT_DEGREES) or not counts.eq(200).all():
            raise ValueError(f"Incomplete {label} data: expected 200 reps per PT cell")
        if not subset.sigma2_shape.eq(2.0).all():
            raise ValueError(f"Unexpected variance-prior shape in {label} data")


def _validate_raw_pretrend(per: pd.DataFrame, scalar: pd.DataFrame) -> None:
    current = per.loc[_target_filter(per, phase="diagnostic", rate=1.0)].copy()
    current = current[current.method.eq("pretrend")]
    if not {"PRE", "PRE_SUBC"}.issubset(set(current.estimand_type.unique())):
        raise ValueError("Current raw pretrend rows omit PRE/PRE_SUBC targets")
    for setting in PT_SETTINGS:
        for degree in PT_DEGREES:
            group = current[(current.setting == setting)
                            & (current.linearity_degree == degree)]
            for etype, eid in (("PRE", "slope"), ("PRE", "any_bonf"),
                               ("PRE_SUBC", "X1_slope")):
                rows = group[(group.estimand_type == etype)
                             & (group.estimand_id == eid)]
                if len(rows) != 200 or rows.rep.nunique() != 200:
                    raise ValueError(f"Expected 200 current {setting}/d{degree}/{etype}/{eid} rows")
                p = pd.to_numeric(rows.p_bayes, errors="coerce")
                if not np.isfinite(p).all():
                    raise ValueError(f"Nonfinite posterior decision tail at {setting}/d{degree}/{eid}")
            # The stored Bonferroni row must agree with the three PRE event-time
            # rows on every replication, not only after aggregation.
            pre = group[(group.estimand_type == "PRE")
                        & group.estimand_id.astype(str).str.startswith("k=")]
            pre["_tail"] = pd.to_numeric(pre.p_bayes, errors="coerce")
            if pre.groupby("rep").size().ne(3).any():
                raise ValueError(f"Expected exactly three pre-period rows at {setting}/d{degree}")
            expected = (pre.groupby("rep")["_tail"].min() * 3.0).clip(upper=1.0)
            stored = group[(group.estimand_type == "PRE")
                           & group.estimand_id.eq("any_bonf")].set_index("rep").p_bayes
            if not np.allclose(expected.sort_index().to_numpy(float),
                               stored.sort_index().to_numpy(float), rtol=0, atol=1e-12):
                raise ValueError(f"PRE any_bonf differs from three-k Bonferroni at {setting}/d{degree}")

            for etype, eid in (("PRE", "slope"), ("PRE", "any_bonf"),
                               ("PRE_SUBC", "X1_slope")):
                metric = _single_row(
                    scalar,
                    _target_filter(scalar, phase="diagnostic", rate=1.0)
                    & scalar.setting.eq(setting)
                    & scalar.linearity_degree.eq(degree)
                    & scalar.method.eq("pretrend")
                    & scalar.estimand_type.eq(etype)
                    & scalar.estimand_id.eq(eid),
                    f"raw scalar metric {setting}/d{degree}/{etype}/{eid}",
                )
                if int(metric.n_reps) != 200 or int(metric.n_reps_attempted) != 200:
                    raise ValueError(f"Raw scalar denominator is not 200 for {setting}/d{degree}/{eid}")
                direct, count, n, _, _ = _rate_record(
                    group[(group.estimand_type == etype)
                          & group.estimand_id.eq(eid)].p_bayes
                )
                if not math.isclose(direct, float(metric.reject05), rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Raw reject05 mismatch for {setting}/d{degree}/{eid}")
                if count != int(round(float(metric.reject05) * n)):
                    raise ValueError(f"Raw rejection count mismatch for {setting}/d{degree}/{eid}")


def _raw_pretrend_summary(per: pd.DataFrame, scalar: pd.DataFrame) -> pd.DataFrame:
    current = per.loc[_target_filter(per, phase="diagnostic", rate=1.0)]
    current = current[current.method.eq("pretrend")]
    rows = []
    for setting in PT_SETTINGS:
        for degree in PT_DEGREES:
            group = current[(current.setting == setting)
                            & (current.linearity_degree == degree)]
            metric = _single_row(
                scalar,
                _target_filter(scalar, phase="diagnostic", rate=1.0)
                & scalar.setting.eq(setting) & scalar.linearity_degree.eq(degree)
                & scalar.method.eq("pretrend") & scalar.estimand_type.eq("PRE")
                & scalar.estimand_id.eq("slope"),
                f"raw PRE slope summary {setting}/d{degree}",
            )
            decisions = {}
            for tag, etype, eid in (("slope", "PRE", "slope"),
                                    ("subgroup", "PRE_SUBC", "X1_slope"),
                                    ("anyk", "PRE", "any_bonf")):
                p = group[(group.estimand_type == etype)
                          & group.estimand_id.eq(eid)].p_bayes
                rate, count, n, lower, upper = _rate_record(p)
                decisions[tag] = (rate, count, n, lower, upper)
            slope = decisions["slope"]
            rows.append({
                "setting": setting, "linearity_degree": degree,
                "method": "pretrend", "slope_detect": slope[0],
                "slope_count": slope[1], "slope_n": slope[2],
                "slope_wilson95_lower": slope[3],
                "slope_wilson95_upper": slope[4],
                "subgroup_detect": decisions["subgroup"][0],
                "subgroup_count": decisions["subgroup"][1],
                "subgroup_n": decisions["subgroup"][2],
                "anyk_detect": decisions["anyk"][0],
                "anyk_n": decisions["anyk"][2],
                "mean_true": float(metric.mean_true),
                "slope_bias": float(metric.bias),
                "slope_cover95": float(metric.cover95),
            })
    return pd.DataFrame(rows)


def _reference_fold_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_path = COMPLETION / "combined_per_replication.csv"
    metric_path = COMPLETION / "metrics_mean_median.csv"
    if not per_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError("Missing historical reference-fold completion inputs")
    per = read_csv(per_path)
    metric = read_csv(metric_path)
    require_columns(per, ["family", "design", "degree", "N", "task_estimator",
                          "method", "estimand_type", "estimand_id", "rep", "true",
                          "p_bayes_tail_min", "p_bayes_two_sided"],
                    "reference-fold per-replication file")
    require_columns(metric, ["family", "design", "degree", "N", "task_estimator",
                             "method", "estimand_type", "estimand_id", "point_summary",
                             "n_reps", "mean_true", "bias", "cover95",
                             "empirical_sd_error"], "reference-fold metrics file")
    return per, metric


def _reference_fold_pretrend_summary(per: pd.DataFrame,
                                     metric: pd.DataFrame) -> pd.DataFrame:
    base = per[(per.family == "correction_completion")
               & (per.task_estimator == "reference_fold_convolution_rf")
               & (per.method == "reference_fold_pretrend")
               & (per.design.isin(PT_SETTINGS))
               & (per.N == 200) & per.degree.isin(PT_DEGREES)]
    mbase = metric[(metric.family == "correction_completion")
                   & (metric.task_estimator == "reference_fold_convolution_rf")
                   & (metric.method == "reference_fold_pretrend")
                   & (metric.point_summary == "mean")
                   & (metric.design.isin(PT_SETTINGS))
                   & (metric.N == 200) & metric.degree.isin(PT_DEGREES)]
    rows = []
    for setting in PT_SETTINGS:
        for degree in PT_DEGREES:
            group = base[(base.design == setting) & (base.degree == degree)]
            mean_metric = _single_row(
                mbase, mbase.design.eq(setting) & mbase.degree.eq(degree)
                & mbase.estimand_type.eq("PRE") & mbase.estimand_id.eq("slope"),
                f"reference-fold slope metric {setting}/d{degree}",
            )
            decisions = {}
            for tag, etype, eid in (("slope", "PRE", "slope"),
                                    ("subgroup", "PRE_SUBC", "X1_slope")):
                target = group[(group.estimand_type == etype)
                               & (group.estimand_id == eid)].copy()
                if len(target) != 200 or target.rep.nunique() != 200:
                    raise ValueError(f"Reference-fold {setting}/d{degree}/{eid} is not n=200")
                tail = _p_tail(target, f"reference-fold {setting}/d{degree}/{eid}")
                decisions[tag] = _rate_record(tail)
                if decisions[tag][2] != 200 or int(mean_metric.n_reps) != 200:
                    raise ValueError(f"Reference-fold metric denominator differs for {setting}/d{degree}/{eid}")
            pre = group[(group.estimand_type == "PRE")
                        & group.estimand_id.astype(str).str.startswith("k=")].copy()
            tails = _p_tail(pre, f"reference-fold any-k {setting}/d{degree}")
            pre["_tail"] = tails.to_numpy(float)
            if pre.groupby("rep").size().ne(3).any():
                raise ValueError(f"Reference-fold any-k set is not three periods at {setting}/d{degree}")
            bonf = (pre.groupby("rep")["_tail"].min() * 3.0).clip(upper=1.0)
            anyk = _rate_record(bonf)

            # Recompute the mean-summary target from the saved records, then
            # compare it with the archived completion metric row.
            slope_rows = group[(group.estimand_type == "PRE")
                               & (group.estimand_id == "slope")]
            err = pd.to_numeric(slope_rows.post_mean, errors="coerce") - pd.to_numeric(slope_rows.true, errors="coerce")
            usable = np.isfinite(err.to_numpy(float))
            if usable.sum() != 200:
                raise ValueError(f"Reference-fold slope error rows are incomplete at {setting}/d{degree}")
            recomputed_bias = float(err.to_numpy(float)[usable].mean())
            if not math.isclose(recomputed_bias, float(mean_metric.bias), rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Reference-fold bias check failed at {setting}/d{degree}")
            slope = decisions["slope"]
            rows.append({
                "setting": setting, "linearity_degree": degree,
                "method": "reference_fold_pretrend", "slope_detect": slope[0],
                "slope_count": slope[1], "slope_n": slope[2],
                "slope_wilson95_lower": slope[3],
                "slope_wilson95_upper": slope[4],
                "subgroup_detect": decisions["subgroup"][0],
                "subgroup_count": decisions["subgroup"][1],
                "subgroup_n": decisions["subgroup"][2],
                "anyk_detect": anyk[0], "anyk_n": anyk[2],
                "mean_true": float(mean_metric.mean_true),
                "slope_bias": float(mean_metric.bias),
                "slope_cover95": float(mean_metric.cover95),
            })
    return pd.DataFrame(rows)


def _historical_published_summary(pub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = pub[(pub.setting.isin(PT_SETTINGS)) & pub.N.eq(200)
              & pub.linearity_degree.isin(PT_DEGREES)
              & pub.estimand_type.eq("PRE") & pub.estimand_id.eq("slope")]
    for slope_row in sub.itertuples(index=False):
        source_method = str(slope_row.method)
        if source_method in {"plain", "corrected", "plain_historical", "corrected_historical"}:
            continue
        method = "pretrend_historical" if source_method == "pretrend" else source_method
        setting, degree = slope_row.setting, int(slope_row.linearity_degree)
        slope_rate = _assert_rejection_row(pd.Series(slope_row._asdict()),
                                           f"published {setting}/d{degree}/{method}/slope")
        subgroup = pub[(pub.setting == setting) & pub.N.eq(200)
                       & pub.linearity_degree.eq(degree)
                       & pub.method.eq(source_method)
                       & pub.estimand_type.eq("PRE_SUBC")
                       & pub.estimand_id.eq("X1_slope")]
        if len(subgroup) > 1:
            raise ValueError(f"Duplicate historical PRE_SUBC/X1_slope row at {setting}/d{degree}/{method}")
        if len(subgroup):
            subgroup_rate = _assert_rejection_row(subgroup.iloc[0],
                f"published {setting}/d{degree}/{method}/X1_slope")
        else:
            subgroup_rate = (np.nan, 0, 0, np.nan, np.nan)
        anyk_rows = pub[(pub.setting == setting) & pub.N.eq(200)
                        & pub.linearity_degree.eq(degree)
                        & pub.method.eq(source_method) & pub.estimand_type.eq("PRE")
                        & pub.estimand_id.eq("any_bonf")]
        if len(anyk_rows) > 1:
            raise ValueError(f"Duplicate historical any_bonf row at {setting}/d{degree}/{method}")
        anyk_rate = (_assert_rejection_row(anyk_rows.iloc[0],
            f"published {setting}/d{degree}/{method}/any_bonf")
                     if len(anyk_rows) else (np.nan, 0, 0, np.nan, np.nan))
        rows.append({
            "setting": setting, "linearity_degree": degree, "method": method,
            "slope_detect": slope_rate[0], "slope_count": slope_rate[1],
            "slope_n": slope_rate[2], "slope_wilson95_lower": slope_rate[3],
            "slope_wilson95_upper": slope_rate[4],
            "subgroup_detect": subgroup_rate[0], "subgroup_count": subgroup_rate[1],
            "subgroup_n": subgroup_rate[2], "anyk_detect": anyk_rate[0],
            "anyk_n": anyk_rate[2], "mean_true": float(slope_row.mean_true),
            "slope_bias": float(slope_row.bias),
            "slope_cover95": float(slope_row.cover95),
        })
    return pd.DataFrame(rows)


def _pretrend_summary(per: pd.DataFrame, scalar: pd.DataFrame,
                      pub: pd.DataFrame) -> pd.DataFrame:
    _validate_raw_pretrend(per, scalar)
    frames = [_raw_pretrend_summary(per, scalar)]
    ref_per, ref_metric = _reference_fold_records()
    frames.append(_reference_fold_pretrend_summary(ref_per, ref_metric))
    frames.append(_historical_published_summary(pub))
    summary = pd.concat(frames, ignore_index=True, sort=False)
    summary = summary[[
        "setting", "linearity_degree", "method", "slope_detect", "slope_count",
        "slope_n", "slope_wilson95_lower", "slope_wilson95_upper",
        "subgroup_detect", "subgroup_count", "subgroup_n", "anyk_detect",
        "anyk_n", "mean_true", "slope_bias", "slope_cover95",
    ]]
    if summary.duplicated(["setting", "linearity_degree", "method"]).any():
        raise ValueError("Pretrend summary has duplicate setting/degree/method rows")
    required = {"pretrend", "reference_fold_pretrend", "twfe_es", "did_dr", "wang"}
    for setting in PT_SETTINGS:
        for degree in PT_DEGREES:
            seen = set(summary[(summary.setting == setting)
                               & (summary.linearity_degree == degree)].method)
            if not required.issubset(seen):
                raise ValueError(f"Pretrend summary is missing required methods at {setting}/d{degree}: {required-seen}")
    return summary.sort_values(["linearity_degree", "setting", "method"],
                               kind="stable").reset_index(drop=True)


def _pretrend_profiles(per: pd.DataFrame) -> pd.DataFrame:
    current = per.loc[_target_filter(per, phase="diagnostic", rate=1.0)]
    current = current[current.method.eq("pretrend")
                      & current.estimand_type.eq("PRE")
                      & current.estimand_id.isin(["k=-4", "k=-3", "k=-2"])].copy()
    rows = []
    for (setting, degree, method, estimand_id), group in current.groupby(
            ["setting", "linearity_degree", "method", "estimand_id"],
            dropna=False, sort=True):
        error = (pd.to_numeric(group.post_mean, errors="coerce")
                 - pd.to_numeric(group.true, errors="coerce")).to_numpy(float)
        truth = pd.to_numeric(group.true, errors="coerce").to_numpy(float)
        valid = np.isfinite(error) & np.isfinite(truth)
        if valid.sum() != 200:
            raise ValueError(f"Incomplete current pretrend profile {setting}/d{degree}/{estimand_id}")
        rows.append({"setting": setting, "linearity_degree": int(degree),
                     "method": method, "estimand_id": estimand_id,
                     "mean_true": float(np.mean(truth[valid])),
                     "bias": float(np.mean(error[valid])),
                     "empirical_sd_error": float(np.std(error[valid], ddof=1))})

    ref_per, ref_metric = _reference_fold_records()
    ref = ref_metric[(ref_metric.family == "correction_completion")
                     & (ref_metric.task_estimator == "reference_fold_convolution_rf")
                     & (ref_metric.method == "reference_fold_pretrend")
                     & (ref_metric.point_summary == "mean")
                     & (ref_metric.design.isin(PT_SETTINGS))
                     & (ref_metric.N == 200) & ref_metric.degree.isin(PT_DEGREES)
                     & (ref_metric.estimand_type == "PRE")
                     & (ref_metric.estimand_id.isin(["k=-4", "k=-3", "k=-2"]))]
    expected_rows = len(PT_SETTINGS) * len(PT_DEGREES) * 3
    if len(ref) != expected_rows:
        raise ValueError(f"Expected {expected_rows} reference-fold profile metrics, found {len(ref)}")
    for row in ref.itertuples(index=False):
        rows.append({"setting": row.design, "linearity_degree": int(row.degree),
                     "method": "reference_fold_pretrend", "estimand_id": row.estimand_id,
                     "mean_true": float(row.mean_true), "bias": float(row.bias),
                     "empirical_sd_error": float(row.empirical_sd_error)})
    result = pd.DataFrame(rows)
    if result.duplicated(["setting", "linearity_degree", "method", "estimand_id"]).any():
        raise ValueError("Pretrend profiles have duplicate keys")
    return result.sort_values(["linearity_degree", "setting", "method", "estimand_id"],
                              kind="stable").reset_index(drop=True)


def _validate_prior_sensitivity(main_per: pd.DataFrame, nonmain_per: pd.DataFrame,
                                main_scalar: pd.DataFrame,
                                nonmain_scalar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = [(setting, n) for setting in SENSITIVITY_SETTINGS for n in SENSITIVITY_NS]
    main_rows = main_per[(main_per.phase == "main")
                         & (main_per.spec == "structured")
                         & (main_per.method == "plain")
                         & (main_per.linearity_degree == SENSITIVITY_DEGREE)
                         & (main_per.setting.isin(SENSITIVITY_SETTINGS))
                         & (main_per.N.isin(SENSITIVITY_NS))
                         & (main_per.sigma2_shape == 2.0)
                         & (main_per.sigma2_rate == 1.0)
                         & (main_per.estimand_type != "CATT")].copy()
    sensitivity_rows = nonmain_per[(nonmain_per.phase == "sensitivity")
                                   & (nonmain_per.spec == "structured")
                                   & (nonmain_per.method == "plain")
                                   & (nonmain_per.linearity_degree == SENSITIVITY_DEGREE)
                                   & (nonmain_per.setting.isin(SENSITIVITY_SETTINGS))
                                   & (nonmain_per.N.isin(SENSITIVITY_NS))
                                   & (nonmain_per.sigma2_shape == 2.0)
                                   & (nonmain_per.sigma2_rate.isin([0.1, 10.0]))
                                   & (nonmain_per.estimand_type != "CATT")].copy()
    require_columns(main_rows, TARGET_SENSITIVITY_KEY + ["true", "post_mean"],
                    "rate-1 main per-replication donor")
    require_columns(sensitivity_rows, TARGET_SENSITIVITY_KEY + ["true", "post_mean"],
                    "nonmain sensitivity per-replication")
    if main_rows.empty or sensitivity_rows.empty:
        raise ValueError("Sensitivity prior comparison rows are missing")
    # The two frozen collectors serialize integer design columns differently
    # (main writes degree as int, non-main may write the same values as float).
    # Normalize these key types before exact key-set and paired-row checks.
    for frame in (main_rows, sensitivity_rows):
        for col in ("N", "rep", "linearity_degree"):
            numeric = pd.to_numeric(frame[col], errors="raise")
            if not np.equal(numeric.to_numpy(float),
                            np.floor(numeric.to_numpy(float))).all():
                raise ValueError(f"Non-integer sensitivity key values in {col}")
            frame[col] = numeric.astype(int)

    # Confirm all six matched design cells have exactly reps 0..99 at each
    # requested prior rate and that the row-level metric support is identical.
    for setting, nobs in cells:
        donor = main_rows[(main_rows.setting == setting) & (main_rows.N == nobs)]
        if set(donor.rep.astype(int).unique()) != set(SENSITIVITY_REPS):
            raise ValueError(f"Rate-1 donor is not reps 0..99 for {setting}/N{nobs}")
        donor_keys = donor[TARGET_SENSITIVITY_KEY].drop(columns="rep").drop_duplicates()
        for rate in (0.1, 10.0):
            compare = sensitivity_rows[(sensitivity_rows.setting == setting)
                                       & (sensitivity_rows.N == nobs)
                                       & (sensitivity_rows.sigma2_rate == rate)]
            if set(compare.rep.astype(int).unique()) != set(SENSITIVITY_REPS):
                raise ValueError(f"Rate-{rate:g} sensitivity is not reps 0..99 for {setting}/N{nobs}")
            if compare.duplicated(TARGET_SENSITIVITY_KEY).any():
                raise ValueError(f"Duplicate sensitivity replication metric in {setting}/N{nobs}/rate{rate:g}")
            comp_keys = compare[TARGET_SENSITIVITY_KEY].drop(columns="rep").drop_duplicates()
            if len(donor_keys) != len(comp_keys) or not donor_keys.sort_values(
                    ["dgp", "setting", "N", "linearity_degree", "estimand_type", "estimand_id", "method"]).reset_index(drop=True).equals(
                    comp_keys.sort_values(["dgp", "setting", "N", "linearity_degree", "estimand_type", "estimand_id", "method"]).reset_index(drop=True)):
                raise ValueError(f"Scalar estimand targets differ at {setting}/N{nobs}/rate{rate:g}")
            paired = donor.merge(compare, on=TARGET_SENSITIVITY_KEY, suffixes=("_rate1", "_comp"),
                                 validate="one_to_one")
            if len(paired) != len(donor):
                raise ValueError(f"Sensitivity pairing lost rows at {setting}/N{nobs}/rate{rate:g}")
            if not np.array_equal(paired.true_rate1.to_numpy(float), paired.true_comp.to_numpy(float)):
                raise ValueError(f"DGP truths differ across prior rates at {setting}/N{nobs}/rate{rate:g}")

    # The full scalar summary table is taken from the frozen collectors, with
    # rate-1 rows sourced from the completed repaired-main overlay.
    main_metrics = main_scalar[(main_scalar.phase == "main")
                               & (main_scalar.spec == "structured")
                               & (main_scalar.method == "plain")
                               & (main_scalar.linearity_degree == SENSITIVITY_DEGREE)
                               & (main_scalar.setting.isin(SENSITIVITY_SETTINGS))
                               & (main_scalar.N.isin(SENSITIVITY_NS))
                               & (main_scalar.sigma2_shape == 2.0)
                               & (main_scalar.sigma2_rate == 1.0)].copy()
    nonmain_metrics = nonmain_scalar[(nonmain_scalar.phase == "sensitivity")
                                     & (nonmain_scalar.spec == "structured")
                                     & (nonmain_scalar.method == "plain")
                                     & (nonmain_scalar.linearity_degree == SENSITIVITY_DEGREE)
                                     & (nonmain_scalar.setting.isin(SENSITIVITY_SETTINGS))
                                     & (nonmain_scalar.N.isin(SENSITIVITY_NS))
                                     & (nonmain_scalar.sigma2_shape == 2.0)
                                     & (nonmain_scalar.sigma2_rate.isin([0.1, 10.0]))].copy()
    for label, frame in (("rate-1 scalar donor", main_metrics),
                         ("nonmain scalar prior sensitivity", nonmain_metrics)):
        if frame.empty or frame.n_reps.ne(100).any():
            raise ValueError(f"{label} must contain 100 replications for every row")
    main_metrics["result_origin"] = "repaired_main_20261014_rate1_donor"
    nonmain_metrics["result_origin"] = "repaired_nonmain_20261026_prior_sensitivity"
    prior_table = pd.concat([nonmain_metrics, main_metrics], ignore_index=True, sort=False)
    if prior_table.duplicated(["setting", "dgp", "N", "linearity_degree", "sigma2_rate",
                               "estimand_type", "estimand_id", "method"]).any():
        raise ValueError("Prior-sensitivity metric table has duplicate rate/target cells")
    prior_table = prior_table.sort_values(["setting", "N", "linearity_degree",
                                           "sigma2_rate", "estimand_type", "estimand_id"],
                                          kind="stable").reset_index(drop=True)

    paired_table = _paired_bootstrap(main_rows, sensitivity_rows)
    return prior_table, paired_table


def _paired_bootstrap(main_rows: pd.DataFrame,
                      sensitivity_rows: pd.DataFrame) -> pd.DataFrame:
    targets = (("ATT", "ATT"), ("GATT", "g=4_t=4"))
    output = []
    for setting in SENSITIVITY_SETTINGS:
        for nobs in SENSITIVITY_NS:
            donor_cell = main_rows[(main_rows.setting == setting) & (main_rows.N == nobs)]
            for estimand_type, estimand_id in targets:
                donor = donor_cell[(donor_cell.estimand_type == estimand_type)
                                   & (donor_cell.estimand_id == estimand_id)]
                donor = donor.sort_values("rep")
                if len(donor) != 100 or donor.rep.nunique() != 100:
                    raise ValueError(f"Missing rate-1 paired bootstrap target {setting}/N{nobs}/{estimand_type}/{estimand_id}")
                base_error = donor.post_mean.to_numpy(float) - donor.true.to_numpy(float)
                for rate in (0.1, 10.0):
                    compare = sensitivity_rows[(sensitivity_rows.setting == setting)
                                               & (sensitivity_rows.N == nobs)
                                               & (sensitivity_rows.sigma2_rate == rate)
                                               & (sensitivity_rows.estimand_type == estimand_type)
                                               & (sensitivity_rows.estimand_id == estimand_id)].sort_values("rep")
                    if len(compare) != 100 or compare.rep.nunique() != 100:
                        raise ValueError(f"Missing rate-{rate:g} paired bootstrap target {setting}/N{nobs}/{estimand_type}/{estimand_id}")
                    if not np.array_equal(donor.rep.to_numpy(int), compare.rep.to_numpy(int)):
                        raise ValueError(f"Replication IDs differ in paired bootstrap target {setting}/N{nobs}/{estimand_type}/{estimand_id}/rate{rate:g}")
                    if not np.array_equal(donor.true.to_numpy(float), compare.true.to_numpy(float)):
                        raise ValueError(f"Truth differs in paired bootstrap target {setting}/N{nobs}/{estimand_type}/{estimand_id}/rate{rate:g}")
                    comp_error = compare.post_mean.to_numpy(float) - compare.true.to_numpy(float)
                    base_rmse = float(np.sqrt(np.mean(base_error ** 2)))
                    comp_rmse = float(np.sqrt(np.mean(comp_error ** 2)))
                    if base_rmse <= 0:
                        raise ValueError(f"Zero rate-1 RMSE for {setting}/N{nobs}/{estimand_type}/{estimand_id}")
                    rmse_ratio = comp_rmse / base_rmse
                    bias_difference = float(np.mean(comp_error - base_error))
                    seed_text = (f"didbcf-prior-paired-bootstrap-v1|{BOOTSTRAP_BASE_SEED}|"
                                 f"{setting}|{nobs}|{estimand_type}|{estimand_id}|{rate:g}")
                    rng_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8],
                                              "big") % (2 ** 32)
                    rng = np.random.default_rng(rng_seed)
                    index = rng.integers(0, 100, size=(BOOTSTRAP_REPS, 100))
                    boot_base = np.sqrt(np.mean(base_error[index] ** 2, axis=1))
                    boot_comp = np.sqrt(np.mean(comp_error[index] ** 2, axis=1))
                    boot_ratio = boot_comp / boot_base
                    boot_bias = np.mean(comp_error[index] - base_error[index], axis=1)
                    output.append({
                        "setting": setting, "N": nobs,
                        "linearity_degree": SENSITIVITY_DEGREE,
                        "estimand_type": estimand_type, "estimand_id": estimand_id,
                        "method": "plain", "baseline_sigma2_rate": 1.0,
                        "comparison_sigma2_rate": rate, "n_reps": 100,
                        "rmse_rate_over_rate1": rmse_ratio,
                        "rmse_ratio_ci95_lower": float(np.quantile(boot_ratio, 0.025)),
                        "rmse_ratio_ci95_upper": float(np.quantile(boot_ratio, 0.975)),
                        "bias_difference_rate_minus_rate1": bias_difference,
                        "bias_difference_ci95_lower": float(np.quantile(boot_bias, 0.025)),
                        "bias_difference_ci95_upper": float(np.quantile(boot_bias, 0.975)),
                        "bootstrap_samples": BOOTSTRAP_REPS,
                        "bootstrap_seed": rng_seed,
                        "result_origin": "paired_replication_summary_bootstrap",
                    })
    result = pd.DataFrame(output)
    if len(result) != len(SENSITIVITY_SETTINGS) * len(SENSITIVITY_NS) * len(targets) * 2:
        raise ValueError("Paired bootstrap did not produce the complete expected grid")
    return result.sort_values(["setting", "N", "estimand_type", "estimand_id",
                               "comparison_sigma2_rate"], kind="stable").reset_index(drop=True)


def _replace_nonmain_rows(pub: pd.DataFrame,
                          scalar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = scalar[(scalar.phase.isin(["diagnostic", "pt_impact"]))
                     & scalar.N.eq(200) & scalar.setting.isin(PT_SETTINGS)
                     & scalar.linearity_degree.isin(PT_DEGREES)
                     & scalar.sigma2_shape.eq(2.0) & scalar.sigma2_rate.eq(1.0)].copy()
    if current.empty:
        raise ValueError("No current nonmain diagnostic/PT-impact metrics to overlay")
    diag = current[current.phase.eq("diagnostic")]
    impact = current[current.phase.eq("pt_impact")]
    if not diag.method.eq("pretrend").all():
        raise ValueError("Diagnostic phase contains unexpected method rows")
    if not impact.method.eq("plain").all():
        raise ValueError("PT-impact phase contains unexpected method rows")
    if diag.spec.ne("pretrend_unit").any() or impact.spec.ne("structured").any():
        raise ValueError("Unexpected model spec in current nonmain overlay")
    if len(diag.setting.unique()) != len(PT_SETTINGS) or len(impact.setting.unique()) != len(PT_SETTINGS):
        raise ValueError("Current nonmain overlay omits PT settings")

    base = pub.copy()
    pt_cells = pd.DataFrame([
        {"setting": setting, "dgp": "canonical", "N": 200,
         "linearity_degree": degree}
        for setting in PT_SETTINGS for degree in PT_DEGREES
    ])
    marked = base.merge(pt_cells.assign(_pt_cell=True), on=PT_CELL_KEY,
                        how="left", validate="many_to_one")
    pt_mask = marked._pt_cell.fillna(False)
    historic_pre = pt_mask & marked.method.eq("pretrend")
    historic_plain = pt_mask & marked.method.eq("plain")
    marked.loc[historic_pre, "method"] = "pretrend_historical"
    marked.loc[historic_plain, "method"] = "plain_historical"
    marked.loc[historic_pre | historic_plain, "result_origin"] = "historical"
    base = marked.drop(columns="_pt_cell")

    diag = diag.copy()
    diag["result_origin"] = "repaired_nonmain_pretrend_20261026"
    impact = impact.copy()
    impact["result_origin"] = "repaired_nonmain_ptimpact_20261026"
    additions = pd.concat([diag, impact], ignore_index=True, sort=False)
    out = pd.concat([base, additions], ignore_index=True, sort=False)
    require_columns(out, METRIC_KEY, "nonmain metrics overlay")
    duplicate = out.duplicated(METRIC_KEY, keep=False)
    if duplicate.any():
        raise ValueError("Duplicate metrics_all key after nonmain overlay:\n"
                         + out.loc[duplicate, METRIC_KEY].head(20).to_string(index=False))
    return out, additions


def _replace_surface_rows(pub: pd.DataFrame,
                          surface: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = surface[(surface.phase == "pt_impact") & surface.N.eq(200)
                      & surface.setting.isin(PT_SETTINGS)
                      & surface.linearity_degree.isin(PT_DEGREES)
                      & surface.sigma2_shape.eq(2.0)
                      & surface.sigma2_rate.eq(1.0)].copy()
    if len(current) != len(PT_SETTINGS) * len(PT_DEGREES):
        raise ValueError(f"Expected {len(PT_SETTINGS)*len(PT_DEGREES)} current PT surfaces, found {len(current)}")
    if not current.method.eq("plain").all():
        raise ValueError("Unexpected method in current PT surface metrics")
    base = pub.copy()
    pt_cells = pd.DataFrame([
        {"setting": setting, "dgp": "canonical", "N": 200,
         "linearity_degree": degree}
        for setting in PT_SETTINGS for degree in PT_DEGREES
    ])
    marked = base.merge(pt_cells.assign(_pt_cell=True), on=PT_CELL_KEY,
                        how="left", validate="many_to_one")
    historic_plain = marked._pt_cell.fillna(False) & marked.method.eq("plain")
    marked.loc[historic_plain, "method"] = "plain_historical"
    marked.loc[historic_plain, "result_origin"] = "historical"
    base = marked.drop(columns="_pt_cell")
    current["result_origin"] = "repaired_nonmain_ptimpact_20261026"
    out = pd.concat([base, current], ignore_index=True, sort=False)
    duplicate = out.duplicated(SURFACE_KEY, keep=False)
    if duplicate.any():
        raise ValueError("Duplicate surface_all key after nonmain PT update")
    return out, current


def _assert_main_unchanged(base: pd.DataFrame, merged: pd.DataFrame,
                           label: str, path: Path) -> None:
    require_columns(base, ["dgp", "setting", "N", "linearity_degree", "method"], label)
    if "result_origin" not in merged:
        raise ValueError(f"{label} overlay lost result_origin")
    target_origin = "repaired_main_20261014"
    subset = merged[merged.result_origin.eq(target_origin)].copy()
    columns = list(base.columns)
    require_columns(subset, columns, f"merged {label}")
    key = [c for c in ["dgp", "setting", "N", "linearity_degree", "spec",
                       "estimand_type", "estimand_id", "method", "phase",
                       "budget_key", "sigma2_shape", "sigma2_rate"] if c in columns]
    expected = base.sort_values(key, kind="stable").reset_index(drop=True)
    actual = subset[columns].sort_values(key, kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False,
                                      check_exact=True, check_like=False)
    except AssertionError as error:
        raise ValueError(f"{label} main rows changed in new overlay: {error}") from error


def _check_surface_main(base: pd.DataFrame, merged: pd.DataFrame) -> None:
    # The main surface rows carry the same origin label as the scalar overlay.
    _assert_main_unchanged(base, merged, "main surface", MAIN / "main_surface_metrics.csv")


def _hash_map(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.relative_to(REPO)): sha256(path) for path in paths}


def _print_summary(pretrend: pd.DataFrame, metrics: pd.DataFrame,
                   paired: pd.DataFrame) -> None:
    raw = pretrend[(pretrend.method == "pretrend")]
    show = raw[raw.setting.isin(["PT_hold", "PT_conditional", "PT_violation_g05",
                                 "PT_violation_g40", "PT_violation_a20",
                                 "PT_violation_het10", "PT_violation_het20",
                                 "PT_violation_het40"])]
    print("pretrend repaired (slope size/power; subgroup column is X1 contrast):")
    print(show[["setting", "linearity_degree", "slope_detect", "subgroup_detect",
                "anyk_detect", "slope_n", "subgroup_n"]]
          .sort_values(["linearity_degree", "setting"])
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    impact = metrics[(metrics.result_origin == "repaired_nonmain_ptimpact_20261026")
                     & metrics.setting.eq("PT_violation_g05")
                     & metrics.estimand_type.eq("ATT")
                     & metrics.estimand_id.eq("ATT")]
    if not impact.empty:
        print("PT_violation_g05 repaired PT-impact ATT bias:")
        print(impact[["linearity_degree", "bias", "rmse", "n_reps"]]
              .sort_values("linearity_degree").to_string(index=False))
    print("paired prior-sensitivity RMSE-ratio ranges (rates 0.1/10 vs 1):")
    print(paired.groupby(["estimand_type", "comparison_sigma2_rate"])
          .rmse_rate_over_rate1.agg(["min", "max"]).to_string())


def main() -> None:
    manifest, package, status, collected_provenance = verify_inputs()
    per_path = COLLECTED / "per_replication.csv"
    scalar_path = COLLECTED / "scalar_metrics.csv"
    surface_path = COLLECTED / "surface_metrics.csv"
    per = read_csv(per_path)
    scalar = read_csv(scalar_path)
    surface = read_csv(surface_path)
    _check_nonmain_shapes(manifest, per, scalar, surface)

    main_metrics_path = MAIN / "metrics_all.csv"
    main_surface_path = MAIN / "surface_all.csv"
    main_scalar_path = MAIN / "main_scalar_metrics.csv"
    main_surface_metrics_path = MAIN / "main_surface_metrics.csv"
    main_per_path = MAIN / "main_per_replication.csv"
    main_metrics = read_csv(main_metrics_path)
    main_surface = read_csv(main_surface_path)
    main_scalar = read_csv(main_scalar_path)
    main_surface_metrics = read_csv(main_surface_metrics_path)
    main_per = read_csv(main_per_path)

    pretrend = _pretrend_summary(per, scalar, main_metrics)
    profiles = _pretrend_profiles(per)
    prior_sensitivity, paired = _validate_prior_sensitivity(
        main_per, per, main_scalar, scalar
    )
    metrics_overlay, new_scalar_rows = _replace_nonmain_rows(main_metrics, scalar)
    surface_overlay, new_surface_rows = _replace_surface_rows(main_surface, surface)

    _assert_main_unchanged(main_scalar, metrics_overlay, "main scalar", main_scalar_path)
    _check_surface_main(main_surface_metrics, surface_overlay)

    OUT.mkdir(parents=True, exist_ok=True)
    # A previous PASS record must not survive a partial rebuild. Remove it
    # only after all in-memory validation has passed, then write fresh hashes
    # and restore provenance.json as the final output below.
    (OUT / "provenance.json").unlink(missing_ok=True)
    write_csv(metrics_overlay, OUT / "metrics_all.csv")
    write_csv(surface_overlay, OUT / "surface_all.csv")
    write_csv(scalar, OUT / "nonmain_scalar_metrics.csv")
    write_csv(surface, OUT / "nonmain_surface_metrics.csv")
    write_csv(pretrend, OUT / "pretrend_summary.csv")
    write_csv(profiles, OUT / "pretrend_profiles.csv")
    write_csv(prior_sensitivity, OUT / "prior_sensitivity.csv")
    write_csv(paired, OUT / "prior_sensitivity_paired.csv")
    shutil.copyfile(per_path, OUT / "nonmain_per_replication.csv")
    for name in ("sqrtn_all.csv", "coverage_grid.csv", "main_scalar_metrics.csv",
                 "main_surface_metrics.csv", "main_per_replication.csv"):
        shutil.copyfile(MAIN / name, OUT / name)

    # These must be byte-identical copies of the prior main overlay inputs.
    for name in ("sqrtn_all.csv", "coverage_grid.csv", "main_scalar_metrics.csv",
                 "main_surface_metrics.csv", "main_per_replication.csv"):
        if sha256(OUT / name) != sha256(MAIN / name):
            raise ValueError(f"Prior main file changed while copying: {name}")
    _assert_main_unchanged(main_scalar, read_csv(OUT / "metrics_all.csv"),
                           "main scalar", main_scalar_path)

    # A concise machine-readable run note, without any posterior draw arrays.
    note = [
        "# Repaired non-main results, 2026-10-26",
        "",
        "This directory overlays the completed non-main metrics on the immutable",
        "2026-10-14 repaired-main aggregate. It contains no posterior arrays and",
        "does not run sampling.",
        "",
        f"Completed jobs: {status['completed_total']} / {status['expected_total']};",
        "all phase counts match the frozen manifest.",
        "",
        "The new pretrend diagnostic rows use the repaired raw `pretrend` fit.",
        "Historical TWFE and other benchmark rows remain labelled historical.",
        "The reference-fold benchmark comes from the archived completion metrics.",
        "The `PT_violation_het20`, degree-3 historical comparator has n=150; paired",
        "comparisons should use the matching replication intersection.",
        "",
        "Prior sensitivity rates 0.1 and 10 are paired to the repaired-main rate-1",
        "donor by seed/index and identical truth. No byte-level panel hashes are",
        "available in these summary tables, so this pairing does not establish",
        "binary dataset identity.",
        "",
    ]
    (OUT / "RESULTS_REPORT.md").write_text("\n".join(note))

    input_paths = [
        Path(__file__),
        PACKAGE / "manifest.json", PACKAGE / "package.json",
        COLLECTED / "status.json", COLLECTED / "provenance.json",
        per_path, scalar_path, surface_path,
        MAIN / "provenance.json", main_metrics_path, main_surface_path,
        main_scalar_path, main_surface_metrics_path, main_per_path,
        MAIN / "sqrtn_all.csv", MAIN / "coverage_grid.csv",
        COMPLETION / "combined_per_replication.csv",
        COMPLETION / "metrics_mean_median.csv",
    ]
    output_paths = [path for path in sorted(OUT.iterdir()) if path.is_file()]
    provenance = {
        "status": "PASS",
        "builder": "Results/analysis/update_repaired_nonmain.py",
        "completed_nonmain_jobs": 9227,
        "expected_nonmain_jobs": 9227,
        "phase_completed": EXPECTED_PHASES,
        "frozen_manifest_hash": manifest["manifest_hash"],
        "package_hash": package["package_hash"],
        "collector_package_hash": collected_provenance["package_hash"],
        "source_hashes": _hash_map(input_paths),
        "outputs": {path.name: sha256(path) for path in output_paths},
        "expected_study_cells": {
            "pretrend": len(PT_SETTINGS) * len(PT_DEGREES),
            "prior_sensitivity_cells_per_rate": len(SENSITIVITY_SETTINGS) * len(SENSITIVITY_NS),
            "prior_sensitivity_rates": list(SENSITIVITY_RATES),
        },
        "diagnostic_filtering": False,
        "performance_filtering": False,
        "new_sampling_calls": 0,
        "posterior_arrays_read": False,
        "main_scalar_values_unchanged": True,
        "prior_main_files_preserved_byte_for_byte": [
            "sqrtn_all.csv", "coverage_grid.csv", "main_scalar_metrics.csv",
            "main_surface_metrics.csv", "main_per_replication.csv",
        ],
        "historical_policy": {
            "pretrend_method_label": "pretrend_historical",
            "pt_impact_method_label": "plain_historical",
            "twfe_and_external_benchmarks": "retained with historical origin",
            "reference_fold_source": "Theory_Calibration/results/aggregated_completion",
        },
        "pairing_note": (
            "Prior sensitivity is paired by common seed/index, design config, and exact truth. "
            "These summary files do not include dataset byte hashes, so pairing does not "
            "establish binary panel identity."
        ),
        "metrics_rows_added": int(len(new_scalar_rows)),
        "surface_rows_added": int(len(new_surface_rows)),
    }
    write_json_last(provenance, OUT / "provenance.json")
    _print_summary(pretrend, metrics_overlay, paired)
    print(f"PASS: wrote repaired non-main overlay to {OUT}")


if __name__ == "__main__":
    main()
