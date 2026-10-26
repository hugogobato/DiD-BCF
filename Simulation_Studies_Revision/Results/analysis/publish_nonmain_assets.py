#!/usr/bin/env python3
"""Publish validated non-main simulation tables and figures.

The script reads the repaired overlay from ``Results/aggregated_repaired_20261026``
by default. It requires a PASS provenance record covering the full 9,227-unit
collection before creating any outputs. Generated paper assets are mirrored in
``Results/tables`` and ``Results/figures``.

Required method labels in ``pretrend_summary.csv`` must identify repaired raw
pretrend estimates, historical reference-fold pretrend estimates, and the
historical TWFE placebo. The label parser accepts common spelling variants.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_FULL_COUNT = 9227
EXPECTED_REPS = 200
EXPECTED_PRIOR_REPS = 100
EXPECTED_PRIOR_ROWS = 18

SUMMARY_COLUMNS = {
    "setting",
    "linearity_degree",
    "method",
    "slope_detect",
    "slope_count",
    "slope_n",
    "slope_wilson95_lower",
    "slope_wilson95_upper",
    "subgroup_detect",
    "subgroup_n",
    "anyk_detect",
    "anyk_n",
    "mean_true",
    "slope_bias",
    "slope_cover95",
}
PROFILE_COLUMNS = {
    "setting",
    "linearity_degree",
    "method",
    "estimand_id",
    "mean_true",
    "bias",
    "empirical_sd_error",
}
PRIOR_COLUMNS = {
    "setting",
    "N",
    "linearity_degree",
    "sigma2_rate",
    "estimand_type",
    "estimand_id",
    "n_reps",
    "bias",
    "rmse",
    "mae",
    "cover95",
    "len95",
}

SCENARIO_ORDER = (
    "hold",
    "conditional",
    "g005",
    "g010",
    "g020",
    "g040",
    "a20",
    "het010",
    "het020",
    "het040",
)
SCENARIO_LABELS = {
    "hold": ("Parallel trends hold", "None"),
    "conditional": ("Conditional PT only", "Unconditional-only difference"),
    "g005": ("Group slope 0.05", "Group trend, 0.05"),
    "g010": ("Group slope 0.10", "Group trend, 0.10"),
    "g020": ("Group slope 0.20", "Group trend, 0.20"),
    "g040": ("Group slope 0.40", "Group trend, 0.40"),
    "a20": ("Unit-effect slope 0.20", "Violation through unit effect"),
    "het010": ("Cancelling heterogeneity 0.10", "Heterogeneous, aggregate cancels"),
    "het020": ("Cancelling heterogeneity 0.20", "Heterogeneous, aggregate cancels"),
    "het040": ("Cancelling heterogeneity 0.40", "Heterogeneous, aggregate cancels"),
}


class AssetError(RuntimeError):
    """Raised when inputs do not support a reproducible publication build."""


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    results_dir = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=results_dir / "aggregated_repaired_20261026",
        help="Overlay directory containing provenance.json and the three CSV files.",
    )
    return parser.parse_args()


def _normal(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _walk_pairs(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + (str(key),)
            yield path, child
            yield from _walk_pairs(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_pairs(child, prefix + (str(index),))


def _is_numeric_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_provenance(path: Path) -> None:
    if not path.is_file():
        raise AssetError(f"Missing required provenance file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"Could not read provenance JSON at {path}: {exc}") from exc
    if not isinstance(data, dict) or str(data.get("status", "")).strip().upper() != "PASS":
        raise AssetError("provenance.json must have top-level status 'PASS'.")

    count_keys = {
        "full",
        "fullcount",
        "fullrows",
        "fullrecords",
        "fullreplications",
        "nfull",
        "nrows",
        "rowcount",
        "recordcount",
        "records",
        "nrecords",
        "collectedrows",
        "collectedrecords",
        "collectedunits",
        "ncollected",
        "completedcount",
        "completedjobs",
        "completednonmainjobs",
        "jobs",
        "replications",
        "nreplications",
        "units",
        "nunits",
        "rows",
        "totalcount",
        "totalrows",
        "totalrecords",
        "expectedtotal",
        "expectedrows",
        "expectedrecords",
        "validatedrows",
        "returnedrows",
        "totalunits",
        "totalreplications",
        "n",
    }
    full_count = False
    total_count = False
    full_scope = False
    for path_parts, value in _walk_pairs(data):
        key = _normal(path_parts[-1]) if path_parts else ""
        if _is_numeric_count(value) and int(value) == EXPECTED_FULL_COUNT:
            if "full" in key or key == "completednonmainjobs":
                full_count = True
            if key in count_keys:
                total_count = True
        if value is True and "full" in key:
            full_scope = True
        if isinstance(value, str) and _normal(value) == "full":
            if key in {"scope", "validation_scope", "collection_scope", "coverage", "completeness"}:
                full_scope = True
        if isinstance(value, str) and re.search(r"\bfull\b.{0,32}\b9227\b|\b9227\b.{0,32}\bfull\b", value, re.I):
            full_count = True

    if not (full_count or (full_scope and total_count)):
        raise AssetError(
            "provenance.json must establish full coverage of 9,227 collected units "
            "(for example, a full-count field equal to 9227 or scope='full' with a "
            "total row/record count of 9227)."
        )


def read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_provenance(input_dir / "provenance.json")
    paths = {
        "pretrend_summary": input_dir / "pretrend_summary.csv",
        "pretrend_profiles": input_dir / "pretrend_profiles.csv",
        "prior_sensitivity": input_dir / "prior_sensitivity.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise AssetError(f"Missing {label} CSV: {path}")
    try:
        summary = pd.read_csv(paths["pretrend_summary"])
        profiles = pd.read_csv(paths["pretrend_profiles"])
        prior = pd.read_csv(paths["prior_sensitivity"])
    except Exception as exc:  # pandas reports parser, encoding, and I/O failures here.
        raise AssetError(f"Could not read one or more input CSV files: {exc}") from exc

    _require_columns(summary, SUMMARY_COLUMNS, "pretrend_summary.csv")
    _require_columns(profiles, PROFILE_COLUMNS, "pretrend_profiles.csv")
    _require_columns(prior, PRIOR_COLUMNS, "prior_sensitivity.csv")
    return summary, profiles, prior


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssetError(f"{name} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise AssetError(f"{name} has no rows.")


def method_kind(value: Any) -> str:
    name = _normal(value)
    if "twfe" in name or name in {"placebo", "marginalplacebo", "twfeeventstudy"}:
        return "twfe"
    if "referencefold" in name or "reffold" in name:
        return "reference_fold"
    if "repaired" in name or "rawpretrend" in name or name in {"raw", "pretrend", "didbcf"}:
        return "repaired"
    return "other"


def scenario_kind(value: Any) -> str:
    name = _normal(value)
    if "conditional" in name:
        return "conditional"
    if "hold" in name or "noviolation" in name or "ptnull" in name:
        return "hold"
    if "het" in name or "heterogeneous" in name or "cancelling" in name:
        for code, number in (("010", "010"), ("020", "020"), ("040", "040")):
            if code in name or str(int(number)) in name or f"0{int(number) // 10}" in name:
                return f"het{code}"
        # Decimal setting names often normalize to 01, 02, or 04.
        if re.search(r"het.*(?:01|010|0p1)", name):
            return "het010"
        if re.search(r"het.*(?:02|020|0p2)", name):
            return "het020"
        if re.search(r"het.*(?:04|040|0p4)", name):
            return "het040"
    if "a20" in name or "unobserv" in name or "unobservable" in name or "uniteffect" in name or "alpha" in name:
        return "a20"
    for code, aliases in {
        "g005": ("g005", "g05", "slope005", "slope05"),
        "g010": ("g010", "g10", "slope010", "slope10"),
        "g020": ("g020", "g20", "slope020", "slope20"),
        "g040": ("g040", "g40", "slope040", "slope40"),
    }.items():
        if any(alias in name for alias in aliases):
            return code
    raise AssetError(f"Unrecognized pretrend setting label {value!r}.")


def error_kind(value: Any) -> str:
    name = _normal(value)
    if "ar1" in name or "serial" in name:
        return "ar1"
    if "iid" in name or "independent" in name or "ordinary" in name or "b2sweep" in name:
        return "iid"
    raise AssetError(f"Unrecognized prior-sensitivity error setting {value!r}; expected iid or AR(1).")


def _float_series(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
    allow_nan: Iterable[str] = (),
) -> None:
    allowed = set(allow_nan)
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="coerce")
        except Exception as exc:
            raise AssetError(f"{name}.{column} is not numeric: {exc}") from exc
        missing = values.isna()
        if missing.any() and column not in allowed:
            bad = frame.loc[values.isna(), column].head(3).tolist()
            raise AssetError(f"{name}.{column} contains missing or nonnumeric values, examples: {bad}")
        frame[column] = values.astype(float)


def wilson_interval(count: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0 or count < 0 or count > n:
        raise AssetError(f"Invalid binomial count: count={count}, n={n}.")
    p = count / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def prepare_pretrend(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    _float_series(
        frame,
        (
            "linearity_degree",
            "slope_detect",
            "slope_count",
            "slope_n",
            "slope_wilson95_lower",
            "slope_wilson95_upper",
            "subgroup_detect",
            "subgroup_n",
            "anyk_detect",
            "anyk_n",
            "mean_true",
            "slope_bias",
            "slope_cover95",
        ),
        "pretrend_summary",
        allow_nan=(
            "slope_wilson95_lower",
            "slope_wilson95_upper",
            "subgroup_detect",
            "subgroup_n",
            "anyk_detect",
            "anyk_n",
            "slope_bias",
            "slope_cover95",
        ),
    )
    frame["method_kind"] = frame["method"].map(method_kind)
    frame["scenario_kind"] = frame["setting"].map(scenario_kind)
    frame = frame[frame["linearity_degree"].isin((1.0, 3.0))].copy()
    frame = frame[frame["method_kind"].isin(("repaired", "reference_fold", "twfe"))].copy()
    if frame.empty:
        raise AssetError("pretrend_summary.csv has no degree-1 or degree-3 rows.")
    if set(frame["scenario_kind"].unique()) != set(SCENARIO_ORDER):
        missing = sorted(set(SCENARIO_ORDER).difference(frame["scenario_kind"].unique()))
        extra = sorted(set(frame["scenario_kind"].unique()).difference(SCENARIO_ORDER))
        raise AssetError(f"Pretrend settings do not match the ten expected designs; missing={missing}, extra={extra}.")
    if (frame["slope_n"] <= 0).any() or (frame["slope_count"] < 0).any() or (frame["slope_count"] > frame["slope_n"]).any():
        raise AssetError("pretrend_summary has invalid slope_count/slope_n values.")
    rates = frame["slope_count"] / frame["slope_n"]
    if ((rates - frame["slope_detect"]).abs() > 0.0051).any():
        raise AssetError("slope_detect must be a rejection frequency consistent with slope_count/slope_n.")
    if (frame["slope_detect"] < 0).any() or (frame["slope_detect"] > 1).any():
        raise AssetError("slope_detect values must lie in [0, 1].")
    if ((frame["slope_wilson95_lower"] < 0) | (frame["slope_wilson95_upper"] > 1) | (frame["slope_wilson95_lower"] > frame["slope_wilson95_upper"])).any():
        raise AssetError("Invalid Wilson interval columns in pretrend_summary.csv.")

    for degree in (1.0, 3.0):
        subset = frame[frame["linearity_degree"] == degree]
        if subset.empty:
            raise AssetError(f"pretrend_summary.csv is missing degree {int(degree)}.")
        for scenario in SCENARIO_ORDER:
            rows = subset[subset["scenario_kind"] == scenario]
            for required_method in ("repaired", "reference_fold", "twfe"):
                selected = rows[rows["method_kind"] == required_method]
                if len(selected) != 1:
                    raise AssetError(
                        f"Expected one {required_method} row for degree {int(degree)}, "
                        f"setting {scenario}; found {len(selected)}."
                    )
            repaired = rows[rows["method_kind"] == "repaired"].iloc[0]
            refold = rows[rows["method_kind"] == "reference_fold"].iloc[0]
            if int(repaired["slope_n"]) != EXPECTED_REPS or int(refold["slope_n"]) != EXPECTED_REPS:
                raise AssetError(
                    f"Repaired and historical reference-fold diagnostics must each use 200 "
                    f"replications: degree {int(degree)}, setting {scenario}."
                )
            twfe = rows[rows["method_kind"] == "twfe"].iloc[0]
            expected_twfe_n = 150 if degree == 3.0 and scenario == "het020" else EXPECTED_REPS
            if int(twfe["slope_n"]) != expected_twfe_n:
                raise AssetError(
                    f"Historical TWFE count for degree {int(degree)}, setting {scenario} "
                    f"is {int(twfe['slope_n'])}, expected {expected_twfe_n}."
                )
            if pd.isna(repaired["slope_wilson95_lower"]) or pd.isna(repaired["slope_wilson95_upper"]):
                raise AssetError(f"Repaired Wilson limits are required for degree {int(degree)}, setting {scenario}.")
    return frame


def prepare_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    frame = profiles.copy()
    _float_series(
        frame,
        ("linearity_degree", "mean_true", "bias", "empirical_sd_error"),
        "pretrend_profiles",
    )
    frame["method_kind"] = frame["method"].map(method_kind)
    frame["scenario_kind"] = frame["setting"].map(scenario_kind)
    frame = frame[frame["linearity_degree"].isin((1.0, 3.0))].copy()
    frame = frame[frame["method_kind"].isin(("repaired", "reference_fold"))].copy()
    if (frame["empirical_sd_error"] < 0).any():
        raise AssetError("empirical_sd_error must be nonnegative.")
    if frame.empty:
        raise AssetError("pretrend_profiles.csv has no repaired or reference-fold lead profiles.")
    frame["estimate"] = frame["mean_true"] + frame["bias"]
    return frame


def _is_att(row: pd.Series) -> bool:
    kind = _normal(row["estimand_type"])
    ident = _normal(row["estimand_id"])
    aliases = {"att", "pooledatt", "averageatt", "overallatt", "treatmentatt"}
    return kind in aliases or ident in aliases


def prepare_prior(prior: pd.DataFrame) -> pd.DataFrame:
    frame = prior.copy()
    _float_series(
        frame,
        ("N", "linearity_degree", "sigma2_rate", "n_reps", "bias", "rmse", "mae", "cover95", "len95"),
        "prior_sensitivity",
    )
    frame["error_kind"] = frame["setting"].map(error_kind)
    frame = frame[frame.apply(_is_att, axis=1)].copy()
    frame = frame[np.isclose(frame["linearity_degree"], 1.0)].copy()
    if len(frame) != EXPECTED_PRIOR_ROWS:
        raise AssetError(
            f"Expected exactly 18 degree-1 ATT rows (2 error settings x N x 3 rates); found {len(frame)}."
        )
    if frame.duplicated(["error_kind", "N", "sigma2_rate"]).any():
        raise AssetError("Prior ATT rows are not unique by error setting, N, and rate.")
    if set(frame["error_kind"].unique()) != {"iid", "ar1"}:
        raise AssetError("Prior ATT table must include exactly iid and AR(1) settings.")
    rates = sorted(frame["sigma2_rate"].unique().tolist())
    if len(rates) != 3:
        raise AssetError(f"Expected three prior rates; found {rates}.")
    if not np.isclose(rates[1], 1.0):
        raise AssetError(f"The middle prior rate must be 1; found rates {rates}.")
    if not (frame["n_reps"] == EXPECTED_PRIOR_REPS).all():
        raise AssetError("Every prior-sensitivity ATT row must use 100 replications.")
    if ((frame["cover95"] < 0) | (frame["cover95"] > 1)).any():
        raise AssetError("cover95 values must lie in [0, 1].")
    if (frame["len95"] < 0).any() or (frame["rmse"] < 0).any() or (frame["mae"] < 0).any():
        raise AssetError("Prior sensitivity RMSE, MAE, and interval length must be nonnegative.")
    for error in ("iid", "ar1"):
        subset = frame[frame["error_kind"] == error]
        ns = sorted(subset["N"].unique().tolist())
        if len(ns) != 3 or any((subset["N"] == n).sum() != 3 for n in ns):
            raise AssetError(f"Prior ATT rows for {error} must have three N values and all three rates per N.")
    return frame.sort_values(["error_kind", "N", "sigma2_rate"]).reset_index(drop=True)


def fmt_num(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def fmt_rate(row: pd.Series, column: str) -> str:
    value = row[column]
    return "--" if pd.isna(value) else fmt_num(value, 3)


def get_row(frame: pd.DataFrame, degree: float, scenario: str, method: str) -> pd.Series:
    rows = frame[
        (frame["linearity_degree"] == degree)
        & (frame["scenario_kind"] == scenario)
        & (frame["method_kind"] == method)
    ]
    if len(rows) != 1:
        raise AssetError(f"Expected one {method} row for degree {int(degree)}, {scenario}; found {len(rows)}.")
    return rows.iloc[0]


def make_supplement_table(frame: pd.DataFrame, degree: float) -> str:
    lines = [
        r"\begin{tabular}{lllr rrr r r r}",
        r"\toprule",
        r"Scenario & Method & Violation & True slope & Slope & Contrast & Any-$k$ & Bias & Coverage & TWFE \\",
        r"\midrule",
    ]
    for scenario in SCENARIO_ORDER:
        repaired = get_row(frame, degree, scenario, "repaired")
        refold = get_row(frame, degree, scenario, "reference_fold")
        twfe = get_row(frame, degree, scenario, "twfe")
        label, violation = SCENARIO_LABELS[scenario]
        true_slope = fmt_num(repaired["mean_true"], 3)
        twfe_rate = fmt_num(twfe["slope_count"] / twfe["slope_n"], 3)
        for index, (row, method_label) in enumerate(
            ((repaired, "Repaired"), (refold, "Ref. fold (hist.)"))
        ):
            shown_label = label if index == 0 else ""
            shown_violation = violation if index == 0 else ""
            cells = (
                shown_label,
                method_label,
                shown_violation,
                true_slope if index == 0 else "",
                fmt_num(row["slope_count"] / row["slope_n"], 3),
                fmt_rate(row, "subgroup_detect"),
                fmt_rate(row, "anyk_detect"),
                fmt_rate(row, "slope_bias"),
                fmt_rate(row, "slope_cover95"),
                twfe_rate if index == 0 else "",
            )
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines.pop()  # The final rule follows the last scenario without extra spacing.
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if degree == 3.0:
        lines.insert(
            -2,
            r"\multicolumn{10}{l}{\footnotesize Historical TWFE uses 150 replications for cancelling heterogeneity 0.20 and 200 otherwise.} \\",
        )
    return "\n".join(lines) + "\n"


def make_main_pretrend_table(frame: pd.DataFrame) -> str:
    rows = [
        ("hold", "PT holds"),
        ("conditional", "Conditional PT only"),
        ("g005", "Group slope 0.05"),
        ("g010", "Group slope 0.10"),
        ("g020", "Group slope 0.20"),
        ("g040", "Group slope 0.40"),
        ("a20", "Unit-effect slope 0.20"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Pre-trend rejection frequencies at $d=1$, based on 200 replications per setting. The TWFE placebo uses a separate, unpaired replication stream.}",
        r"\label{tab:pretrend_main}",
        r"\spacingset{1}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Scenario & True slope & DiD-BCF & TWFE placebo \\",
        r"\midrule",
    ]
    for scenario, label in rows:
        raw = get_row(frame, 1.0, scenario, "repaired")
        twfe = get_row(frame, 1.0, scenario, "twfe")
        lines.append(
            " & ".join(
                (
                    label,
                    fmt_num(raw["mean_true"], 3),
                    fmt_num(raw["slope_count"] / raw["slope_n"], 3),
                    fmt_num(twfe["slope_count"] / twfe["slope_n"], 3),
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        )
    )
    return "\n".join(lines) + "\n"


def make_prior_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Prior-rate sensitivity of pooled ATT estimates. Each cell uses 100 replications; the rate-1 results are from the completed main grid.}",
        r"\label{tab:prior_sensitivity}",
        r"\begin{tabular}{lrrr rrrr}",
        r"\toprule",
        r"Error process & $N$ & Prior rate & Bias & RMSE & MAE & 95\% coverage & Mean 95\% length \\",
        r"\midrule",
    ]
    error_labels = {"iid": "Independent", "ar1": "AR(1)"}
    for _, row in frame.iterrows():
        lines.append(
            " & ".join(
                (
                    error_labels[row["error_kind"]],
                    str(int(row["N"])),
                    fmt_num(row["sigma2_rate"], 1),
                    fmt_num(row["bias"], 3),
                    fmt_num(row["rmse"], 3),
                    fmt_num(row["mae"], 3),
                    fmt_num(row["cover95"], 3),
                    fmt_num(row["len95"], 3),
                )
            )
            + r" \\"
        )
        if row["sigma2_rate"] == sorted(frame["sigma2_rate"].unique())[1]:
            lines.append(r"\addlinespace[1pt]")
    if lines[-1] == r"\addlinespace[1pt]":
        lines.pop()
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def _parse_event_time(value: Any) -> tuple[float, str]:
    text = str(value).strip()
    normalized = text.lower().replace("minus", "-")
    normalized = re.sub(r"(^|[_\s])m(?=\d)", r"\1-", normalized)
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if match:
        try:
            return float(match.group(0)), text
        except ValueError:
            pass
    return float("inf"), text


def _profile_rows(frame: pd.DataFrame, degree: float, scenario: str) -> pd.DataFrame:
    selected = frame[
        (frame["linearity_degree"] == degree)
        & (frame["scenario_kind"] == scenario)
        & frame["method_kind"].isin(("repaired", "reference_fold"))
    ].copy()
    if selected.empty:
        return selected
    selected["x_numeric"], selected["x_label"] = zip(*selected["estimand_id"].map(_parse_event_time))
    selected = selected.sort_values(["x_numeric", "estimand_id", "method_kind"])
    if selected["estimand_id"].nunique() < 2:
        raise AssetError(f"Lead profile for degree {int(degree)}, {scenario} needs at least two leads.")
    methods = set(selected["method_kind"].unique())
    if methods != {"repaired", "reference_fold"}:
        raise AssetError(f"Lead profile for degree {int(degree)}, {scenario} must include repaired and reference-fold rows.")
    return selected


def _plot_rejection_panel(ax: plt.Axes, frame: pd.DataFrame, degree: float) -> None:
    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    raw_rates, raw_low, raw_high = [], [], []
    twfe_rates, twfe_low, twfe_high = [], [], []
    for scenario in SCENARIO_ORDER:
        raw = get_row(frame, degree, scenario, "repaired")
        twfe = get_row(frame, degree, scenario, "twfe")
        raw_p = raw["slope_count"] / raw["slope_n"]
        raw_rates.append(raw_p)
        # Use the validated, exported Wilson limits for the repaired diagnostic.
        raw_low.append(max(0.0, raw_p - raw["slope_wilson95_lower"]))
        raw_high.append(max(0.0, raw["slope_wilson95_upper"] - raw_p))
        twfe_p = twfe["slope_count"] / twfe["slope_n"]
        low, high = wilson_interval(twfe["slope_count"], twfe["slope_n"])
        twfe_rates.append(twfe_p)
        twfe_low.append(twfe_p - low)
        twfe_high.append(high - twfe_p)
    ax.axvspan(-0.45, 1.45, color="#eef1f4", zorder=0)
    ax.errorbar(
        x - 0.08,
        raw_rates,
        yerr=np.array([raw_low, raw_high]),
        fmt="o-",
        capsize=2.5,
        color="#1f5a94",
        label="Repaired diagnostic",
        linewidth=1.5,
        markersize=4,
    )
    ax.errorbar(
        x + 0.08,
        twfe_rates,
        yerr=np.array([twfe_low, twfe_high]),
        fmt="s--",
        capsize=2.5,
        color="#a34a2a",
        label="Historical TWFE placebo",
        linewidth=1.3,
        markersize=3.5,
    )
    short_labels = {
        "hold": "PT\nholds",
        "conditional": "Cond.\nonly",
        "g005": "Group\n.05",
        "g010": "Group\n.10",
        "g020": "Group\n.20",
        "g040": "Group\n.40",
        "a20": "Unit\neffect",
        "het010": "Het.\n.10",
        "het020": "Het.\n.20",
        "het040": "Het.\n.40",
    }
    ax.set_xticks(x, [short_labels[key] for key in SCENARIO_ORDER])
    ax.set_ylim(-0.035, 1.04)
    ax.set_ylabel("Rejection frequency")
    ax.set_title("Overall slope test by design")
    ax.grid(axis="y", color="#d9dde2", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.5,
        -0.26,
        "Shading marks designs where conditional parallel trends holds. Bars are 95% Wilson intervals.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
    )


def _plot_profiles(ax: plt.Axes, frame: pd.DataFrame, degree: float) -> None:
    profiles = pd.concat(
        (
            _profile_rows(frame, degree, "hold"),
            _profile_rows(frame, degree, "g020"),
        ),
        ignore_index=True,
    )
    palette = {"repaired": "#1f5a94", "reference_fold": "#a34a2a"}
    method_label = {"repaired": "Repaired", "reference_fold": "Reference fold (historical)"}
    scenario_label = {"hold": "PT holds", "g020": "Group slope 0.20"}
    line_style = {"hold": "-", "g020": "--"}
    for scenario in ("hold", "g020"):
        for method in ("repaired", "reference_fold"):
            rows = profiles[
                (profiles["scenario_kind"] == scenario)
                & (profiles["method_kind"] == method)
            ].sort_values("x_numeric")
            if rows.empty:
                raise AssetError(f"Missing {method} lead profile for degree {int(degree)}, {scenario}.")
            # These error bars are empirical SDs of replication errors, not confidence intervals.
            ax.errorbar(
                rows["x_numeric"],
                rows["estimate"],
                yerr=rows["empirical_sd_error"],
                fmt="o" if scenario == "hold" else "s",
                linestyle=line_style[scenario],
                color=palette[method],
                capsize=2,
                linewidth=1.25,
                markersize=3.5,
                label=f"{method_label[method]}, {scenario_label[scenario]}",
            )
    # True lead profiles are shared across the two estimators for a given design.
    for scenario, color in (("hold", "#555555"), ("g020", "#777777")):
        rows = profiles[profiles["scenario_kind"] == scenario].sort_values("x_numeric")
        truth = rows.drop_duplicates("estimand_id")
        ax.plot(
            truth["x_numeric"],
            truth["mean_true"],
            color=color,
            linestyle=":" if scenario == "hold" else "-.",
            linewidth=1.0,
            label=f"Truth, {scenario_label[scenario]}",
        )
    all_rows = profiles.drop_duplicates("estimand_id").sort_values("x_numeric")
    ticks = all_rows["x_numeric"].to_numpy()
    tick_labels = all_rows["x_label"].tolist()
    ax.set_xticks(ticks, tick_labels)
    ax.axhline(0.0, color="#888888", linewidth=0.7, zorder=0)
    ax.set_xlabel("Event time relative to adoption")
    ax.set_ylabel("Mean lead effect")
    ax.set_title("Lead profiles; bars show across-replication error SD")
    ax.grid(axis="y", color="#d9dde2", linewidth=0.7)
    ax.legend(frameon=False, fontsize=7, loc="best")


def make_pretrend_figure(frame: pd.DataFrame, profiles: pd.DataFrame, degree: float) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1.18, 1.0]})
    _plot_rejection_panel(axes[0], frame, degree)
    _plot_profiles(axes[1], profiles, degree)
    fig.suptitle(f"Pre-trend diagnostic, degree {int(degree)}", y=1.01, fontsize=12)
    fig.text(
        0.5,
        -0.02,
        "Lead-profile error bars are across-replication SDs of estimation error, not posterior or Monte Carlo uncertainty.",
        ha="center",
        fontsize=7.5,
    )
    if degree == 3.0:
        fig.text(
            0.5,
            -0.055,
            "Repaired and reference-fold rows use 200 replications; historical TWFE uses 150 for cancelling heterogeneity 0.20 and 200 otherwise.",
            ha="center",
            fontsize=7.5,
        )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight", metadata={"Title": f"Pre-trend diagnostic, degree {int(degree)}"})
    plt.close(fig)
    return buffer.getvalue()


def make_prior_figure(frame: pd.DataFrame) -> bytes:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex="col")
    metrics = (("rmse", "ATT RMSE"), ("cover95", "95% interval coverage"))
    error_titles = {"iid": "Independent errors", "ar1": "AR(1) errors"}
    colors = {0: "#1f5a94", 1: "#b06a00", 2: "#2e7a4d"}
    ns = sorted(frame["N"].unique().tolist())
    rates = sorted(frame["sigma2_rate"].unique().tolist())
    for row_index, error in enumerate(("iid", "ar1")):
        for column_index, (metric, metric_label) in enumerate(metrics):
            ax = axes[row_index, column_index]
            subset = frame[frame["error_kind"] == error]
            for n_index, n_value in enumerate(ns):
                cells = subset[subset["N"] == n_value].sort_values("sigma2_rate")
                if len(cells) != len(rates):
                    raise AssetError(f"Incomplete prior sensitivity series for {error}, N={n_value}.")
                ax.plot(
                    cells["sigma2_rate"],
                    cells[metric],
                    marker="o",
                    linewidth=1.35,
                    markersize=4,
                    color=colors[n_index],
                    label=f"N={int(n_value)}",
                )
            ax.axvline(1.0, color="#777777", linewidth=0.8, linestyle=":")
            if metric == "cover95":
                ax.axhline(0.95, color="#777777", linewidth=0.8, linestyle="--")
                ax.set_ylim(0.0, 1.02)
            ax.set_xscale("log", base=10)
            ax.set_xticks(rates, [fmt_num(rate, 1) for rate in rates])
            ax.grid(color="#d9dde2", linewidth=0.7)
            ax.set_title(f"{error_titles[error]}: {metric_label}", fontsize=9.5)
            if column_index == 0:
                ax.set_ylabel(metric_label)
            if row_index == 1:
                ax.set_xlabel("Working-error prior rate")
            if row_index == 0 and column_index == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Prior-rate sensitivity of pooled ATT", y=1.01, fontsize=12)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight", metadata={"Title": "Prior-rate sensitivity of pooled ATT"})
    plt.close(fig)
    return buffer.getvalue()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_artifacts(
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    prior: pd.DataFrame,
    repo_root: Path,
) -> dict[Path, bytes]:
    artifacts: dict[Path, bytes] = {}
    table_assets = {
        "tab_pretrend.tex": make_supplement_table(summary, 1.0),
        "tab_pretrend_d3.tex": make_supplement_table(summary, 3.0),
        "tab_pretrend_main.tex": make_main_pretrend_table(summary),
        "tab_prior_sensitivity.tex": make_prior_table(prior),
    }
    figure_assets = {
        "fig_pretrend.pdf": make_pretrend_figure(summary, profiles, 1.0),
        "fig_pretrend_d3.pdf": make_pretrend_figure(summary, profiles, 3.0),
        "fig_prior_sensitivity.pdf": make_prior_figure(prior),
    }
    for name, content in table_assets.items():
        artifacts[repo_root.parent / "JBES_Submission" / "sim_tables" / name] = content.encode("utf-8")
        artifacts[repo_root / "Simulation_Studies_Revision" / "Results" / "tables" / name] = content.encode("utf-8")
    for name, content in figure_assets.items():
        artifacts[repo_root.parent / "JBES_Submission" / "sim_figures" / name] = content
        artifacts[repo_root / "Simulation_Studies_Revision" / "Results" / "figures" / name] = content
    return artifacts


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    try:
        summary_raw, profiles_raw, prior_raw = read_inputs(input_dir)
        summary = prepare_pretrend(summary_raw)
        profiles = prepare_profiles(profiles_raw)
        prior = prepare_prior(prior_raw)
        artifacts = build_artifacts(summary, profiles, prior, repo_root)
        for path, content in artifacts.items():
            atomic_write(path, content)
    except AssetError as exc:
        print(f"Publication assets not written: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Publication assets not written: unexpected error: {exc}", file=sys.stderr)
        return 1
    print(f"Validated provenance and wrote {len(artifacts)} mirrored publication assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
