#!/usr/bin/env python3
"""Verify the corrected-estimator numbers now printed in the JBES submission.

Every new ``DiD-BCF (corrected)`` entry in ``JBES_Submission/sim_tables/*.tex``
and in the main-text Table 2 must equal the value produced by the archived
Theory_Calibration aggregation (``metrics_mean_median.csv``, which is the
input of the reconciliation tables), and the reconciliation table fragments must
carry the same values for the cells they report.  The script parses both sets of
LaTeX fragments, recomputes the expected values from the aggregation, and fails
loudly on any mismatch.

Run with the project venv:
    source /home/hugo_souto/Stuff/Research/DiD-BCF/.venv/bin/activate
    python Simulation_Studies_Revision/Theory_Calibration/scripts/verify_jbes_corrected_numbers.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ETE = Path(__file__).resolve().parents[1]
REPO = ETE.parent.parent
JBES = REPO.parent / "JBES_Submission"
TAB = JBES / "sim_tables"
RECON_TAB = ETE / "reconciliation" / "tables"
METRICS = ETE / "results" / "aggregated_correction" / "metrics_mean_median.csv"
COMP_METRICS = ETE / "results" / "aggregated_completion" / "metrics_mean_median.csv"
COMP_PER_REP = (ETE / "results" / "aggregated_completion"
                / "combined_per_replication.csv")

EST = "reference_fold_convolution_rf"
EST_METHOD = "reference_fold_convolution"
SUMMARY = "mean"

ROW = re.compile(r"^\s*(?:\\quad\s*)*(?P<label>[^&]*?)\s*&(?P<body>.*)\\\\\s*$")


def f3(x: float) -> str:
    return f"{x:0.3f}"


def ratio_raw(m: dict) -> float:
    """The JBES tables report posterior spread over the raw spread of the
    estimates, not over the error SD used in the reconciliation note."""
    return m["avg_post_sd"] / m["raw_sd_est"]


def load_metrics() -> pd.DataFrame:
    # ``design == "null"`` is a legitimate value; disable NA coercion.
    return pd.read_csv(METRICS, keep_default_na=False, na_values=[])


def load_completion_metrics() -> pd.DataFrame:
    return pd.read_csv(COMP_METRICS, keep_default_na=False, na_values=[])


def load_completion_per_rep() -> pd.DataFrame:
    return pd.read_csv(COMP_PER_REP, engine="python", keep_default_na=False,
                       na_values=[], on_bad_lines="skip")


def load_published_metrics() -> pd.DataFrame:
    return pd.read_csv(REPO / "Simulation_Studies_Revision" / "Results"
                       / "aggregated" / "metrics_all.csv")


COMP_DESIGNS = {
    "B1_baseline": ["baseline"],
    "B1_serial_corr": ["serial"],
    "B1_null": ["null"],
    "B1_strong_confounder": ["strong_confounder"],
    "B1_selection_obs": ["selection_obs"],
    "B2_sweep": ["baseline", "baseline_sweep", "baseline_sweep_d3"],
    "B2_sweep_serial": ["serial", "serial_sweep", "serial_sweep_d3"],
    "D_staggered": ["staggered"],
    "PT_hold": ["PT_hold"],
    "PT_conditional": ["PT_conditional"],
    "PT_violation_g05": ["PT_violation_g05"],
    "PT_violation_g10": ["PT_violation_g10"],
    "PT_violation_g20": ["PT_violation_g20"],
    "PT_violation_g40": ["PT_violation_g40"],
    "PT_violation_a20": ["PT_violation_a20"],
    "PT_violation_het10": ["PT_violation_het10"],
    "PT_violation_het20": ["PT_violation_het20"],
    "PT_violation_het40": ["PT_violation_het40"],
}


def comp_metric(frame: pd.DataFrame, setting: str, degree: int, N: int,
                estimand_type: str, estimand_id: str,
                task_est: str = EST, method: str = EST_METHOD) -> dict:
    sub = frame[(frame["design"].isin(COMP_DESIGNS[setting]))
                & (frame["degree"] == degree) & (frame["N"] == N)
                & (frame["task_estimator"] == task_est)
                & (frame["method"] == method)
                & (frame["point_summary"] == SUMMARY)
                & (frame["estimand_type"] == estimand_type)
                & (frame["estimand_id"] == estimand_id)]
    if len(sub) != 1:
        raise AssertionError(
            f"expected exactly one completion metric row for ({setting}, "
            f"d{degree}, N{N}, {estimand_type}/{estimand_id}), found "
            f"{len(sub)}")
    return sub.iloc[0].to_dict()


def comp_size(frame: pd.DataFrame, design: str, degree: int, N: int,
              method: str, estimand_type: str, estimand_id: str):
    sub = frame[(frame["design"] == design) & (frame["degree"] == degree)
                & (frame["N"] == N) & (frame["method"] == method)
                & (frame["estimand_type"] == estimand_type)
                & (frame["estimand_id"] == estimand_id)]
    p = pd.to_numeric(sub["p_bayes_tail_min"], errors="coerce")
    p = p[p.notna()]
    if len(p) == 0:
        return float("nan"), 0
    r = float((p < 0.025).mean())
    return r, len(p)


def metric(frame: pd.DataFrame, design: str, degree: int, N: int,
           estimand_type: str, estimand_id: str) -> dict:
    sub = frame[(frame["design"] == design) & (frame["degree"] == degree)
                & (frame["N"] == N) & (frame["task_estimator"] == EST)
                & (frame["point_summary"] == SUMMARY)
                & (frame["estimand_type"] == estimand_type)
                & (frame["estimand_id"] == estimand_id)]
    if len(sub) != 1:
        raise AssertionError(
            f"expected exactly one metric row for ({design}, d{degree}, N{N}, "
            f"{estimand_type}/{estimand_id}), found {len(sub)}")
    return sub.iloc[0].to_dict()


def parse_rows(path: Path) -> list[tuple[str, list[str]]]:
    return parse_rows_text(path.read_text())


def parse_rows_text(text: str) -> list[tuple[str, list[str]]]:
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        label = re.sub(r"\s+", " ", m.group("label")).strip()
        values = [v.strip() for v in m.group("body").split("&")]
        rows.append((label, values))
    return rows


def find_row(path: Path, label: str, numeric_hint: str) -> list[str]:
    hits = [vals for lab, vals in parse_rows(path)
            if lab == label and numeric_hint in " & ".join(vals)]
    if not hits:
        raise AssertionError(
            f"{path.name}: no row labeled {label!r} containing "
            f"{numeric_hint!r}")
    if any(h != hits[0] for h in hits[1:]):
        raise AssertionError(
            f"{path.name}: ambiguous rows labeled {label!r} containing "
            f"{numeric_hint!r}: {hits}")
    return hits[0]


def eq(actual: str, expected: str, context: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{context}: table has {actual!r}, expected {expected!r}")


def labelled_cells(path: Path, label_fragment: str) -> list[list[str]]:
    """Cells after the estimator label for every line containing it."""
    out = []
    for line in path.read_text().splitlines():
        if label_fragment not in line:
            continue
        cells = [re.sub(r"\\\\\s*$", "", c).strip()
                 for c in line.split("&")]
        idx = next(i for i, c in enumerate(cells) if label_fragment in c)
        cells[idx] = re.sub(r"^\\quad\s*", "", cells[idx]).strip()
        out.append(cells[idx + 1:])
    return out


def main() -> int:
    metrics = load_metrics()
    errors: list[str] = []

    # ------------------------------------------------------------------
    # 1. Cells rerun with the reference-fold convolution: JBES values must
    #    equal the aggregation.
    # ------------------------------------------------------------------
    # tab_b1_att.tex: d, Bias, SD(err), RMSE, Cov.90, Cov.95, Len.95, ratio.
    b1_att = TAB / "tab_b1_att.tex"
    for design, degree in (("baseline", 1), ("baseline", 2),
                           ("serial", 1), ("serial", 2)):
        m = metric(metrics, design, degree, 200, "ATT", "ATT")
        row = find_row(b1_att, "DiD-BCF (corrected)",
                       f"{f3(m['bias'])} & {f3(m['emp_sd'])}")
        expected = [str(degree), f3(m["bias"]), f3(m["emp_sd"]), f3(m["rmse"]),
                    f3(m["cover90"]), f3(m["cover95"]), f3(m["len95"]),
                    f3(ratio_raw(m))]
        for a, e, col in zip(row, expected,
                             ["d", "bias", "sd", "rmse", "cov90", "cov95",
                              "len95", "ratio"]):
            eq(a, e, f"tab_b1_att {design} d{degree} {col}", errors)

    # tab_b1_gatt.tex: same columns, GATT(4,4), N=200.
    b1_gatt = TAB / "tab_b1_gatt.tex"
    for design, degree in (("baseline", 1), ("baseline", 2),
                           ("serial", 1), ("serial", 2)):
        m = metric(metrics, design, degree, 200, "GATT", "g=4_t=4")
        row = find_row(b1_gatt, "DiD-BCF (corrected)",
                       f"{f3(m['bias'])} & {f3(m['emp_sd'])}")
        expected = [str(degree), f3(m["bias"]), f3(m["emp_sd"]), f3(m["rmse"]),
                    f3(m["cover90"]), f3(m["cover95"]), f3(m["len95"]),
                    f3(ratio_raw(m))]
        for a, e, col in zip(row, expected,
                             ["d", "bias", "sd", "rmse", "cov90", "cov95",
                              "len95", "ratio"]):
            eq(a, e, f"tab_b1_gatt {design} d{degree} {col}", errors)

    # tab_sweep.tex (d=1, columns N, Bias, SD, RMSE, Cov.95, Len, ratio)
    # and tab_sweep_deg.tex (columns N, d, Bias, SD, RMSE, Cov.95, Len,
    # ratio) at N=200 and N=800.
    for path, degree in ((TAB / "tab_sweep.tex", 1),
                         (TAB / "tab_sweep_deg.tex", 2)):
        for design in ("baseline", "serial"):
            for N in (200, 800):
                m = metric(metrics, design, degree, N, "GATT", "g=4_t=4")
                row = find_row(path, "DiD-BCF (corrected)",
                               f"{f3(m['bias'])} & {f3(m['emp_sd'])} & "
                               f"{f3(m['rmse'])}")
                if path.name == "tab_sweep.tex":
                    expected = [str(N), f3(m["bias"]), f3(m["emp_sd"]),
                                f3(m["rmse"]), f3(m["cover95"]),
                                f3(m["len95"]), f3(ratio_raw(m))]
                    cols = ["N", "bias", "sd", "rmse", "cov95", "len95",
                            "ratio"]
                else:
                    expected = [str(N), str(degree), f3(m["bias"]),
                                f3(m["emp_sd"]), f3(m["rmse"]),
                                f3(m["cover95"]), f3(m["len95"]),
                                f3(ratio_raw(m))]
                    cols = ["N", "d", "bias", "sd", "rmse", "cov95", "len95",
                            "ratio"]
                for a, e, col in zip(row, expected, cols):
                    eq(a, e,
                       f"{path.name} {design} d{degree} N{N} {col}", errors)

    # tab_null.tex: d, Bias, SD(err), Cov.90, Cov.95 at d=1,2, N=200.
    null = TAB / "tab_null.tex"
    for estimand_type, estimand_id in (("ATT", "ATT"), ("ES", "k=0"),
                                       ("GATT", "g=4_t=4")):
        for degree in (1, 2):
            m = metric(metrics, "null", degree, 200, estimand_type,
                       estimand_id)
            row = find_row(null, "DiD-BCF (corrected)",
                           f"{f3(m['bias'])} & {f3(m['emp_sd'])}")
            expected = [str(degree), f3(m["bias"]), f3(m["emp_sd"]),
                        f3(m["cover90"]), f3(m["cover95"])]
            for a, e, col in zip(row, expected,
                                 ["d", "bias", "sd", "cov90", "cov95"]):
                eq(a, e, f"tab_null {estimand_type} d{degree} {col}", errors)

    # Sharp-null size: the JBES table Size and MCSE columns must match the
    # reconciliation table (GATT/ATT) and the per-replication archive (ES).
    rec_null = parse_rows(RECON_TAB / "tab_null.tex")
    rec_row = [v for lab, v in rec_null
               if lab == "New reference fold, RF"]
    if len(rec_row) != 1:
        errors.append("tab_null (reconciliation): expected one reference-fold "
                      f"row, found {len(rec_row)}")
        rec_sizes = None
    else:
        rec_sizes = {"GATT": rec_row[0][0:4], "ATT": rec_row[0][4:8]}
    per_rep = pd.read_csv(
        ETE / "results" / "aggregated_correction"
        / "combined_per_replication.csv",
        engine="python", keep_default_na=False, na_values=[],
        on_bad_lines="skip")
    for estimand_type, estimand_id in (("ATT", "ATT"), ("ES", "k=0"),
                                       ("GATT", "g=4_t=4")):
        sub = per_rep[(per_rep["design"] == "null")
                      & (per_rep["task_estimator"] == EST)
                      & (per_rep["estimand_type"] == estimand_type)
                      & (per_rep["estimand_id"] == estimand_id)]
        for degree in (1, 2):
            cell = sub[(sub["degree"] == degree) & (sub["N"] == 200)]
            p = float((pd.to_numeric(cell["p_bayes_tail_min"],
                                     errors="coerce") < 0.025).mean())
            m = metric(metrics, "null", degree, 200, estimand_type,
                       estimand_id)
            row = find_row(null, "DiD-BCF (corrected)",
                           f"{f3(m['bias'])} & {f3(m['emp_sd'])}")
            eq(row[5], f3(p),
               f"tab_null {estimand_type} d{degree} size", errors)
            eq(row[6], f3(np.sqrt(p * (1 - p) / len(cell))),
               f"tab_null {estimand_type} d{degree} MCSE", errors)
            if rec_sizes is not None and estimand_type in rec_sizes:
                idx = degree - 1
                eq(row[5], rec_sizes[estimand_type][idx],
                   f"tab_null {estimand_type} d{degree} vs reconciliation",
                   errors)

    # tab_smalln.tex: the reference-fold rows must report the N=200 rerun
    # values in the last four columns (Bias, RMSE, Cov., ratio).
    smalln = TAB / "tab_smalln.tex"
    ref_rows = [c[-4:] for c in labelled_cells(
        smalln, "DiD-BCF (ref. fold)")]
    for design, target in (("baseline", "gatt"), ("baseline", "att"),
                           ("serial", "gatt"), ("serial", "att")):
        et, eid = (("GATT", "g=4_t=4") if target == "gatt"
                   else ("ATT", "ATT"))
        m = metric(metrics, design, 1, 200, et, eid)
        # tab_smalln prints coverage and ratio to two decimals.
        values = [f3(m["bias"]), f3(m["rmse"]), f"{m['cover95']:.2f}",
                  f"{ratio_raw(m):.2f}"]
        if values not in ref_rows:
            errors.append(
                f"tab_smalln {design} {target}: no reference-fold row matches "
                f"{values} (found {ref_rows})")

    # ------------------------------------------------------------------
    # 1b. Completion rerun cells: JBES values must equal the completion
    #     aggregation.
    # ------------------------------------------------------------------
    comp = load_completion_metrics()
    comp_rep = load_completion_per_rep()

    def check_corrected_row(path, setting, degree, N, etype, eid, cols,
                            label="DiD-BCF (corrected)", extra=None,
                            hint=None):
        extra = extra or {}
        m = comp_metric(comp, setting, degree, N, etype, eid)
        if hint is None:
            hint = f"{f3(m['bias'])} & {f3(m['emp_sd'])}"
        row = find_row(path, label, hint)
        expected = []
        for col in cols:
            if col == "degree":
                expected.append(str(degree))
            elif col == "N":
                expected.append(str(N))
            elif col == "ratio":
                expected.append(f3(ratio_raw(m)))
            elif col == "ratio2":
                expected.append(f"{ratio_raw(m):.2f}")
            elif col in extra:
                expected.append(str(extra[col]))
            else:
                expected.append(f3(m[col]))
        for a, e, col in zip(row, expected, cols):
            eq(a, e, f"{path.name} {setting} d{degree} N{N} "
               f"{etype}/{eid} {col}", errors)
        return m

    b1cols = ["degree", "bias", "emp_sd", "rmse", "cover90", "cover95",
              "len95", "ratio"]
    for path, etype, eid in ((TAB / "tab_b1_att.tex", "ATT", "ATT"),
                             (TAB / "tab_b1_gatt.tex", "GATT", "g=4_t=4")):
        for setting, degrees in (("B1_strong_confounder", (1, 2, 3)),
                                 ("B1_selection_obs", (1, 2, 3)),
                                 ("B1_baseline", (3,)),
                                 ("B1_serial_corr", (3,))):
            for degree in degrees:
                check_corrected_row(path, setting, degree, 200, etype, eid,
                                    b1cols)

    # tab_null.tex d=3 rows, including the honest sizes.
    for etype, eid in (("ATT", "ATT"), ("ES", "k=0"), ("GATT", "g=4_t=4")):
        m = check_corrected_row(TAB / "tab_null.tex", "B1_null", 3, 200,
                                etype, eid,
                                ["degree", "bias", "emp_sd", "cover90",
                                 "cover95"])
        row = find_row(TAB / "tab_null.tex", "DiD-BCF (corrected)",
                       f"{f3(m['bias'])} & {f3(m['emp_sd'])}")
        r, n = comp_size(comp_rep, "null", 3, 200, EST_METHOD, etype, eid)
        eq(row[5], f3(r), f"tab_null {etype} d3 size", errors)
        eq(row[6], f3(np.sqrt(r * (1 - r) / n)),
           f"tab_null {etype} d3 MCSE", errors)

    # tab_sweep.tex (d=1) and tab_sweep_deg.tex (d=2,3) corrected rows.
    for path, degree, N in ((TAB / "tab_sweep.tex", 1, 50),
                            (TAB / "tab_sweep.tex", 1, 100),
                            (TAB / "tab_sweep.tex", 1, 400)):
        for setting in ("B2_sweep", "B2_sweep_serial"):
            check_corrected_row(path, setting, degree, N, "GATT", "g=4_t=4",
                                ["N", "bias", "emp_sd", "rmse", "cover95",
                                 "len95", "ratio"])
    for degree, N in ((2, 50), (2, 100), (2, 400), (3, 50), (3, 100),
                      (3, 200), (3, 400), (3, 800)):
        for setting in ("B2_sweep", "B2_sweep_serial"):
            check_corrected_row(TAB / "tab_sweep_deg.tex", setting, degree,
                                N, "GATT", "g=4_t=4",
                                ["N", "degree", "bias", "emp_sd", "rmse",
                                 "cover95", "len95", "ratio"])

    # tab_smalln.tex ref.fold rows at N=50 and N=100.
    smalln_ref = labelled_cells(smalln, "DiD-BCF (ref. fold)")
    for setting, target in (("B2_sweep", "gatt"), ("B2_sweep", "att"),
                            ("B2_sweep_serial", "gatt"),
                            ("B2_sweep_serial", "att")):
        et, eid = (("GATT", "g=4_t=4") if target == "gatt"
                   else ("ATT", "ATT"))
        for N in (50, 100):
            m = comp_metric(comp, setting, 1, N, et, eid)
            values = [f3(m["bias"]), f3(m["rmse"]), f"{m['cover95']:.2f}",
                      f"{ratio_raw(m):.2f}"]
            group = [50, 100, 200].index(N)
            if not any(values == row[4 * group:4 * group + 4]
                       for row in smalln_ref):
                errors.append(
                    f"tab_smalln {setting} {target} N{N}: no reference-fold "
                    f"row matches {values}")

    # tab_staggered.tex (ES d=1), tab_staggered_deg.tex (ES d=2,3),
    # tab_staggered_cells.tex (GATT d=1).
    for k in (0, 1, 2, 3):
        check_corrected_row(TAB / "tab_staggered.tex", "D_staggered", 1,
                            200, "ES", f"k={k}",
                            ["k", "mean_true", "bias", "emp_sd", "rmse",
                             "cover95"],
                            extra={"k": k})
    for degree in (2, 3):
        for k in (0, 1, 2, 3):
            check_corrected_row(TAB / "tab_staggered_deg.tex", "D_staggered",
                                degree, 200, "ES", f"k={k}",
                                ["degree", "k", "mean_true", "bias",
                                 "emp_sd", "rmse", "cover95"],
                                extra={"k": k})
    for g, t, k in ((4, 4, 0), (5, 5, 0), (6, 6, 0), (4, 5, 1), (5, 6, 1),
                    (6, 7, 1), (4, 6, 2), (5, 7, 2), (4, 7, 3)):
        m = comp_metric(comp, "D_staggered", 1, 200, "GATT", f"g={g}_t={t}")
        check_corrected_row(TAB / "tab_staggered_cells.tex", "D_staggered",
                            1, 200, "GATT", f"g={g}_t={t}",
                            ["g", "t", "k", "mean_true", "bias", "rmse",
                             "cover95"],
                            extra={"g": g, "t": t, "k": k},
                            hint=f"{g} & {t} & {k} & {f3(m['mean_true'])}")

    # tab_staggered_gatt.tex: cell-averaged reference rows.
    for degree in (1, 2, 3):
        cells = [f"g={g}_t={t}" for g in (4, 5, 6) for t in range(g, 8)]
        vals = [comp_metric(comp, "D_staggered", degree, 200, "GATT", eid)
                for eid in cells]
        exp = [str(degree), "9", f3(np.mean([v["mean_true"] for v in vals])),
               f3(np.mean([v["bias"] for v in vals])),
               f3(np.mean([v["emp_sd"] for v in vals])),
               f3(np.mean([v["rmse"] for v in vals])),
               f3(np.mean([v["cover95"] for v in vals]))]
        row = find_row(TAB / "tab_staggered_gatt.tex", "DiD-BCF (corrected)",
                       f"{exp[3]} & {exp[4]}")
        for a, e, col in zip(row, exp,
                             ["d", "cells", "true", "bias", "sd", "rmse",
                              "cov95"]):
            eq(a, e, f"tab_staggered_gatt d{degree} {col}", errors)

    # ------------------------------------------------------------------
    # 2. The reconciliation fragments themselves must carry the same values
    #    for the cells they report.
    # ------------------------------------------------------------------
    recon = {
        ("baseline", "ATT"): RECON_TAB / "tab_repro_att.tex",
        ("serial", "ATT"): RECON_TAB / "tab_repro_att.tex",
        ("baseline", "GATT"): RECON_TAB / "tab_repro_baseline.tex",
        ("serial", "GATT"): RECON_TAB / "tab_repro_serial.tex",
    }
    for (design, etype), path in recon.items():
        for degree in (1, 2):
            m = metric(metrics, design, degree, 200, etype,
                       "g=4_t=4" if etype == "GATT" else "ATT")
            row = find_row(path, "New reference fold, RF",
                           f"{f3(m['bias'])} & {f3(m['emp_sd'])}")
            expected = [f3(m["bias"]), f3(m["emp_sd"]), f3(m["rmse"]),
                        f3(m["cover95"]), f3(m["len95"]), f3(m["sd_ratio"]),
                        str(int(m["n_reps"]))]
            for a, e, col in zip(row, expected,
                                 ["bias", "sd", "rmse", "cov95", "len95",
                                  "ratio", "n"]):
                eq(a, e, f"{path.name} {design} d{degree} {etype} {col}",
                   errors)
    for design, degree in (("baseline", 1), ("baseline", 2),
                           ("serial", 1), ("serial", 2)):
        m = metric(metrics, design, degree, 800, "GATT", "g=4_t=4")
        row = find_row(RECON_TAB / "tab_repro_sweep.tex",
                       "New reference fold, RF",
                       f"{f3(m['bias'])} & {f3(m['emp_sd'])}")
        expected = [f3(m["bias"]), f3(m["emp_sd"]), f3(m["rmse"]),
                    f3(m["cover95"]), f3(m["len95"]), f3(m["sd_ratio"]),
                    str(int(m["n_reps"]))]
        for a, e, col in zip(row, expected,
                             ["bias", "sd", "rmse", "cov95", "len95", "ratio",
                              "n"]):
            eq(a, e, f"tab_repro_sweep {design} d{degree} {col}", errors)

    # ------------------------------------------------------------------
    # 3. Main-text Table 2 rows match the corrected ATT cells.
    # ------------------------------------------------------------------
    text = (JBES / "main.tex").read_text()
    comp = load_completion_metrics()
    table2 = [(metric(metrics, "baseline", 1, 200, "ATT", "ATT"), None),
              (metric(metrics, "serial", 1, 200, "ATT", "ATT"), None),
              (comp_metric(comp, "B1_strong_confounder", 1, 200, "ATT", "ATT"),
               "completion"),
              (comp_metric(comp, "B1_selection_obs", 1, 200, "ATT", "ATT"),
               "completion")]
    for m, _src in table2:
        fragment = (f"${f3(m['bias'])}$ & {f3(m['rmse'])} & "
                    f"{f3(m['cover95'])} & {ratio_raw(m):.2f}")
        if fragment not in text:
            errors.append(
                f"main.tex Table 2: missing corrected row {fragment}")

    # ------------------------------------------------------------------
    # 4. Non-rerun corrected rows must be labeled as the earlier
    #    implementation.  After the completion rerun only the two surface
    #    tables keep earlier labels (no surface estimands were rerun).
    # ------------------------------------------------------------------
    earlier = "corrected, earlier"
    expected_counts = {
        "tab_b1_att.tex": 0,
        "tab_b1_gatt.tex": 0,
        "tab_sweep.tex": 0,
        "tab_sweep_deg.tex": 0,
        "tab_null.tex": 0,
        "tab_smalln_surface.tex": 2,
        "tab_surface.tex": 15,
        "tab_staggered.tex": 0,
        "tab_staggered_gatt.tex": 0,
        "tab_staggered_cells.tex": 0,
        "tab_staggered_deg.tex": 0,
    }
    for name, expected in expected_counts.items():
        path = TAB / name
        n = sum(1 for line in path.read_text().splitlines()
                if earlier in line and "&" in line and line.rstrip().endswith("\\\\"))
        if n != expected:
            errors.append(
                f"{name}: expected {expected} earlier-label rows, found {n}")

    # ------------------------------------------------------------------
    # 5. Pre-trend tables: raw and fold rows match the completion
    #    aggregation; the TWFE column matches the published run.
    # ------------------------------------------------------------------
    pub = load_published_metrics()
    pt_violations = {"PT_hold": "none (PTA holds)",
                     "PT_conditional": "unconditional only",
                     "PT_violation_g05": "group slope 0.05",
                     "PT_violation_g10": "group slope 0.10",
                     "PT_violation_g20": "group slope 0.20",
                     "PT_violation_g40": "group slope 0.40",
                     "PT_violation_a20": r"via $\alpha_i$",
                     "PT_violation_het10": "heterogeneous, cancels",
                     "PT_violation_het20": "heterogeneous, cancels",
                     "PT_violation_het40": "heterogeneous, cancels"}

    def check_pt_table(path, degree, text=None):
        rows = parse_rows_text(text) if text is not None else parse_rows(path)
        scenario = None
        seen = set()
        for label, cells in rows:
            if label:
                scenario = label.replace(r"\_", "_")
            if scenario is None:
                continue
            version = cells[0] if cells else ""
            if version not in ("raw", "ref. fold"):
                continue
            key = (scenario, version)
            if key in seen:
                errors.append(f"{path.name}: duplicate row {key}")
            seen.add(key)
            meth = ("pretrend" if version == "raw"
                    else "reference_fold_pretrend")
            task_est = ("raw_structured" if version == "raw"
                        else "reference_fold_convolution_rf")
            sub = comp_rep[(comp_rep["design"] == scenario)
                           & (comp_rep["degree"] == degree)
                           & (comp_rep["method"] == meth)]
            if sub.empty:
                errors.append(f"{path.name}: no completion rows for {key}")
                continue
            p_slope = pd.to_numeric(sub.loc[
                (sub["estimand_type"] == "PRE")
                & (sub["estimand_id"] == "slope"),
                "p_bayes_tail_min"], errors="coerce")
            p_con = pd.to_numeric(sub.loc[
                (sub["estimand_type"] == "PRE_SUBC")
                & (sub["estimand_id"] == "X1_slope"),
                "p_bayes_tail_min"], errors="coerce")
            ks = sub[(sub["estimand_type"] == "PRE")
                     & (sub["estimand_id"].str.startswith("k="))]
            kvals = pd.to_numeric(ks["p_bayes_tail_min"], errors="coerce")
            ks = ks.assign(_p=kvals.values).dropna(subset=["_p"])
            anyk = ((ks.groupby("rep")["_p"].min() * 3.0).clip(upper=1.0)
                    < 0.025).mean() if not ks.empty else float("nan")
            m = comp_metric(comp, scenario, degree, 200, "PRE", "slope",
                            task_est=task_est, method=meth)
            tw = pub[(pub["setting"] == scenario)
                     & (pub["method"] == "twfe_es")
                     & (pub["linearity_degree"] == degree)
                     & (pub["estimand_type"] == "PRE")
                     & (pub["estimand_id"] == "slope")]
            twfe = float(tw.iloc[0]["reject05"]) if len(tw) else float("nan")
            # cells: [version, violation, true, slope, contrast,
            #         anyk, bias, cover, twfe]
            eq(cells[1], pt_violations[scenario],
               f"{path.name} {key} violation", errors)
            eq(cells[2], f3(m["mean_true"]),
               f"{path.name} {key} true", errors)
            eq(cells[3], f3(float((p_slope < 0.025).mean())),
               f"{path.name} {key} slope", errors)
            eq(cells[4], f3(float((p_con < 0.025).mean())),
               f"{path.name} {key} contrast", errors)
            eq(cells[5], f3(float(anyk)), f"{path.name} {key} anyk", errors)
            eq(cells[6], f3(m["bias"]), f"{path.name} {key} bias", errors)
            eq(cells[7], f3(m["cover95"]), f"{path.name} {key} cover",
               errors)
            if version == "raw":
                eq(cells[8], f3(twfe), f"{path.name} {key} twfe", errors)
            elif cells[8].strip():
                errors.append(f"{path.name} {key}: expected blank TWFE cell, "
                              f"found {cells[8]!r}")
        if len(seen) != 20:
            errors.append(f"{path.name}: expected 20 scenario-version rows, "
                          f"found {len(seen)}")

    check_pt_table(TAB / "tab_pretrend.tex", 1)
    check_pt_table(TAB / "tab_pretrend_d3.tex", 3)

    # The main-text pre-trend table carries the same d=1 rows inline.
    main_text = (JBES / "main.tex").read_text()
    start = main_text.find(r"\label{tab:pretrend_main}")
    if start < 0:
        errors.append("main.tex: tab:pretrend_main label not found")
    else:
        end = main_text.find(r"\end{table}", start)
        check_pt_table(Path("main.tex tab:pretrend_main"), 1,
                       main_text[start:end])

    if errors:
        print("FAIL")
        for e in errors:
            print("  -", e)
        return 1
    print("PASS: all JBES corrected numbers match the reconciliation "
          "aggregation and labels are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
