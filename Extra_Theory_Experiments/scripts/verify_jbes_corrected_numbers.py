#!/usr/bin/env python3
"""Verify the corrected-estimator numbers now printed in the JBES submission.

Every new ``DiD-BCF (corrected)`` entry in ``JBES_Submission/sim_tables/*.tex``
and in the main-text Table 2 must equal the value produced by the archived
Extra_Theory_Experiments aggregation (``metrics_mean_median.csv``, which is the
input of the reconciliation tables), and the reconciliation table fragments must
carry the same values for the cells they report.  The script parses both sets of
LaTeX fragments, recomputes the expected values from the aggregation, and fails
loudly on any mismatch.

Run with the project venv:
    source /home/hugo_souto/Stuff/Research/DiD-BCF/.venv/bin/activate
    python Extra_Theory_Experiments/scripts/verify_jbes_corrected_numbers.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ETE = Path(__file__).resolve().parents[1]
REPO = ETE.parent
JBES = REPO.parent / "JBES_Submission"
TAB = JBES / "sim_tables"
RECON_TAB = ETE / "reconciliation" / "tables"
METRICS = ETE / "results" / "aggregated_correction" / "metrics_mean_median.csv"

EST = "reference_fold_convolution_rf"
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
    rows = []
    for line in path.read_text().splitlines():
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
    import numpy as np

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
    # 3. Main-text Table 2 rows match the reconciliation ATT cells.
    # ------------------------------------------------------------------
    text = (JBES / "main.tex").read_text()
    for design, fragment in (
            ("baseline", "$-0.015$ & 0.203 & 0.930 & 0.86"),
            ("serial", "$-0.018$ & 0.172 & 0.930 & 0.93")):
        m = metric(metrics, design, 1, 200, "ATT", "ATT")
        # Table 2 prints the ratio to two decimals.
        assert fragment.split("&")[0].strip("$ ") == f3(m["bias"])
        assert fragment.split("&")[1].strip() == f3(m["rmse"])
        assert fragment.split("&")[2].strip() == f3(m["cover95"])
        assert fragment.split("&")[3].strip() == f"{ratio_raw(m):.2f}"
        if fragment not in text:
            errors.append(f"main.tex Table 2: missing corrected row {fragment}")

    # ------------------------------------------------------------------
    # 4. Non-rerun corrected rows must be labeled as the earlier
    #    implementation.
    # ------------------------------------------------------------------
    earlier = "corrected, earlier"
    expected_counts = {
        "tab_b1_att.tex": 8,
        "tab_b1_gatt.tex": 8,
        "tab_sweep.tex": 6,
        "tab_sweep_deg.tex": 16,
        "tab_null.tex": 3,
        "tab_smalln_surface.tex": 2,
        "tab_surface.tex": 15,
        "tab_staggered.tex": 4,
        "tab_staggered_gatt.tex": 3,
        "tab_staggered_cells.tex": 9,
        "tab_staggered_deg.tex": 8,
    }
    for name, expected in expected_counts.items():
        path = TAB / name
        n = sum(1 for line in path.read_text().splitlines()
                if earlier in line and "&" in line and line.rstrip().endswith("\\\\"))
        if n != expected:
            errors.append(
                f"{name}: expected {expected} earlier-label rows, found {n}")

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
