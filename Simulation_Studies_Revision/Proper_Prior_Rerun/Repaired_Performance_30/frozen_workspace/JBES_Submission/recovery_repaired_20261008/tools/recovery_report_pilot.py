#!/usr/bin/env python3
"""Build the diagnostic-pilot acceptance report from a verified collection.

Reads the collector output (``chain_diagnostics.csv``, ``trace_diagnostics.csv``,
``status.json``) and the pilot manifest, applies the frozen acceptance targets
from ``pilot_protocol.md``, compares B1/B2 (warmup) and B1/B3 (retained length)
with the frozen stability rule, and writes:

    PILOT_ACCEPTANCE.md      human-readable verdict and tables
    pilot_acceptance.csv     one row per job/budget/parameter check
    pilot_stability.csv      paired budget comparisons
    pilot_traces_<case>.png  retained-draw traces (when --incoming is given)

This is a decision aid for the E2 gate; it does not certify convergence.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recovery_common import ACCEPTANCE, write_json_atomic

KEY_PARAMETERS = ("ATT", "sigma2")
PAIR_BUDGETS = (("B1_warm2500_ret1000", "B2_warm5000_ret1000"),
                ("B1_warm2500_ret1000", "B3_warm2500_ret2000"),
                ("B2_warm5000_ret1000", "B3_warm2500_ret2000"))


def _endpoint_ratio(row) -> float:
    ratios = []
    for low, high in (("q025", "q975"), ("q05", "q95")):
        width = abs(row[high] - row[low])
        errors = [row.get(low + "_mcse", np.nan), row.get(high + "_mcse", np.nan)]
        if not np.isfinite(errors).all():
            return float("nan")
        error = max(errors)
        if width <= 0 or not np.isfinite(error):
            return float("nan")
        ratios.append(error / width)
    return float(max(ratios))


def acceptance_table(diagnostics: pd.DataFrame, jobs: dict) -> pd.DataFrame:
    rows = []
    for record in diagnostics.itertuples():
        job = jobs.get(record.job_id, {})
        mean_ratio = (record.mean_mcse / record.post_sd
                      if record.post_sd else np.nan)
        endpoint_ratio = _endpoint_ratio(record._asdict())
        rows.append({
            "job_id": record.job_id, "budget_key": record.budget_key,
            "kind": job.get("kind"), "setting": job.get("setting"),
            "N": job.get("N"), "degree": job.get("linearity_degree"),
            "rep": job.get("rep"), "parameter": record.parameter,
            "rhat": record.rhat, "bulk_ess": record.bulk_ess,
            "tail_ess": record.tail_ess, "mean_mcse": record.mean_mcse,
            "post_sd": record.post_sd, "mean_mcse_over_sd": mean_ratio,
            "endpoint_mcse_over_width": endpoint_ratio,
            "pass_rhat": record.rhat <= ACCEPTANCE["rhat_max"],
            "pass_ess": np.isfinite([record.bulk_ess, record.tail_ess]).all()
                        and min(record.bulk_ess, record.tail_ess) >= ACCEPTANCE["ess_min"],
            "pass_mean": mean_ratio <= ACCEPTANCE["mean_mcse_over_sd_max"],
            "pass_endpoint": endpoint_ratio
                             <= ACCEPTANCE["endpoint_mcse_over_width_max"],
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        from recovery_accept import required_parameter
        frame["is_required"] = frame.parameter.map(required_parameter)
    if not frame.empty:
        numeric = ["pass_rhat", "pass_ess", "pass_mean", "pass_endpoint"]
        frame["pass_all"] = frame[numeric].all(axis=1)
    return frame


def stability_table(diagnostics: pd.DataFrame, jobs: dict) -> pd.DataFrame:
    index = diagnostics.set_index(["job_id", "budget_key", "parameter"])
    rows = []
    for first, second in PAIR_BUDGETS:
        for key in index.index.droplevel([1, 2]).unique():
            parameters = diagnostics.loc[diagnostics.job_id == key, "parameter"].unique()
            for parameter in parameters:
                try:
                    a = index.loc[(key, first, parameter)]
                    b = index.loc[(key, second, parameter)]
                except KeyError:
                    continue
                if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
                    continue
                combined_mean = float(np.hypot(a["mean_mcse"], b["mean_mcse"]))
                mean_diff = float(b["post_mean"] - a["post_mean"])
                ref_sd = float(b["post_sd"])
                mean_flag = not np.isfinite([combined_mean, mean_diff, ref_sd]).all() or abs(mean_diff) > max(
                    2 * combined_mean,
                    ACCEPTANCE["stability_mean_tolerance"] * ref_sd)
                combined_q025 = float(np.hypot(a.get("q025_mcse", np.nan),
                                               b.get("q025_mcse", np.nan)))
                combined_q975 = float(np.hypot(a.get("q975_mcse", np.nan),
                                               b.get("q975_mcse", np.nan)))
                width = abs(b["q975"] - b["q025"])
                tolerance = ACCEPTANCE["stability_endpoint_tolerance"] * width

                def endpoint_flagged(diff, combined):
                    if not np.isfinite([diff, combined, tolerance]).all():
                        return True
                    return abs(diff) > max(2 * combined, tolerance)

                endpoint_flag = (
                    endpoint_flagged(b["q025"] - a["q025"], combined_q025)
                    or endpoint_flagged(b["q975"] - a["q975"], combined_q975))
                for lo, hi in (("q05", "q95"),):
                    width90 = abs(b[hi] - b[lo])
                    for q in (lo, hi):
                        combined = float(np.hypot(a.get(q + "_mcse", np.nan), b.get(q + "_mcse", np.nan)))
                        difference = abs(b[q] - a[q])
                        if not np.isfinite(combined) or difference > max(2 * combined, ACCEPTANCE["stability_endpoint_tolerance"] * width90):
                            endpoint_flag = True
                job = jobs.get(key, {})
                rows.append({
                    "job_id": key, "setting": job.get("setting"),
                    "parameter": parameter, "first": first, "second": second,
                    "mean_first": a["post_mean"], "mean_second": b["post_mean"],
                    "mean_diff": mean_diff,
                    "combined_mean_mcse": combined_mean,
                    "mean_stability_flag": bool(mean_flag),
                    "endpoint_stability_flag": bool(endpoint_flag),
                })
    columns = ["job_id", "setting", "parameter", "first", "second",
               "mean_first", "mean_second", "mean_diff", "combined_mean_mcse",
               "mean_stability_flag", "endpoint_stability_flag"]
    return pd.DataFrame(rows, columns=columns)


def trace_plots(collection: Path, incoming: list, jobs: dict,
                out_dir: Path) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import zipfile
    except Exception:
        return []
    from recovery_collect import _load_draws, iter_complete_records
    entries = {}
    for source, member, record, loader in iter_complete_records(incoming):
        if record.get("chain_index") is None:
            continue
        key = (record["stat_job_id"], record["budget_key"])
        entries.setdefault(key, []).append({"record": record, "loader": loader})
    written = []
    for (job_id, budget), records in sorted(entries.items()):
        job = jobs.get(job_id, {})
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
        for column, parameter in enumerate(("ATT", "sigma2")):
            ax = axes[column]
            for entry in sorted(records, key=lambda r: r["record"]["chain_index"]):
                record = entry["record"]
                arrays = _load_draws(entry)
                if parameter not in arrays:
                    continue
                draws = arrays[parameter].reshape(-1)
                ax.plot(np.arange(1, draws.size + 1), draws, lw=.7,
                        label=f"chain {record['chain_index']}")
            ax.set_title(f"{parameter}: {job.get('setting')} "
                         f"N={job.get('N')} d={job.get('linearity_degree')} "
                         f"rep={job.get('rep')}\n{budget}")
            ax.set_xlabel("retained draw")
            ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / f"pilot_traces_{job_id}_{budget}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path.name)
    return written


def markdown_table(frame: pd.DataFrame, columns, digits=4) -> str:
    if frame.empty:
        return "(no rows)\n"
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--incoming", type=Path, nargs="*", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    jobs = {job["id"]: job for job in manifest["jobs"]}
    diagnostics = pd.read_csv(args.collection / "chain_diagnostics.csv")
    acceptance = acceptance_table(diagnostics, jobs)
    acceptance.to_csv(args.out / "pilot_acceptance.csv", index=False)
    stability = stability_table(diagnostics, jobs)
    stability.to_csv(args.out / "pilot_stability.csv", index=False)
    trace_path = args.collection / "trace_diagnostics.csv"
    traces = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
    status = json.loads((args.collection / "status.json").read_text())
    plot_names = (trace_plots(args.collection, args.incoming, jobs, args.out)
                  if args.incoming else [])

    key = acceptance[acceptance["parameter"].isin(KEY_PARAMETERS)]
    if acceptance.empty:
        summary = pd.DataFrame(columns=["budget_key", "checks", "passed"])
    else:
        summary = (acceptance[acceptance.is_required].assign(checks=1)
                   .groupby("budget_key")[["checks", "pass_all"]]
                   .sum().reset_index()
                   .rename(columns={"pass_all": "passed"}))
    lines = [
        "# Diagnostic pilot: frozen acceptance report",
        "",
        f"Protocol: `pilot_protocol.md` (frozen before results). Manifest "
        f"`{manifest['manifest_hash']}`.",
        f"Collected units: {status['complete_units']} / "
        f"{status['expected_units']}; conflicts {status['conflicting_units']}; "
        f"duplicates {status['duplicate_units']}.",
        "",
        "Rows below apply the frozen targets: R-hat <= 1.01, min(bulk,tail) "
        "ESS >= 400, mean MCSE <= 2% of posterior SD, endpoint MCSE <= 1% of "
        "the interval width. These are project precision screens, not "
        "convergence certificates.",
        "",
        "## Acceptance by budget",
        "",
        markdown_table(summary, list(summary.columns)),
        "",
        "## Key parameters (ATT, sigma2)",
        "",
        markdown_table(
            key[["job_id", "budget_key", "parameter", "rhat", "bulk_ess",
                 "tail_ess", "mean_mcse_over_sd", "endpoint_mcse_over_width",
                 "pass_all"]],
            ["job_id", "budget_key", "parameter", "rhat", "bulk_ess",
             "tail_ess", "mean_mcse_over_sd", "endpoint_mcse_over_width",
             "pass_all"]),
        "",
        "## Frozen stability comparisons",
        "",
        markdown_table(
            stability[["job_id", "parameter", "first", "second", "mean_diff",
                       "combined_mean_mcse", "mean_stability_flag",
                       "endpoint_stability_flag"]],
            ["job_id", "parameter", "first", "second", "mean_diff",
             "combined_mean_mcse", "mean_stability_flag",
             "endpoint_stability_flag"]),
        "",
        "Flags are investigated, not treated as automatic sampler defects. "
        "Multiple comparisons create flags even for adequate chains.",
        "",
    ]
    if not traces.empty:
        drift = traces[["run_id", "trace_matches_draws", "n_sweeps",
                        "att_burnin_first_quarter", "att_retained_first_quarter",
                        "att_retained_last_quarter", "sigma2_burnin_mean",
                        "sigma2_retained_mean", "sigma2_first_quarter",
                        "sigma2_last_quarter"]]
        lines += ["## Trace drift (first/last quarters)",
                  "", markdown_table(drift, list(drift.columns)), ""]
    if plot_names:
        lines += ["## Trace figures", "",
                  "Plots written next to this report: "
                  + ", ".join(f"`{name}`" for name in plot_names), ""]
    lines += [
        "## Gate decision (fill after review)",
        "",
        "1. accepted budget (or none): ______________________",
        "2. evidence for the decision, including unresolved failures:",
        "   ______________________________________________________",
        "3. if none accepted, diagnosed mechanism and next pilot:",
        "   ______________________________________________________",
        "",
        "Do not release production notebooks before this decision is recorded "
        "and frozen in `manifest_production.json`.",
    ]
    report = "\n".join(lines) + "\n"
    (args.out / "PILOT_ACCEPTANCE.md").write_text(report)
    write_json_atomic(args.out / "pilot_report_meta.json", {
        "protocol": "pilot_protocol.md", "manifest_hash": manifest["manifest_hash"],
        "files": {name: __import__("recovery_common").sha256_file(args.collection / name) for name in ("status.json", "chain_diagnostics.csv")},
        "collection_status": {k: status[k] for k in
                              ("complete_units", "expected_units",
                               "conflicting_units", "duplicate_units",
                               "smoke_rejected")},
        "acceptance_rows": int(len(acceptance)),
        "acceptance_pass_rows": int(acceptance["pass_all"].sum())
        if not acceptance.empty else 0,
    })
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
