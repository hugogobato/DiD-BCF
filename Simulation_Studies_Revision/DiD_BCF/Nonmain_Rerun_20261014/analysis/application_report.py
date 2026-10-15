#!/usr/bin/env python3
"""Create manuscript-review tables for the empirical MPDTA application.

This reads the compact collection written by ``helpers/collect.py``. It never
fits a model or reads posterior arrays. The default run requires all 27
application jobs; ``--allow-partial`` is restricted to the frozen
``validation/reuse_collection`` fixture and labels every resulting report as
partial validation output.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable


PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PACKAGE / "manifest.json"
EXPECTED_PHASES = {"application": 9, "application_sensitivity": 18}
EXPECTED_APPLICATION_JOBS = sum(EXPECTED_PHASES.values())
EXPECTED_RATES = (0.1, 1.0, 10.0)
ESTIMATOR_SPECS = ("structured_rfx_unit", "structured")
EXPECTED_GROUP_COUNTS = {
    "Small population": 72,
    "Medium population": 119,
    "High population": 100,
}
APPLICATION_TABLES = (
    "application_gate.csv",
    "application_catt.csv",
    "application_contrasts.csv",
    "per_replication.csv",
    "diagnostics.csv",
)
FORBIDDEN_APPLICATION_COLUMNS = {
    "true", "error", "absolute_error", "squared_error", "coverage",
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise ValueError(f"Cannot read CSV file {path}: {exc}") from exc


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: object, label: str) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid integer for {label}: {value!r}") from exc


def as_float(value: object, label: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid number for {label}: {value!r}") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"Non-finite number for {label}: {value!r}")
    return result


def check_partial_location(input_dir: Path) -> None:
    parts = input_dir.resolve().parts
    if len(parts) < 2 or parts[-2:] != ("validation", "reuse_collection"):
        raise ValueError(
            "--allow-partial is only permitted for a collection directory named "
            "validation/reuse_collection"
        )


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def job_value(row: dict[str, str], field: str) -> str:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing {field} in row for job {row.get('job_id')}")
    return value


def add_job_metadata(rows: list[dict[str, str]], jobs: dict[str, dict],
                     expected_kind: str = "application") -> list[dict[str, str]]:
    result = []
    for original in rows:
        job_id = original.get("job_id", "")
        if job_id not in jobs:
            raise ValueError(f"Application table contains unknown job_id {job_id!r}")
        job = jobs[job_id]
        if job.get("phase") not in EXPECTED_PHASES or job.get("kind") != expected_kind:
            raise ValueError(f"Unexpected job kind for application output {job_id}")
        if original.get("spec") not in (None, "", job.get("spec")):
            raise ValueError(f"Specification mismatch for application job {job_id}")
        if original.get("rep") not in (None, "", str(job.get("rep"))):
            raise ValueError(f"Replication mismatch for application job {job_id}")
        row = dict(original)
        row.update({
            "job_phase": str(job["phase"]),
            "job_kind": str(job["kind"]),
            "sigma2_shape": str(job["sigma2_shape"]),
            "sigma2_rate": str(job["sigma2_rate"]),
            "setting": str(job["setting"]),
        })
        result.append(row)
    return result


def write_enriched(path: Path, rows: list[dict[str, str]], original_fields: list[str]) -> None:
    metadata_fields = ["job_phase", "job_kind", "sigma2_shape", "sigma2_rate", "setting"]
    fields = list(dict.fromkeys([*original_fields, *metadata_fields]))
    write_csv(path, rows, fields)


def draw_interval(ax, x: float, y: float, lower: float, upper: float,
                  color: str, marker: str, orientation: str = "horizontal",
                  cap: float = 0.06) -> None:
    """Draw exported interval endpoints literally, even if mean is outside them."""
    if orientation == "horizontal":
        ax.hlines(y, lower, upper, color=color, linewidth=1.7, zorder=2)
        ax.vlines([lower, upper], y - cap, y + cap, color=color, linewidth=1.1, zorder=2)
    elif orientation == "vertical":
        ax.vlines(x, lower, upper, color=color, linewidth=1.7, zorder=2)
        ax.hlines([lower, upper], x - cap, x + cap, color=color, linewidth=1.1, zorder=2)
    else:
        raise ValueError(f"Unknown interval orientation: {orientation}")
    ax.plot(x, y, marker=marker, color=color, markersize=6,
            linestyle="none", zorder=3)


def make_plots(output_dir: Path, primary_id_by_role: dict[str, str | None],
               gates_by_job: dict[str, list[dict[str, str]]],
               catt_by_job: dict[str, list[dict[str, str]]],
               gates: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Write available standalone vector PDFs; Matplotlib is optional and lazy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_config = tempfile.TemporaryDirectory(prefix="didbcf-application-mpl-")
    previous_config = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = temp_config.name
    try:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError as exc:
            raise ValueError("--plots requires Matplotlib; install it in the reporting environment") from exc

        figures: list[tuple[str, str]] = []
        with matplotlib.rc_context({
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }):
            unit_id = primary_id_by_role.get("Primary: unit random effects")
            comparison_id = primary_id_by_role.get("Comparison: no random effects")
            if unit_id and comparison_id:
                labels = ["All treated", *EXPECTED_GROUP_COUNTS]
                counts = {
                    row["group"]: row["n_obs"]
                    for row in gates_by_job[unit_id]
                }
                y_base = {group: len(labels) - index - 1 for index, group in enumerate(labels)}
                palette = {
                    "Primary: unit random effects": "#1f77b4",
                    "Comparison: no random effects": "#d17a22",
                }
                markers = {
                    "Primary: unit random effects": "o",
                    "Comparison: no random effects": "s",
                }
                fig, ax = plt.subplots(figsize=(8.6, 4.4))
                for role, job_id, offset in (
                    ("Primary: unit random effects", unit_id, 0.13),
                    ("Comparison: no random effects", comparison_id, -0.13),
                ):
                    for row in gates_by_job[job_id]:
                        group = row["group"]
                        mean = as_float(row["att_pct"], "att_pct")
                        lower = as_float(row["lo95_pct"], "lo95_pct")
                        upper = as_float(row["hi95_pct"], "hi95_pct")
                        draw_interval(ax, mean, y_base[group] + offset, lower, upper,
                                      palette[role], markers[role])
                ax.axvline(0, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
                ax.set_yticks([y_base[group] for group in labels])
                ax.set_yticklabels([f"{group} (n={counts[group]})" for group in labels])
                ax.set_xlabel("Treatment effect (%)")
                ax.grid(axis="x", color="#dddddd", linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                legend = [
                    Line2D([0], [0], marker=markers[role], color=palette[role],
                           linestyle="-", markersize=6,
                           label=("Structured (unit random effects)"
                                  if role == "Primary: unit random effects"
                                  else "Structured (no random effects)"))
                    for role in palette
                ]
                fig.suptitle("MPDTA: primary estimate and structured comparison", y=0.99)
                fig.legend(handles=legend, loc="upper center", ncol=2,
                           frameon=False, bbox_to_anchor=(0.5, 0.94))
                fig.text(0.01, 0.01,
                         "Posterior means and 95% credible intervals; counts refer to treated county-year observations.",
                         ha="left", va="bottom", fontsize=8)
                fig.tight_layout(rect=(0, 0.05, 1, 0.86))
                filename = "application_primary_intervals.pdf"
                fig.savefig(output_dir / filename, format="pdf", bbox_inches="tight")
                plt.close(fig)
                figures.append(("Primary and structured comparison", filename))

            all_treated = [row for row in gates if row.get("group") == "All treated"]
            if all_treated:
                colors = {"structured_rfx_unit": "#1f77b4", "structured": "#d17a22"}
                markers = {"structured_rfx_unit": "o", "structured": "s"}
                rate_positions = {0.1: 0, 1.0: 1, 10.0: 2}
                offsets = {"structured_rfx_unit": -0.12, "structured": 0.12}
                fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), sharey=True)
                for rep, ax in enumerate(axes):
                    rep_rows = [row for row in all_treated if int(row["rep"]) == rep]
                    for row in rep_rows:
                        rate = as_float(row["sigma2_rate"], "sigma2_rate")
                        spec = row["spec"]
                        x = rate_positions[rate] + offsets[spec]
                        mean = as_float(row["att_pct"], "att_pct")
                        lower = as_float(row["lo95_pct"], "lo95_pct")
                        upper = as_float(row["hi95_pct"], "hi95_pct")
                        draw_interval(ax, x, mean, lower, upper, colors[spec], markers[spec],
                                      orientation="vertical", cap=0.08)
                    if not rep_rows:
                        ax.text(0.5, 0.5, "No completed fit in this bundle",
                                transform=ax.transAxes, ha="center", va="center", fontsize=8)
                    bundle_label = "prespecified bundle" if rep == 0 else "same-data computational repeat"
                    ax.set_title(f"Seed bundle {rep}\n({bundle_label})", fontsize=9)
                    ax.set_xticks([0, 1, 2], ["0.1", "1", "10"])
                    ax.set_xlabel("Global prior rate")
                    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
                    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
                    ax.spines[["top", "right"]].set_visible(False)
                axes[0].set_ylabel("All-treated ATT (%)")
                legend = [
                    Line2D([0], [0], marker=markers[spec], color=colors[spec],
                           linestyle="-", markersize=6,
                           label=("structured, unit random effects" if spec == "structured_rfx_unit"
                                  else "structured, no random effects"))
                    for spec in ("structured_rfx_unit", "structured")
                ]
                fig.legend(handles=legend, loc="upper center", ncol=2,
                           frameon=False, bbox_to_anchor=(0.5, 1.0))
                fig.suptitle("MPDTA all-treated ATT: prior-rate sensitivity", y=0.93)
                fig.text(0.01, 0.01,
                         "Bars are exported 95% posterior intervals for each fit; replications are shown separately and are not pooled.",
                         ha="left", va="bottom", fontsize=8)
                fig.tight_layout(rect=(0, 0.06, 1, 0.82))
                filename = "application_prior_sensitivity.pdf"
                fig.savefig(output_dir / filename, format="pdf", bbox_inches="tight")
                plt.close(fig)
                figures.append(("Prior-rate sensitivity by computational replication", filename))

            if unit_id:
                catt = catt_by_job[unit_id]
                group_names = {
                    1: "Small population",
                    2: "Medium population",
                    3: "High population",
                }
                values_by_group = {
                    group: [as_float(row["catt_pct"], "catt_pct") for row in catt
                            if as_int(row["lpop_group"], "lpop_group") == group]
                    for group in group_names
                }
                values = [value for group_values in values_by_group.values()
                          for value in group_values]
                if len(values) == 291 and all(values_by_group.values()):
                    low, high = min(values), max(values)
                    pad = max((high - low) * 0.04, 0.25)
                    bins = [low - pad + (high - low + 2 * pad) * index / 16
                            for index in range(17)]
                    colors = {1: "#4c78a8", 2: "#f2a541", 3: "#59a14f"}
                    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), sharex=True, sharey=True)
                    for group, ax in zip(group_names, axes):
                        group_values = values_by_group[group]
                        ax.hist(group_values, bins=bins, color=colors[group], alpha=0.8,
                                edgecolor="white", linewidth=0.7)
                        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.9)
                        ax.set_title(f"{group_names[group]}\n(n={len(group_values)} county-years)", fontsize=9)
                        ax.set_xlabel("Transformed posterior mean (%)")
                        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
                        ax.spines[["top", "right"]].set_visible(False)
                    axes[0].set_ylabel("Treated county-year rows")
                    fig.suptitle("MPDTA: county-year CATT estimates by population group", y=0.98)
                    fig.text(0.01, 0.01,
                             "Transformed posterior mean log effects, seed bundle 0, prior rate 1. Counties may contribute multiple years.",
                             ha="left", va="bottom", fontsize=8)
                    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
                    filename = "application_catt_distribution.pdf"
                    fig.savefig(output_dir / filename, format="pdf", bbox_inches="tight")
                    plt.close(fig)
                    figures.append(("County-year CATT percentage distribution", filename))

        return figures
    finally:
        if previous_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_config
        temp_config.cleanup()


def run(input_dir: Path, output_dir: Path, allow_partial: bool, plots: bool = False) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if allow_partial:
        check_partial_location(input_dir)

    manifest = read_json(MANIFEST_PATH)
    manifest_jobs = manifest.get("jobs")
    if not isinstance(manifest_jobs, list):
        raise ValueError("Manifest jobs must be a list")
    app_jobs = {
        str(job["id"]): job for job in manifest_jobs
        if job.get("phase") in EXPECTED_PHASES
    }
    phase_expected = Counter(job["phase"] for job in app_jobs.values())
    if dict(phase_expected) != EXPECTED_PHASES or len(app_jobs) != EXPECTED_APPLICATION_JOBS:
        raise ValueError(f"Manifest application grid is not the frozen 27-job design: {phase_expected}")

    status = read_json(input_dir / "status.json")
    if status.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("Collection status was produced from a different manifest")
    status_expected = status.get("phase_expected", {})
    status_completed = status.get("phase_completed", {})
    for phase, expected in EXPECTED_PHASES.items():
        if int(status_expected.get(phase, -1)) != expected:
            raise ValueError(f"Collection expected count for {phase} is not {expected}")

    missing_ids = set(status.get("missing_ids", status.get("missing_job_ids", [])))
    present_ids = set(app_jobs) - missing_ids
    completed_by_phase = {
        phase: sum(app_jobs[job_id]["phase"] == phase for job_id in present_ids)
        for phase in EXPECTED_PHASES
    }
    for phase, expected_count in completed_by_phase.items():
        if int(status_completed.get(phase, -1)) != expected_count:
            raise ValueError(f"Collection status and missing-job registry disagree for {phase}")
    missing_app_ids = set(app_jobs) - present_ids
    if missing_app_ids and not allow_partial:
        raise ValueError(
            f"Application collection is incomplete: {len(present_ids)}/{EXPECTED_APPLICATION_JOBS} "
            f"jobs present; pass --allow-partial only for validation/reuse_collection"
        )
    if not present_ids:
        raise ValueError("No application jobs are present in this collection")

    fields_by_name: dict[str, list[str]] = {}
    raw_tables: dict[str, list[dict[str, str]]] = {}
    for name in APPLICATION_TABLES:
        fields, rows = read_csv(input_dir / name)
        fields_by_name[name] = fields
        raw_tables[name] = rows
    for name in APPLICATION_TABLES[:3]:
        forbidden = FORBIDDEN_APPLICATION_COLUMNS.intersection(fields_by_name[name])
        if forbidden:
            raise ValueError(f"Empirical application table {name} has forbidden truth/error columns: {sorted(forbidden)}")

    # Validate that every present estimation fit has its three exported tables,
    # the fixed 291 treated county-year rows, and the expected analysis groups.
    fit_jobs = {job_id: job for job_id, job in app_jobs.items()
                if job_id in present_ids and job.get("kind") == "application"}
    table_rows = {
        name: [row for row in raw_tables[name] if row.get("job_id") in fit_jobs]
        for name in APPLICATION_TABLES[:3]
    }
    for name, rows in table_rows.items():
        found_ids = {row.get("job_id", "") for row in rows}
        if found_ids != set(fit_jobs):
            raise ValueError(f"{name} does not contain exactly the collected application fit IDs")

    gates_by_job: dict[str, list[dict[str, str]]] = {}
    catt_by_job: dict[str, list[dict[str, str]]] = {}
    contrasts_by_job: dict[str, list[dict[str, str]]] = {}
    for job_id in fit_jobs:
        gates = [row for row in table_rows["application_gate.csv"] if row["job_id"] == job_id]
        catt = [row for row in table_rows["application_catt.csv"] if row["job_id"] == job_id]
        contrasts = [row for row in table_rows["application_contrasts.csv"] if row["job_id"] == job_id]
        if len(gates) != 4 or {row.get("group") for row in gates} != {
            "All treated", *EXPECTED_GROUP_COUNTS.keys()
        }:
            raise ValueError(f"Application gate table for {job_id} must have the four prespecified groups")
        if len(contrasts) != 3:
            raise ValueError(f"Application contrasts table for {job_id} must have three pairwise contrasts")
        if len(catt) != 291:
            raise ValueError(f"Application CATT table for {job_id} has {len(catt)} rows, expected 291")
        if len({(row.get("unit_id"), row.get("time")) for row in catt}) != 291:
            raise ValueError(f"Application CATT table for {job_id} has duplicate county-year rows")
        if len({row.get("unit_id") for row in catt}) != 191:
            raise ValueError(f"Application CATT table for {job_id} does not have 191 counties")
        catt_group_counts = Counter(as_int(row.get("lpop_group"), "lpop_group") for row in catt)
        if catt_group_counts != {1: 72, 2: 119, 3: 100}:
            raise ValueError(f"Application CATT group counts changed for job {job_id}: {catt_group_counts}")
        gate_counts = {row["group"]: as_int(row["n_obs"], "application n_obs")
                       for row in gates if row["group"] != "All treated"}
        if gate_counts != EXPECTED_GROUP_COUNTS:
            raise ValueError(f"Application subgroup counts changed for job {job_id}: {gate_counts}")
        gates_by_job[job_id], catt_by_job[job_id], contrasts_by_job[job_id] = gates, catt, contrasts

    # The prespecified rate/specification/seed matrix is kept as individual
    # fits. These replications use the same empirical panel and are not pooled.
    seen_grid = set()
    for job_id, job in fit_jobs.items():
        key = (float(job["sigma2_rate"]), str(job["spec"]), int(job["rep"]))
        if key in seen_grid:
            raise ValueError(f"Duplicate rate/specification/replication fit: {key}")
        seen_grid.add(key)
        if key[0] not in EXPECTED_RATES or key[1] not in ESTIMATOR_SPECS or key[2] not in (0, 1, 2):
            raise ValueError(f"Unexpected application sensitivity cell: {key}")
    if not missing_app_ids:
        expected_grid = {
            (rate, spec, rep)
            for rate in EXPECTED_RATES
            for spec in ESTIMATOR_SPECS
            for rep in (0, 1, 2)
        }
        if seen_grid != expected_grid:
            raise ValueError("Complete application collection does not contain the 3x2x3 estimator grid")

    # The pretrend decision and estimate rows are retained verbatim, with their
    # manifest job kind restored because collect writes this as a derived table.
    per_rep_rows = raw_tables["per_replication.csv"]
    pretrend_ids = {job_id for job_id in present_ids
                    if app_jobs[job_id].get("kind") == "application_pretrend"}
    pretrend_rows = [row for row in per_rep_rows if row.get("job_id") in pretrend_ids]
    if {row.get("job_id") for row in pretrend_rows} != pretrend_ids:
        raise ValueError("Per-replication table is missing an application pretrend fit")
    if pretrend_rows:
        if not {"estimand_type", "estimand_id", "p_bayes", "q025", "q975"}.issubset(
            fields_by_name["per_replication.csv"]
        ):
            raise ValueError("Application pretrend rows lack decision or interval columns")
        unique_decisions = set()
        for row in pretrend_rows:
            key = (row.get("job_id"), row.get("estimand_type"), row.get("estimand_id"))
            if key in unique_decisions:
                raise ValueError(f"Duplicate application pretrend row: {key}")
            unique_decisions.add(key)
    # The campaign collector uses one wide simulation schema, so application
    # pretrend rows carry empty truth and surface-metric columns. Drop those
    # columns here to keep this empirical report free of truth/error/coverage
    # quantities.
    pretrend_fields = [
        field for field in fields_by_name["per_replication.csv"]
        if field not in FORBIDDEN_APPLICATION_COLUMNS and not field.startswith("surf_")
    ]
    pretrend_rows = [{field: row.get(field, "") for field in pretrend_fields}
                     for row in pretrend_rows]
    pretrend_rows = add_job_metadata(pretrend_rows, app_jobs, expected_kind="application_pretrend")

    enriched_tables: dict[str, list[dict[str, str]]] = {}
    for filename in APPLICATION_TABLES[:3]:
        enriched_tables[filename] = add_job_metadata(table_rows[filename], fit_jobs)

    output_dir.mkdir(parents=True, exist_ok=True)
    primary_id_by_role = {
        "Primary: unit random effects": next(
            (job_id for job_id, job in fit_jobs.items()
             if job["spec"] == "structured_rfx_unit" and float(job["sigma2_rate"]) == 1.0 and int(job["rep"]) == 0),
            None,
        ),
        "Comparison: no random effects": next(
            (job_id for job_id, job in fit_jobs.items()
             if job["spec"] == "structured" and float(job["sigma2_rate"]) == 1.0 and int(job["rep"]) == 0),
            None,
        ),
    }
    if any(job_id is None for job_id in primary_id_by_role.values()):
        if not allow_partial:
            raise ValueError("Prespecified rep=0, prior-rate=1 primary/comparison fits are missing")
    primary_rows = []
    for role, job_id in primary_id_by_role.items():
        if job_id is None:
            continue
        for row in enriched_tables["application_gate.csv"]:
            if row["job_id"] == job_id:
                primary_rows.append({**row, "analysis_role": role})

    primary_fields = list(dict.fromkeys([
        *fields_by_name["application_gate.csv"], "job_phase", "job_kind",
        "sigma2_shape", "sigma2_rate", "setting", "analysis_role",
    ]))
    write_csv(output_dir / "application_primary_estimates.csv", primary_rows, primary_fields)

    # One file per computational replication, keeping all estimator and prior
    # settings separate for review. Raw numeric fields remain the collector's
    # original CSV strings, including exported log and percent bounds.
    counts_by_rep: dict[int, dict[str, int]] = {}
    for rep in (0, 1, 2):
        rep_jobs = {job_id for job_id, job in fit_jobs.items() if int(job["rep"]) == rep}
        counts_by_rep[rep] = {
            "fits": len(rep_jobs),
            "gate_rows": sum(row["job_id"] in rep_jobs for row in enriched_tables["application_gate.csv"]),
            "catt_rows": sum(row["job_id"] in rep_jobs for row in enriched_tables["application_catt.csv"]),
            "contrast_rows": sum(row["job_id"] in rep_jobs for row in enriched_tables["application_contrasts.csv"]),
        }
        for kind, filename in (
            ("application_gate.csv", f"application_gates_rep{rep}.csv"),
            ("application_catt.csv", f"application_catt_rep{rep}.csv"),
            ("application_contrasts.csv", f"application_contrasts_rep{rep}.csv"),
        ):
            rep_rows = [row for row in enriched_tables[kind]
                        if int(row["rep"]) == rep]
            original_fields = fields_by_name[kind]
            write_enriched(output_dir / filename, rep_rows, original_fields)
    pretrend_fields = list(dict.fromkeys([
        *pretrend_fields, "job_phase", "job_kind",
        "sigma2_shape", "sigma2_rate", "setting",
    ]))
    write_csv(output_dir / "application_pretrend_diagnostics.csv", pretrend_rows, pretrend_fields)
    figures_written = (
        make_plots(
            output_dir,
            primary_id_by_role,
            gates_by_job,
            catt_by_job,
            enriched_tables["application_gate.csv"],
        )
        if plots else []
    )

    report_status = (
        f"**Partial validation only:** {len(present_ids)}/{EXPECTED_APPLICATION_JOBS} "
        "application and prior-sensitivity jobs are present. This report is not a complete "
        "application result and is not for manuscript inference."
        if missing_app_ids else
        f"**Application results complete:** all {EXPECTED_APPLICATION_JOBS}/{EXPECTED_APPLICATION_JOBS} "
        "application and prior-sensitivity jobs are present. This completeness statement covers "
        "the application jobs only."
    )
    if missing_app_ids and not allow_partial:
        raise ValueError("Missing application jobs cannot be reported as complete")

    all_completed = int(status.get("completed_total", 0))
    all_expected = int(status.get("expected_total", 0))
    primary_gate = []
    for role, job_id in primary_id_by_role.items():
        if job_id:
            for row in gates_by_job[job_id]:
                primary_gate.append((role, row))
    primary_report_rows = []
    for role, row in primary_gate:
        primary_report_rows.append([
            role, row["group"], row["n_obs"], row["att_log"],
            f"{row['lo95_log']} to {row['hi95_log']}", row["att_pct"],
            f"{row['lo95_pct']} to {row['hi95_pct']}", row["p_bayes"],
        ])

    sensitivity_report_rows = []
    for row in enriched_tables["application_gate.csv"]:
        if row.get("group") != "All treated":
            continue
        job = fit_jobs[row["job_id"]]
        sensitivity_report_rows.append([
            row["sigma2_rate"], row["spec"], row["rep"], row["att_log"],
            f"{row['lo95_log']} to {row['hi95_log']}", row["att_pct"],
            f"{row['lo95_pct']} to {row['hi95_pct']}", row["p_bayes"],
        ])
    sensitivity_report_rows.sort(key=lambda row: (float(row[0]), row[1], int(row[2])))

    decision_ids = {("PRE", "any"), ("PRE", "any_bonf"),
                    ("PRE_SUBC", "lpop_any"), ("PRE_SUBC", "lpop_any_bonf")}
    decisions = [row for row in pretrend_rows
                 if (row.get("estimand_type"), row.get("estimand_id")) in decision_ids]
    decisions.sort(key=lambda row: (
        float(row["sigma2_rate"]), int(row["rep"]), row["estimand_type"], row["estimand_id"]
    ))
    decision_table_rows = [[
        row["sigma2_rate"], row["rep"], row["estimand_type"], row["estimand_id"],
        row["p_bayes"],
    ] for row in decisions]

    report_lines = [
        "# Empirical application analysis",
        "",
        report_status,
        "",
        f"The collection status is {all_completed}/{all_expected} total non-main jobs. "
        "Application completeness is reported separately because this report uses only the "
        "27 empirical application and prior-sensitivity jobs.",
        "",
        "The application uses the 500-county MPDTA panel. The application estimand tables "
        "contain 291 treated county-year CATT rows and 191 distinct treated counties per "
        "estimation fit. The treated county-year group counts are 72 small-population, "
        "119 medium-population, and 100 high-population rows.",
        "",
        "## Prespecified estimates",
        "",
        "The primary fit is `structured_rfx_unit` at prior rate 1, replication 0. "
        "`structured` at the same rate and replication is the no-random-effects comparison. "
        "Log-point and percentage means and interval bounds below are copied from the "
        "collector's exported fields. The accompanying CSV retains their original precision.",
        "",
        markdown_table(
            ["Fit", "Group", "Treated county-years", "ATT (log)", "95% interval (log)",
             "ATT (%)", "95% interval (%)", "p_bayes"],
            primary_report_rows,
        ),
        "",
        "## Prior-rate sensitivity and computational repeats",
        "",
        "The table keeps the all-treated estimate for each rate, specification, and replication "
        "separate. Replications 1 and 2 reuse the same observed panel and are computational "
        "repeats, not independent samples. We do not pool posterior quantiles or average "
        "confidence or credible intervals across them.",
        "",
        markdown_table(
            ["Prior rate", "Specification", "Replication", "ATT (log)",
             "95% interval (log)", "ATT (%)", "95% interval (%)", "p_bayes"],
            sensitivity_report_rows,
        ),
        "",
        "The seed-specific CSVs contain all three treated groups, all 291 county-year CATT rows, "
        "and all three group contrasts for each available fit. They preserve the exported numeric "
        "strings without recomputing intervals.",
        "",
        "The frozen CATT export defines `catt_pct` as `100 * (exp(catt_log) - 1)`, "
        "where `catt_log` is the posterior mean log effect. Its distribution plot therefore "
        "shows transformed posterior means. The GATE/ATT percentage summaries instead "
        "transform the group-average effect draw by draw before taking the posterior mean. "
        "These are the original export conventions and are not interchangeable.",
        "",
        "## Application pretrend decisions",
        "",
        "The exported `application_pretrend` rows are retained in "
        "`application_pretrend_diagnostics.csv`, including event-time and subgroup summaries, "
        "95% bounds, and the pipeline's `p_bayes` decision rows (`any`, `any_bonf`, "
        "`lpop_any`, and `lpop_any_bonf`). The table below displays those original decision "
        "values; no new decision rule is applied here.",
        "",
        markdown_table(
            ["Prior rate", "Replication", "Estimand type", "Decision row", "p_bayes"],
            decision_table_rows,
        ) if decision_table_rows else "No application pretrend decision rows are present in this collection.",
        "",
        "## Reproducibility counts",
        "",
        markdown_table(
            ["Phase", "Expected jobs", "Present jobs"],
            [[phase, str(expected), str(completed_by_phase[phase])]
             for phase, expected in EXPECTED_PHASES.items()],
        ),
        "",
        f"Estimation fits present: {len(fit_jobs)} of 18. Application pretrend fits present: "
        f"{len(pretrend_ids)} of 9. Primary/comparison CATT rows: "
        f"{sum(291 for job_id in primary_id_by_role.values() if job_id)}. "
        f"Pretrend rows retained: {len(pretrend_rows)}; decision rows shown: {len(decisions)}.",
        "",
        "## Files",
        "",
        "`application_primary_estimates.csv` contains the prespecified primary and comparison "
        "group estimates. `application_gates_rep0.csv` through `application_gates_rep2.csv`, "
        "the corresponding `application_catt_rep*.csv` and `application_contrasts_rep*.csv` "
        "files retain seed-specific estimates. `application_pretrend_diagnostics.csv` contains "
        "the pretrend estimates and original decision rows.",
    ]
    if figures_written:
        report_lines.extend([
            "",
            "## Figures",
            "",
            "The vector PDF figures use exported posterior means and interval bounds. The CATT "
            "histograms describe 291 treated county-year estimates; counties may recur across years.",
            "",
            *[f"- [{label}]({filename})" for label, filename in figures_written],
        ])
    report_lines.append("")
    report_path = output_dir / "application_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "report": str(report_path),
        "application_jobs_present": len(present_ids),
        "application_jobs_expected": EXPECTED_APPLICATION_JOBS,
        "application_fit_jobs_present": len(fit_jobs),
        "application_pretrend_jobs_present": len(pretrend_ids),
        "partial": bool(missing_app_ids),
        "figures": [filename for _, filename in figures_written],
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=PACKAGE / "Results" / "application_collected",
        help="directory created by helpers/collect.py (default: Results/application_collected)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent,
        help="directory for the report and manuscript-review CSVs",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="validation only; accepted only for validation/reuse_collection",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="also write standalone vector PDF figures (requires Matplotlib)",
    )
    args = parser.parse_args()
    try:
        result = run(args.input, args.output_dir, args.allow_partial, args.plots)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
