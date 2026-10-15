#!/usr/bin/env python3
"""Prepare and, only with ``--apply``, integrate the repaired MPDTA results.

The default is a write-free dry run. This script consumes only the complete
metrics collection, never fits a model, and fails closed unless all 27
application and application-sensitivity jobs have validated fit records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PACKAGE / "analysis"
EXPECTED_PHASES = {"application": 9, "application_sensitivity": 18}
EXPECTED_APPLICATION_JOBS = 27
EXPECTED_RATES = (0.1, 1.0, 10.0)
EXPECTED_SPECS = ("structured_rfx_unit", "structured")
EXPECTED_REPS = (0, 1, 2)
EXPECTED_GROUPS = (
    "All treated", "Small population", "Medium population", "High population",
)
EXPECTED_GROUP_COUNTS = {
    "All treated": 291,
    "Small population": 72,
    "Medium population": 119,
    "High population": 100,
}
FORBIDDEN_EMPIRICAL_FIELDS = {
    "true", "error", "absolute_error", "squared_error", "coverage",
}
PRETREND_DECISION_IDS = ("slope", "any", "any_bonf")
PRETREND_CUTOFF = 0.025
OUTPUT_DIR_NAME = "empirical_repaired_20261014"
BACKUP_DIR_NAME = "application_update_20261014/before"


class IntegrationError(RuntimeError):
    """Raised when source evidence or manuscript anchors are insufficient."""


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "JBES_Submission" / "main.tex").is_file():
            return parent
    raise IntegrationError("Cannot locate repository containing JBES_Submission/main.tex")


REPO = repo_root()
DEFAULT_INPUT = PACKAGE / "Results" / "application_collected"
CENTRAL_OUTPUT = PACKAGE.parents[1] / "Results" / OUTPUT_DIR_NAME
PAPER = REPO / "JBES_Submission"
MAIN_TEX = PAPER / "main.tex"
SUPP_TEX = PAPER / "supplementary_material.tex"
PAPER_FIGURE = PAPER / "images" / "fig_mpdta_gate_kde.pdf"
PAPER_PRIOR_TEX = PAPER / "sim_tables" / "tab_application_prior_sensitivity.tex"
PAPER_PRETREND_TEX = PAPER / "sim_tables" / "tab_application_pretrend.tex"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrationError(f"Expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise IntegrationError(f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise IntegrationError(f"Cannot read {path}: {exc}") from exc


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_float(value: object, name: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrationError(f"Invalid {name}: {value!r}") from exc
    if not math.isfinite(number):
        raise IntegrationError(f"Non-finite {name}: {value!r}")
    return number


def expected_job_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise IntegrationError("Manifest has no jobs list")
    application_jobs = {
        str(job.get("id")): job for job in jobs
        if job.get("phase") in EXPECTED_PHASES
    }
    counts = Counter(job["phase"] for job in application_jobs.values())
    if len(application_jobs) != EXPECTED_APPLICATION_JOBS or dict(counts) != EXPECTED_PHASES:
        raise IntegrationError(
            f"Frozen application manifest grid mismatch: {dict(counts)}; expected {EXPECTED_PHASES}"
        )
    if any(not job_id or job_id == "None" for job_id in application_jobs):
        raise IntegrationError("Manifest has an application job without an ID")
    return application_jobs


def package_identity() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the frozen manifest, package, helpers, assignments, and source hashes."""
    helper_path = PACKAGE / "helpers" / "common.py"
    spec = importlib.util.spec_from_file_location("didbcf_nonmain_common", helper_path)
    if spec is None or spec.loader is None:
        raise IntegrationError(f"Cannot load package identity helper: {helper_path}")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    try:
        package, manifest = common.verify_package()
    except Exception as exc:
        raise IntegrationError(f"Frozen package/source validation failed: {exc}") from exc
    return package, manifest


def collection_summary(input_dir: Path) -> dict[str, Any]:
    """Validate completeness counts and validated fit-record IDs before parsing tables."""
    package, manifest = package_identity()
    jobs = expected_job_map(manifest)
    status_path = input_dir / "status.json"
    provenance_path = input_dir / "provenance.json"
    if not status_path.is_file():
        raise IntegrationError(f"Missing collected status manifest: {status_path}")
    if not provenance_path.is_file():
        raise IntegrationError(f"Missing collection provenance and validated source IDs: {provenance_path}")
    status = read_json(status_path)
    provenance = read_json(provenance_path)
    if status.get("manifest_hash") != manifest.get("manifest_hash"):
        raise IntegrationError("Collected status manifest hash does not match the frozen campaign")
    if provenance.get("package_hash") != package.get("package_hash"):
        raise IntegrationError("Collection provenance does not match the validated package hash")
    for field in ("application_truth_or_error_metrics_created", "posterior_arrays_needed_for_collection"):
        if status.get(field) is not False:
            raise IntegrationError(f"Unexpected collection status flag {field}={status.get(field)!r}")

    expected = status.get("phase_expected", {})
    completed = status.get("phase_completed", {})
    missing_ids = status.get("missing_ids", status.get("missing_job_ids"))
    if not isinstance(missing_ids, list) or any(not isinstance(item, str) for item in missing_ids):
        raise IntegrationError("Collection status lacks a validated missing-job ID list")
    missing = set(missing_ids)
    all_job_ids = {str(job["id"]) for job in manifest["jobs"]}
    if not missing.issubset(all_job_ids):
        raise IntegrationError("Collection missing-job list contains unknown IDs")
    expected_by_phase = manifest.get("phase_expected", {})
    for phase in EXPECTED_PHASES:
        frozen_n = int(expected_by_phase.get(phase, -1))
        status_n = int(expected.get(phase, -1))
        status_done = int(completed.get(phase, -1))
        expected_n = EXPECTED_PHASES[phase]
        found_n = sum(jobs[job_id]["phase"] == phase for job_id in jobs if job_id not in missing)
        if frozen_n != expected_n or status_n != expected_n or status_done != found_n:
            raise IntegrationError(
                f"Collected {phase} counts are inconsistent: manifest={frozen_n}, "
                f"status={status_done}/{status_n}, validated IDs={found_n}/{expected_n}"
            )

    records = provenance.get("fit_record_hashes")
    if not isinstance(records, dict):
        raise IntegrationError("Collection provenance has no fit-record hash map")
    completed_total = int(status.get("completed_total", -1))
    if completed_total != len(records):
        raise IntegrationError(
            f"Validated fit-record IDs ({len(records)}) disagree with completed_total ({completed_total})"
        )
    for job_id, record_hash in records.items():
        if job_id not in all_job_ids or not isinstance(record_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", record_hash):
            raise IntegrationError(f"Invalid validated fit-record ID/hash: {job_id!r}")
    missing_app = sorted(set(jobs) & missing)
    completed_app = sorted(set(jobs) - missing)
    unvalidated_app = sorted(set(jobs) - set(records))
    if unvalidated_app:
        raise IntegrationError(
            "Application IDs are absent from validated provenance fit_record_hashes: "
            + ", ".join(unvalidated_app)
        )
    phase_completed = {
        phase: sum(jobs[job_id]["phase"] == phase for job_id in completed_app)
        for phase in EXPECTED_PHASES
    }
    return {
        "package": package,
        "manifest": manifest,
        "jobs": jobs,
        "status": status,
        "provenance": provenance,
        "phase_completed": phase_completed,
        "missing_application_ids": missing_app,
        "completed_application_ids": completed_app,
        "validated_application_record_hashes": {
            job_id: records[job_id] for job_id in sorted(jobs) if job_id in records
        },
        "complete": not missing_app and phase_completed == EXPECTED_PHASES,
    }


def load_application_report_module():
    report_path = ANALYSIS_DIR / "application_report.py"
    spec = importlib.util.spec_from_file_location("didbcf_application_report_validator", report_path)
    if spec is None or spec.loader is None:
        raise IntegrationError(f"Cannot load application report validator: {report_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validated_report(input_dir: Path, summary: dict[str, Any], work_dir: Path) -> dict[str, Path]:
    if not summary["complete"]:
        phase_counts = summary["phase_completed"]
        missing = len(summary["missing_application_ids"])
        raise IntegrationError(
            "Application collection is incomplete: "
            f"{sum(phase_counts.values())}/{EXPECTED_APPLICATION_JOBS} present "
            f"({phase_counts}), {missing} application job IDs missing. "
            "No manuscript or results files were written."
        )
    module = load_application_report_module()
    report_out = work_dir / "validated_report"
    try:
        result = module.run(input_dir, report_out, allow_partial=False, plots=False)
    except Exception as exc:
        raise IntegrationError(f"Existing compact-output validator rejected the collection: {exc}") from exc
    if result.get("partial") or result.get("application_jobs_present") != EXPECTED_APPLICATION_JOBS:
        raise IntegrationError("Existing compact-output validator did not confirm all 27 application jobs")
    if result.get("application_fit_jobs_present") != 18 or result.get("application_pretrend_jobs_present") != 9:
        raise IntegrationError("Existing compact-output validator found insufficient estimator/pretrend rows")
    names = (
        "application_primary_estimates.csv",
        "application_gates_rep0.csv", "application_gates_rep1.csv", "application_gates_rep2.csv",
        "application_catt_rep0.csv", "application_catt_rep1.csv", "application_catt_rep2.csv",
        "application_contrasts_rep0.csv", "application_contrasts_rep1.csv", "application_contrasts_rep2.csv",
        "application_pretrend_diagnostics.csv",
    )
    files = {name: report_out / name for name in names}
    absent = [str(path) for path in files.values() if not path.is_file()]
    if absent:
        raise IntegrationError("Validated report omitted required compact outputs: " + ", ".join(absent))
    return files


def format_fixed(raw: str, places: int = 2) -> str:
    """Format an exported number for manuscript display, without changing artifacts."""
    value = Decimal(raw)
    quantum = Decimal("1").scaleb(-places)
    return f"{value.quantize(quantum, rounding=ROUND_HALF_UP):.{places}f}"


def format_p(raw: str) -> str:
    value = parse_float(raw, "p_bayes")
    if value < 0.001:
        return "\\ensuremath{<0.001}"
    return format_fixed(raw, 3)


def spec_tex(spec: str) -> str:
    if spec == "structured_rfx_unit":
        return r"Unit intercept"
    if spec == "structured":
        return r"No unit intercept"
    raise IntegrationError(f"Unknown application specification {spec!r}")


def ordered_gate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    group_index = {group: index for index, group in enumerate(EXPECTED_GROUPS)}
    return sorted(rows, key=lambda row: group_index.get(row.get("group", ""), 99))


def exact_primary_rows(files: dict[str, Path]) -> dict[str, list[dict[str, str]]]:
    _, primary = read_csv(files["application_primary_estimates.csv"])
    if len(primary) != 8:
        raise IntegrationError(f"Primary/comparison table has {len(primary)} rows, expected 8")
    by_role: dict[str, list[dict[str, str]]] = {
        "Primary: unit random effects": [],
        "Comparison: no random effects": [],
    }
    for row in primary:
        role = row.get("analysis_role", "")
        if role not in by_role:
            raise IntegrationError(f"Unexpected application analysis role {role!r}")
        by_role[role].append(row)
    for role, rows in by_role.items():
        if {row.get("group") for row in rows} != set(EXPECTED_GROUPS):
            raise IntegrationError(f"{role} does not contain the four required groups")
        if len({(row.get("sigma2_rate"), row.get("rep")) for row in rows}) != 1:
            raise IntegrationError(f"{role} mixes prior rates or computational bundles")
    return {role: ordered_gate_rows(rows) for role, rows in by_role.items()}


def all_seed_rows(files: dict[str, Path]) -> dict[int, dict[str, list[dict[str, str]]]]:
    output: dict[int, dict[str, list[dict[str, str]]]] = {}
    for rep in EXPECTED_REPS:
        _, gates = read_csv(files[f"application_gates_rep{rep}.csv"])
        _, catts = read_csv(files[f"application_catt_rep{rep}.csv"])
        _, contrasts = read_csv(files[f"application_contrasts_rep{rep}.csv"])
        if len(gates) != 24 or len(catts) != 1746 or len(contrasts) != 18:
            raise IntegrationError(
                f"Rep {rep} compact row counts are insufficient: gates={len(gates)}, "
                f"CATT={len(catts)}, contrasts={len(contrasts)}; expected 24/1746/18"
            )
        output[rep] = {"gates": gates, "catt": catts, "contrasts": contrasts}
    return output


def pretrend_rows(file_path: Path) -> list[dict[str, str]]:
    _, rows = read_csv(file_path)
    fit_rows = [row for row in rows if row.get("job_kind") == "application_pretrend"]
    keys = {(row.get("sigma2_rate"), row.get("rep"), row.get("estimand_type"), row.get("estimand_id"))
            for row in fit_rows}
    output: list[dict[str, str]] = []
    for rate in EXPECTED_RATES:
        for rep in EXPECTED_REPS:
            rate_string = str(rate)
            def find(estimand_id: str) -> dict[str, str]:
                matches = [row for row in fit_rows
                           if parse_float(row.get("sigma2_rate", "nan"), "sigma2_rate") == rate
                           and int(row.get("rep", "-1")) == rep
                           and row.get("estimand_type") == "PRE"
                           and row.get("estimand_id") == estimand_id]
                if len(matches) != 1:
                    raise IntegrationError(
                        f"Expected one empirical pretrend {estimand_id} row for rate={rate}, rep={rep}; found {len(matches)}"
                    )
                return matches[0]
            slope, any_row, bonf = find("slope"), find("any"), find("any_bonf")
            for row in (slope, any_row, bonf):
                if not row.get("p_bayes"):
                    raise IntegrationError(f"Missing pretrend p_bayes for rate={rate}, rep={rep}")
            if not all(slope.get(key) for key in ("post_mean", "q025", "q975")):
                raise IntegrationError(f"Missing slope estimate/interval for rate={rate}, rep={rep}")
            output.append({
                "sigma2_rate": slope["sigma2_rate"],
                "rep": slope["rep"],
                "slope_post_mean": slope["post_mean"],
                "slope_q025": slope["q025"],
                "slope_q975": slope["q975"],
                "slope_p_bayes_one_sided": slope["p_bayes"],
                "any_p_bayes_one_sided": any_row["p_bayes"],
                "any_bonf_p_bayes_one_sided": bonf["p_bayes"],
                "slope_decision_at_0.025": "reject" if parse_float(slope["p_bayes"], "slope p") < PRETREND_CUTOFF else "do_not_reject",
                "any_decision_at_0.025": "reject" if parse_float(any_row["p_bayes"], "any p") < PRETREND_CUTOFF else "do_not_reject",
                "any_bonf_decision_at_0.025": "reject" if parse_float(bonf["p_bayes"], "any_bonf p") < PRETREND_CUTOFF else "do_not_reject",
            })
    if len(output) != 9:
        raise IntegrationError(f"Empirical pretrend table has {len(output)} rows, expected 9")
    return output


def gate_tex_table(primary_by_role: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Repaired-engine DiD-BCF estimates for the MPDTA treated county-years. The unit-intercept specification is primary; the model without unit intercepts is a comparison. Both use prior rate 1 and computational bundle 0. Effects and equal-tailed 95\% posterior intervals are on the percentage scale.}",
        r"\label{tab:mpdta_gate}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Specification & Group & $n$ (county-years) & Effect & 95\% CI & $p_{\mathrm{Bayes}}$ \\",
        r"\midrule",
    ]
    role_order = ("Primary: unit random effects", "Comparison: no random effects")
    for role in role_order:
        for row in primary_by_role[role]:
            lines.append(
                f"{spec_tex(row['spec'])} & {row['group']} & {row['n_obs']} & "
                f"${format_fixed(row['att_pct'])}\\%$ & "
                f"$[{format_fixed(row['lo95_pct'])}\\%,\\ {format_fixed(row['hi95_pct'])}\\%]$ & "
                f"{format_p(row['p_bayes'])} \\\\"
            )
        if role != role_order[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def prior_sensitivity_tex(rows: list[dict[str, str]]) -> str:
    all_treated = [row for row in rows if row.get("group") == "All treated"]
    expected_keys = {
        (rate, spec, rep) for rate in EXPECTED_RATES
        for spec in EXPECTED_SPECS for rep in EXPECTED_REPS
    }
    found = {
        (parse_float(row["sigma2_rate"], "sigma2_rate"), row["spec"], int(row["rep"]))
        for row in all_treated
    }
    if found != expected_keys or len(all_treated) != 18:
        raise IntegrationError(f"Prior sensitivity requires all 18 all-treated rows; found {len(all_treated)}")
    all_treated.sort(key=lambda row: (parse_float(row["sigma2_rate"], "rate"), row["spec"], int(row["rep"])))
    lines = [
        r"\begin{table}[htbp]", r"\centering\scriptsize",
        r"\caption{All-treated MPDTA estimates across the working-error prior-rate sensitivity grid. Bundle 0 is prespecified; bundles 1 and 2 are separate computational repeats on the same panel. Intervals are not pooled.}",
        r"\label{tab:application_prior_sensitivity}",
        r"\begin{tabular}{rllrrr}", r"\toprule",
        r"Prior rate & Specification & Bundle & ATT (\%) & 95\% interval (\%) & $p_{\mathrm{Bayes}}$ \\",
        r"\midrule",
    ]
    for row in all_treated:
        rate = format_fixed(row["sigma2_rate"], 1)
        rep = int(row["rep"])
        bundle = str(rep)
        lines.append(
            f"{rate} & {spec_tex(row['spec'])} & {bundle} & ${format_fixed(row['att_pct'])}\\%$ & "
            f"$[{format_fixed(row['lo95_pct'])}\\%,\\ {format_fixed(row['hi95_pct'])}\\%]$ & "
            f"{format_p(row['p_bayes'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def pretrend_tex(rows: list[dict[str, str]]) -> str:
    if len(rows) != 9:
        raise IntegrationError(f"Pretrend table requires nine rows, found {len(rows)}")
    lines = [
        r"\begin{table}[htbp]", r"\centering\scriptsize",
        r"\caption{Empirical MPDTA pretrend slope and original any-lead decision summaries. Slope intervals are central 95\% posterior intervals; the slope, any-lead, and Bonferroni-any $p_{\mathrm{Bayes}}$ values are one-sided posterior-tail probabilities evaluated against the prespecified 0.025 cutoff.}",
        r"\label{tab:application_pretrend}",
        r"\begin{tabular}{rrrrrrr}", r"\toprule",
        r"Prior rate & Bundle & Slope & 95\% interval & $p_{\mathrm{Bayes}}$ (slope) & Any & Bonferroni any \\",
        r"\midrule",
    ]
    for row in rows:
        rep = int(row["rep"])
        bundle = "prespecified" if rep == 0 else f"repeat {rep}"
        slope_decision = r"reject" if row["slope_decision_at_0.025"] == "reject" else r"do not reject"
        any_decision = r"reject" if row["any_decision_at_0.025"] == "reject" else r"do not reject"
        bonf_decision = r"reject" if row["any_bonf_decision_at_0.025"] == "reject" else r"do not reject"
        lines.append(
            f"{format_fixed(row['sigma2_rate'], 1)} & {rep} & ${format_fixed(row['slope_post_mean'], 4)}$ & "
            f"$[{format_fixed(row['slope_q025'], 4)},\\ {format_fixed(row['slope_q975'], 4)}]$ & "
            f"{format_p(row['slope_p_bayes_one_sided'])} & "
            f"{format_p(row['any_p_bayes_one_sided'])} & "
            f"{format_p(row['any_bonf_p_bayes_one_sided'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def zero_crossing(row: dict[str, str]) -> bool:
    lower, upper = parse_float(row["lo95_pct"], "lo95_pct"), parse_float(row["hi95_pct"], "hi95_pct")
    return lower <= 0.0 <= upper


def interval_cell_label(row: dict[str, str]) -> str:
    rate = format_fixed(row["sigma2_rate"], 1)
    spec = "unit-RE" if row["spec"] == "structured_rfx_unit" else "no-RE"
    return f"rate {rate}/{spec}/bundle {row['rep']}"


def application_sensitivity_summary(all_seeds: dict[int, dict[str, list[dict[str, str]]]]) -> str:
    all_rows = [row for by_rep in all_seeds.values() for row in by_rep["gates"]
                if row.get("group") == "All treated"]
    if len(all_rows) != 18:
        raise IntegrationError(f"Sensitivity summary needs 18 cells, found {len(all_rows)}")
    unit_rows = [row for row in all_rows if row.get("spec") == "structured_rfx_unit"]
    unit_points = [parse_float(row["att_pct"], "att_pct") for row in unit_rows]
    crossing_rows = [row for row in all_rows if zero_crossing(row)]
    crossings = [interval_cell_label(row) for row in crossing_rows]
    if crossings:
        if (len(crossing_rows) == 3 and all(row["spec"] == "structured" for row in crossing_rows)
                and all(parse_float(row["sigma2_rate"], "rate") == 10 for row in crossing_rows)):
            crossing_text = (
                "All nine unit-intercept intervals exclude zero. The no-unit-intercept intervals "
                "include zero at prior rate 10 in all three bundles; its other six intervals exclude zero."
            )
        else:
            crossing_text = "The intervals that include zero are " + "; ".join(crossings) + "."
    else:
        crossing_text = "None of the 18 exported intervals includes zero."
    return (
        "The prior-rate sensitivity keeps the three computational bundles and two "
        "specifications separate (see the supplementary application sensitivity and pretrend tables). "
        "Across the nine unit-random-effects "
        f"rate-by-bundle settings, all-treated posterior means range from "
        f"${min(unit_points):.2f}\\%$ to ${max(unit_points):.2f}\\%$. {crossing_text} "
        "Bundles 1 and 2 are computational repeats on the same empirical panel. "
        "No posterior quantiles or intervals are pooled across bundles."
    )


def empirical_pretrend_summary(rows: list[dict[str, str]]) -> str:
    matches = [row for row in rows
               if parse_float(row["sigma2_rate"], "pretrend rate") == 1.0 and int(row["rep"]) == 0]
    if len(matches) != 1:
        raise IntegrationError(f"Expected one primary empirical pretrend row, found {len(matches)}")
    row = matches[0]

    def decision(p: str) -> str:
        return "rejects" if parse_float(p, "pretrend p_bayes") < PRETREND_CUTOFF else "does not reject"

    raw_rejects = sum(parse_float(r["any_p_bayes_one_sided"], "any p") < PRETREND_CUTOFF for r in rows)
    adjusted_rejects = sum(parse_float(r["any_bonf_p_bayes_one_sided"], "adjusted p") < PRETREND_CUTOFF for r in rows)
    return (
        "The primary empirical pretrend fit (prior rate 1, computational bundle 0) estimates "
        f"a slope of ${format_fixed(row['slope_post_mean'], 4)}$ with central 95\\% posterior "
        f"interval $[{format_fixed(row['slope_q025'], 4)},\\ {format_fixed(row['slope_q975'], 4)}]$ "
        f"and one-sided $p_{{\\mathrm{{Bayes}}}}={format_p(row['slope_p_bayes_one_sided'])}$. "
        f"Under the prespecified one-sided cutoff $p_{{\\mathrm{{Bayes}}}}<0.025$, the slope test "
        f"{decision(row['slope_p_bayes_one_sided'] )}. The pipeline's raw any-lead decision "
        f"has $p_{{\\mathrm{{Bayes}}}}={format_p(row['any_p_bayes_one_sided'])}$ and its "
        f"Bonferroni-any decision has $p_{{\\mathrm{{Bayes}}}}={format_p(row['any_bonf_p_bayes_one_sided'])}$ "
        f"and {decision(row['any_bonf_p_bayes_one_sided'])} at the same cutoff. "
        f"Across all nine prior-rate/bundle fits, the unadjusted any-lead test rejects in {raw_rejects} "
        f"and the Bonferroni-adjusted test in {adjusted_rejects}. "
        "These one-sided diagnostic probabilities are distinct from the two-sided application-effect summaries. "
        "Nonrejection does not establish parallel trends."
    )


def main_application_prose(primary: dict[str, list[dict[str, str]]],
                            all_seeds: dict[int, dict[str, list[dict[str, str]]]],
                            pretrend: list[dict[str, str]]) -> tuple[str, str, str]:
    unit = primary["Primary: unit random effects"]
    no_unit = primary["Comparison: no random effects"]
    unit_all = next(row for row in unit if row["group"] == "All treated")
    no_unit_all = next(row for row in no_unit if row["group"] == "All treated")
    unit_cross = [row["group"] for row in unit if zero_crossing(row)]
    no_unit_cross = [row["group"] for row in no_unit if zero_crossing(row)]
    if unit_cross:
        primary_subgroup_text = "The primary 95\\% intervals include zero for " + ", ".join(unit_cross) + "."
    else:
        primary_subgroup_text = "None of the primary group intervals includes zero."
    if no_unit_cross:
        no_unit_text = "In the no-random-effects fit, the subgroup intervals include zero for " + ", ".join(no_unit_cross) + "."
    else:
        no_unit_text = "None of the no-random-effects subgroup intervals includes zero."
    intro = (
        "We refit the \\texttt{mpdta} application with the structured DiD-BCF used in the "
        "simulations ($a(G_i,X_i)+b(t,X_i)$ baseline, \\texttt{lpop} as modifier, "
        "$\\taug$ intercept, and XBCF warm start; 50 GFR sweeps and 500 retained draws "
        "per chain, thinning five, across three chains). The prespecified primary estimate is the "
        "unit-random-effects fit at prior rate 1 and computational bundle 0; the structured "
        "fit without random effects at the same rate and bundle is a comparison. Bundles 1 and "
        "2 repeat computation on the same panel and are reported separately as sensitivity "
        "checks, not as independent empirical samples."
    )
    comparison = (
        f"The primary overall estimate is ${format_fixed(unit_all['att_pct'])}\\%$ "
        f"[${format_fixed(unit_all['lo95_pct'])}\\%$, ${format_fixed(unit_all['hi95_pct'])}\\%$]. "
        f"The no-random-effects comparison is ${format_fixed(no_unit_all['att_pct'])}\\%$ "
        f"[${format_fixed(no_unit_all['lo95_pct'])}\\%$, ${format_fixed(no_unit_all['hi95_pct'])}\\%$]. "
        "Group effects average over treated county-year observations in the observed sample; "
        "they are not transported population averages. "
        f"{primary_subgroup_text} {no_unit_text} The two overall posterior intervals both exclude zero; "
        "the fitted unit-intercept specification changes subgroup uncertainty, but this comparison "
        "does not by itself establish why the intervals differ."
    )
    primary_contrasts = all_seeds[0]["contrasts"]
    unit_primary_id = unit_all["job_id"]
    contrasts = [row for row in primary_contrasts if row["job_id"] == unit_primary_id]
    if len(contrasts) != 3:
        raise IntegrationError(f"Primary unit fit has {len(contrasts)} contrasts, expected three")
    contrasts.sort(key=lambda row: row["contrast"])
    contrast_sentences = []
    for row in contrasts:
        contrast_sentences.append(
            f"{row['contrast']} $ {format_fixed(row['diff_log'], 4)}$ "
            f"[${format_fixed(row['lo95_log'], 4)},\\ {format_fixed(row['hi95_log'], 4)}$], "
            f"$p_{{\\mathrm{{Bayes}}}}={format_p(row['p_bayes'])}$"
        )
    all_include = all(
        parse_float(row["lo95_log"], "contrast lower") <= 0.0 <= parse_float(row["hi95_log"], "contrast upper")
        for row in contrasts
    )
    contrast_intro = (
        "The primary treated-sample group contrasts (log points) are "
        + "; ".join(contrast_sentences)
        + (". All three 95\\% contrast intervals include zero, so the primary fit does not support a population-tercile gradient."
           if all_include else ". The displayed intervals should be read individually; at least one excludes zero." )
    )
    sens = application_sensitivity_summary(all_seeds)
    pre = empirical_pretrend_summary(pretrend)
    return intro, comparison + "\n\n" + contrast_intro + "\n\n" + sens, pre


def make_figure(out_path: Path, primary: dict[str, list[dict[str, str]]],
                catt_rows: list[dict[str, str]]) -> None:
    """Rebuild the two-panel KDE and exported ATT/GATE intervals as a vector PDF."""
    import matplotlib

    matplotlib.use("Agg")
    try:
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError as exc:
        raise IntegrationError("--apply requires Matplotlib and SciPy to recreate the KDE figure") from exc

    values_by_group: dict[str, list[float]] = {group: [] for group in EXPECTED_GROUPS[1:]}
    group_names = {1: "Small population", 2: "Medium population", 3: "High population"}
    for row in catt_rows:
        group = group_names.get(int(row["lpop_group"]))
        if group is None:
            raise IntegrationError(f"Unexpected CATT population group code {row.get('lpop_group')!r}")
        values_by_group[group].append(parse_float(row["catt_pct"], "catt_pct"))
    if sum(map(len, values_by_group.values())) != 291 or [len(values_by_group[g]) for g in EXPECTED_GROUPS[1:]] != [72, 119, 100]:
        raise IntegrationError("Primary CATT export does not contain the expected 291/72/119/100 county-year rows")

    all_values = [value for group in EXPECTED_GROUPS[1:] for value in values_by_group[group]]
    low, high = min(all_values), max(all_values)
    pad = max((high - low) * 0.08, 1.0)
    grid = [low - pad + (high - low + 2 * pad) * index / 399 for index in range(400)]
    colors = {
        "All treated": "#333333",
        "Small population": "#4c78a8",
        "Medium population": "#f2a541",
        "High population": "#59a14f",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_config = tempfile.TemporaryDirectory(prefix="didbcf-app-figure-mpl-")
    old_mpl_config = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = temp_config.name
    try:
        with matplotlib.rc_context({
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }):
            fig, (ax_kde, ax_gate) = plt.subplots(
                1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]},
            )
            kde_all = gaussian_kde(all_values)
            ax_kde.plot(grid, kde_all(grid), color=colors["All treated"], linewidth=2,
                        label="All treated (291 county-years)")
            for group in EXPECTED_GROUPS[1:]:
                values = values_by_group[group]
                density = gaussian_kde(values)(grid)
                ax_kde.plot(grid, density, color=colors[group], linewidth=1.5,
                            label=f"{group} ({len(values)} county-years)")
            ax_kde.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
            ax_kde.set_title("Transformed posterior-mean CATTs")
            ax_kde.set_xlabel("$100[\\exp(\\mathbb{E}(\\tau_i\\mid y))-1]$ (%)")
            ax_kde.set_ylabel("Kernel density")
            ax_kde.grid(axis="y", color="#dddddd", linewidth=0.55)
            ax_kde.spines["top"].set_visible(False)
            ax_kde.spines["right"].set_visible(False)
            ax_kde.legend(frameon=False, fontsize=7.3, loc="best")

            offsets = {
                "Primary: unit random effects": 0.13,
                "Comparison: no random effects": -0.13,
            }
            palette = {
                "Primary: unit random effects": "#1f77b4",
                "Comparison: no random effects": "#d17a22",
            }
            markers = {
                "Primary: unit random effects": "o",
                "Comparison: no random effects": "s",
            }
            rows_by_role = primary
            y_positions = {group: len(EXPECTED_GROUPS) - i - 1 for i, group in enumerate(EXPECTED_GROUPS)}
            for role in ("Primary: unit random effects", "Comparison: no random effects"):
                for row in rows_by_role[role]:
                    group = row["group"]
                    y = y_positions[group] + offsets[role]
                    lower = parse_float(row["lo95_pct"], "lo95_pct")
                    upper = parse_float(row["hi95_pct"], "hi95_pct")
                    mean = parse_float(row["att_pct"], "att_pct")
                    ax_gate.hlines(y, lower, upper, color=palette[role], linewidth=1.7, zorder=2)
                    ax_gate.vlines([lower, upper], y - 0.07, y + 0.07,
                                   color=palette[role], linewidth=1.0, zorder=2)
                    ax_gate.plot(mean, y, marker=markers[role], color=palette[role],
                                 markersize=5.5, linestyle="none", zorder=3)
            ax_gate.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
            ax_gate.set_yticks([y_positions[g] for g in EXPECTED_GROUPS])
            count_by_group = {row["group"]: row["n_obs"] for row in primary["Primary: unit random effects"]}
            ax_gate.set_yticklabels([f"{g} (n={count_by_group[g]})" for g in EXPECTED_GROUPS])
            ax_gate.set_title("ATT and treated-group averages")
            ax_gate.set_xlabel("Effect (%)")
            ax_gate.grid(axis="x", color="#dddddd", linewidth=0.55)
            ax_gate.spines["top"].set_visible(False)
            ax_gate.spines["right"].set_visible(False)
            legend = [
                Line2D([0], [0], marker=markers[role], color=palette[role], linestyle="-",
                       markersize=5.5, label=("Unit random effects" if "unit" in role else "No random effects"))
                for role in ("Primary: unit random effects", "Comparison: no random effects")
            ]
            ax_gate.legend(handles=legend, frameon=False, fontsize=7.6,
                           loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
            fig.suptitle("MPDTA application, primary prior rate 1, computational bundle 0", y=0.99)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            fig.savefig(out_path, format="pdf", bbox_inches="tight", metadata={
                "Title": "MPDTA application repaired-engine results",
                "Subject": "Exported DiD-BCF application estimates and CATT distribution",
                "Creator": "integrate_application.py",
            })
            plt.close(fig)
    finally:
        if old_mpl_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mpl_config
        temp_config.cleanup()


def single_replace(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise IntegrationError(f"Expected exactly one manuscript anchor for {label}, found {count}")
    return text.replace(old, new, 1)


def replace_application_table(main: str, table: str) -> str:
    start = "We refit the \\texttt{mpdta} application with the structured DiD-BCF used in the simulations"
    if main.count(start) != 1:
        raise IntegrationError("Could not identify exactly one primary application paragraph")
    left = main.index(start)
    table_label = r"\label{tab:mpdta_gate}"
    label_pos = main.find(table_label, left)
    if label_pos < 0 or main.find(table_label, label_pos + 1) >= 0:
        raise IntegrationError("Could not identify the unique primary application table")
    end = main.find(r"\end{table}", label_pos)
    if end < 0:
        raise IntegrationError("Primary application table has no end marker")
    end += len(r"\end{table}")
    old = main[left:end]
    if old.count(table_label) != 1 or old.count(r"\caption{") != 1:
        raise IntegrationError("Primary application table anchor changed unexpectedly")
    # Replace the table label's full environment and its preceding old paragraph.
    paragraph_table = table
    paragraph_start = main.find("\n\\begin{table}", left)
    if paragraph_start < 0 or paragraph_start > label_pos:
        raise IntegrationError("Could not isolate the primary application table environment")
    paragraph = main[left:paragraph_start].strip()
    return main[:left] + paragraph_table + main[end:]


def section_replace(main: str, start_marker: str, end_marker: str,
                    replacement: str, label: str) -> str:
    if main.count(start_marker) != 1 or main.count(end_marker) != 1:
        raise IntegrationError(f"Expected unique section anchors for {label}")
    start = main.index(start_marker)
    end = main.index(end_marker, start)
    if end <= start:
        raise IntegrationError(f"Reversed section anchors for {label}")
    return main[:start] + replacement + main[end:]


def patch_main(main_text: str, primary: dict[str, list[dict[str, str]]],
               all_seeds: dict[int, dict[str, list[dict[str, str]]]],
               pretrend: list[dict[str, str]], gate_table: str,
               prior_table_name: str, pretrend_table_name: str) -> str:
    intro, comparison_and_sensitivity, pre_summary = main_application_prose(primary, all_seeds, pretrend)
    app_pending = "The pre-trend simulations and county application remain historical; their full repaired-engine reruns are pending."
    main_text = single_replace(
        main_text, app_pending,
        "The pre-trend simulations remain historical pending their repaired reruns; the county application now uses the completed repaired-engine compact analysis.",
        "abstract application status",
    )
    methods_pending = "The pre-trend simulations and empirical application remain historical; their full repaired-engine reruns are pending."
    main_text = single_replace(
        main_text, methods_pending,
        "The pre-trend simulations remain historical pending their repaired reruns. The empirical application uses the completed repaired-engine compact analysis described in Section~\\ref{real_life_did_bcf}.",
        "methods application status",
    )
    section_open = "The results in this section use the historical implementation. The full repaired-engine application rerun remains pending."
    main_text = single_replace(
        main_text, section_open,
        "The primary DiD-BCF results in this section use the completed repaired-engine application analysis. The rate-1, computational bundle-0 unit-random-effects fit is primary; the no-random-effects fit is a structured comparison. Computational bundles 1 and 2 are reported separately and are not pooled. The reduced-data correction is retained as a historical comparison.",
        "application section status",
    )

    # Replace exactly the old primary paragraph and first table, leaving the
    # historical fixed-K estimator table and every unrelated section intact.
    start = "We refit the \\texttt{mpdta} application with the structured DiD-BCF used in the simulations"
    end_marker = r"\label{tab:mpdta_gate}"
    if main_text.count(start) != 1 or main_text.count(end_marker) != 1:
        raise IntegrationError("Primary application paragraph/table anchors are not unique")
    p0 = main_text.index(start)
    l0 = main_text.index(end_marker, p0)
    table_end = main_text.find(r"\end{table}", l0)
    if table_end < 0:
        raise IntegrationError("Primary application table end marker is missing")
    table_end += len(r"\end{table}")
    # The generated table replaces the old explanatory paragraph and table.
    main_text = main_text[:p0] + intro + "\n\n" + gate_table.rstrip() + main_text[table_end:]

    old_no_unit = (
        "The overall effect is $-4.67\\%$ [$-7.26\\%$, $-2.06\\%$], "
        "$p_{\\mathrm{Bayes}}=0.001$ (Table~\\ref{tab:mpdta_gate}). Tercile effects average over treated "
        "county-year observations in the observed sample; we do not transport these estimates to population averages. "
        "Unit intercepts are load-bearing: without them the point estimates are similar ($-4.51\\%$ overall) but "
        "intervals roughly double and every group spans zero ($-9.72\\%$ to $+0.91\\%$ overall). Absorbing "
        "county-level unobserved heterogeneity is what makes the effect estimable in this county panel, as expected, "
        "and why \\texttt{structured\\_rfx\\_unit} is reported."
    )
    no_unit_replacement = comparison_and_sensitivity.split("\n\n", 1)[0]
    main_text = single_replace(main_text, old_no_unit, no_unit_replacement, "primary versus no-random-effects prose")

    old_contrasts = (
        "The treated-observation tercile means do not support heterogeneity by county size. Posterior differences (log points) "
        "are Medium$-$Small $-0.0129$ [$-0.0499$, $+0.0291$] $p=0.453$, High$-$Small $+0.0072$ [$-0.0375$, $+0.0636$] "
        "$p=0.863$, and High$-$Medium $+0.0201$ [$-0.0101$, $+0.0683$] $p=0.332$. The high-population treated subgroup itself "
        "is $p=0.059$ and just crosses zero before any multiple-testing correction, with only $72$--$119$ treated observations "
        "per tercile. The defensible statement is a negative effect throughout that the observed treated sample cannot resolve by county size."
    )
    contrast_para = comparison_and_sensitivity.split("\n\n", 1)[1].split("\n\n", 1)[0]
    main_text = single_replace(main_text, old_contrasts, contrast_para, "primary group contrasts")

    prior_para = application_sensitivity_summary(all_seeds)
    correction_anchor = "The fixed-$K$ reference-fold convolution provides a historical comparison"
    if main_text.count(correction_anchor) != 1:
        raise IntegrationError("Historical corrected-estimator paragraph anchor is not unique")
    main_text = main_text.replace(correction_anchor, prior_para + "\n\n" + correction_anchor, 1)

    # Update only the raw repaired-engine numeric citations inside the retained
    # historical-correction paragraph. Its corrected-estimator numbers remain byte-for-byte.
    corr_start = main_text.index(correction_anchor)
    corr_end = main_text.find("\n\nFigure~\\ref{Figure3_did_bcf}", corr_start)
    if corr_end < 0:
        raise IntegrationError("Cannot bound historical correction comparison paragraph")
    corr = main_text[corr_start:corr_end]
    old_raw = "$-4.67\\%$ [$-7.26\\%$, $-2.06\\%$]"
    unit_all = next(row for row in primary["Primary: unit random effects"] if row["group"] == "All treated")
    new_raw = (
        f"${format_fixed(unit_all['att_pct'])}\\%$ "
        f"[${format_fixed(unit_all['lo95_pct'])}\\%$, ${format_fixed(unit_all['hi95_pct'])}\\%$]"
    )
    if corr.count(old_raw) != 1:
        raise IntegrationError(f"Expected one historical-comparison reference to raw old ATT, found {corr.count(old_raw)}")
    corr = corr.replace(old_raw, new_raw)
    main_text = main_text[:corr_start] + corr + main_text[corr_end:]

    old_catt_pretrend = (
        "Figure~\\ref{Figure3_did_bcf} distinguishes heterogeneity from uncertainty. The left panel is the cross-county "
        "spread of posterior-mean CATTs ($\\mathrm{sd}\\approx0.06$), a real quantity but not uncertainty; the right panel is "
        "posterior uncertainty for the overall ATT and the treated-tercile averages. A historical pre-trend diagnostic on the same "
        "data (unconstrained lead model, Section~S5) finds no violation (slope $-0.0058$ [$-0.0195$, $+0.0063$], $p=0.248$)."
    )
    new_catt_pretrend = (
        "Figure~\\ref{Figure3_did_bcf} separates cross-sectional dispersion of transformed posterior-mean CATTs from "
        "posterior uncertainty in the group-average effects. The left panel has 291 treated county-year rows from 191 distinct "
        "counties; counties can contribute multiple years. " + pre_summary
    )
    main_text = single_replace(main_text, old_catt_pretrend, new_catt_pretrend, "CATT and empirical pretrend summary")

    old_caption = (
        "\\caption{Left: KDE of posterior-mean CATTs across treated counties (cross-sectional heterogeneity). Right: 95\\% "
        "credible intervals for the ATT and treated-observation tercile averages under the raw \\texttt{structured\\_rfx\\_unit} "
        "fit of Table~\\ref{tab:mpdta_gate}.}"
    )
    new_caption = (
        "\\caption{Left: KDE of $100[\\exp(\\mathbb{E}(\\tau_i\\mid y))-1]$ across the 291 treated county-year rows "
        "(191 distinct counties); this displays cross-sectional dispersion of transformed posterior-mean CATTs, not posterior "
        "uncertainty, and counties may recur across years. Right: exported posterior means and 95\\% interval bounds for the "
        "overall ATT and treated county-year population-group averages, comparing the primary unit-random-effects fit with the "
        "structured no-random-effects fit at prior rate 1, computational bundle 0.}"
    )
    main_text = single_replace(main_text, old_caption, new_caption, "application figure caption")

    old_discussion = (
        "The overall negative effect is consistent with \\textcite{Thompson2009} and \\textcite{Kalenkoski2013}, who emphasize "
        "that county-level analyses can differ from state aggregates, but the absence of supported heterogeneity here cautions "
        "against reading treated-county population terciles as distinct impacts. Smaller counties may host more small businesses "
        "and wage-sensitive sectors \\parencite{jpmorganchase_rural_divide_2025,ratcliffe2016defining,usda_county_typology_2025}, "
        "yet at this sample the population gradient among treated counties is not statistically distinguishable under the raw fit "
        "and only marginal under the calibrated correction, where the high-population treated subgroup separates from the other two "
        "at $p\\approx0.05$ before multiple-testing adjustment. The value of DiD-BCF in this application is therefore not a settled "
        "ranked ordering of treated-subgroup effects but a reproducible, heterogeneity-capable estimate whose uncertainty is reported "
        "on the same scale as the effect: the left panel of Figure~\\ref{Figure3_did_bcf} would otherwise suggest an order of "
        "magnitude more precision than the posterior carries."
    )
    new_discussion = (
        "The primary repaired-engine fit estimates a negative overall effect, consistent in direction with \\textcite{Thompson2009} "
        "and \\textcite{Kalenkoski2013}, who emphasize that county-level analyses can differ from state aggregates. The three primary "
        "treated-sample population-group contrasts do not support distinct impacts by county size. Smaller counties may host more "
        "small businesses and wage-sensitive sectors \\parencite{jpmorganchase_rural_divide_2025,ratcliffe2016defining,usda_county_typology_2025}, "
        "but the primary compact analysis does not establish a population gradient. The separate reduced-data correction remains a "
        "historical estimator comparison: its previously reported marginal high-population contrasts are not new repaired-engine "
        "application results. The figure distinguishes county-year variation in transformed posterior-mean CATTs from the exported "
        "uncertainty in group averages."
    )
    main_text = single_replace(main_text, old_discussion, new_discussion, "application discussion")

    main_text = single_replace(
        main_text,
        "The historical teen-employment application illustrates the distinction between an overall effect and limited evidence about differences among treated-county population terciles; a full repaired-engine application rerun remains pending.",
        "The repaired-engine teen-employment application reports a negative overall estimate and limited evidence about differences among treated-county population terciles, with prior-rate and empirical pretrend sensitivity reported separately.",
        "conclusion application status",
    )
    main_text = single_replace(
        main_text,
        "The completed main grid does not resolve finite-chain convergence, and the full repaired-engine pre-trend and application reruns remain pending.",
        "The completed main grid does not resolve finite-chain convergence, and repaired-engine pre-trend simulation reruns remain pending.",
        "conclusion simulation-only pending status",
    )
    # Keep the application sensitivity summary adjacent to the contrast prose.
    sensitivity_marker = prior_para
    if main_text.count(sensitivity_marker) != 1:
        raise IntegrationError("Generated sensitivity prose does not have a unique insertion")
    # The exact table name appears in the prose and is resolved in the supplement.
    return main_text


def patch_supplement(supp_text: str) -> str:
    old = "The pre-trend simulations and application results in this paper remain historical pending their full repaired-engine reruns."
    new = (
        "The pre-trend simulations remain historical pending their full repaired-engine reruns. "
        "The empirical application results in the main paper use the complete repaired-engine application collection."
    )
    supp_text = single_replace(supp_text, old, new, "supplement blanket application status")
    anchor = r"\subsection{Conditional surfaces across designs}\label{ssec:surfres}"
    if supp_text.count(anchor) != 1:
        raise IntegrationError("Supplementary conditional-surfaces anchor is not unique")
    addition = (
        "\\subsection{Empirical application prior sensitivity and pretrend}\\label{ssec:application_sensitivity}\n"
        "The application sensitivity grid reports every prior-rate/specification/computational-bundle fit separately. "
        "Bundles 1 and 2 reuse the same panel and are not pooled. The empirical pretrend table reports the one-sided posterior-tail "
        "probabilities and the prespecified $0.025$ decision rule. These application diagnostics are distinct from the simulation "
        "pretrend operating-characteristic results above.\n"
        "\\input{sim_tables/tab_application_prior_sensitivity.tex}\n"
        "\\input{sim_tables/tab_application_pretrend.tex}\n\n"
    )
    return supp_text.replace(anchor, addition + anchor, 1)


def source_output_rows(files: dict[str, Path], summary: dict[str, Any],
                       all_seeds: dict[int, dict[str, list[dict[str, str]]]],
                       pretrend: list[dict[str, str]]) -> dict[str, Any]:
    _, primary_rows = read_csv(files["application_primary_estimates.csv"])
    return {
        "collection_manifest_hash": summary["manifest"]["manifest_hash"],
        "package_hash": summary["package"]["package_hash"],
        "validated_application_job_ids": summary["completed_application_ids"],
        "validated_fit_record_hashes": summary["validated_application_record_hashes"],
        "phase_completed": summary["phase_completed"],
        "phase_expected": EXPECTED_PHASES,
        "replications_are_same_panel_computational_bundles": True,
        "primary_estimates": primary_rows,
        "seed_bundles": {
            str(rep): {name: rows for name, rows in row_tables.items()}
            for rep, row_tables in all_seeds.items()
        },
        "empirical_pretrend_summary": pretrend,
    }


def make_artifacts(input_dir: Path, summary: dict[str, Any], work_dir: Path) -> dict[str, bytes]:
    files = validated_report(input_dir, summary, work_dir)
    primary = exact_primary_rows(files)
    seeds = all_seed_rows(files)
    pretrend = pretrend_rows(files["application_pretrend_diagnostics.csv"])
    sens_rows = [row for rep_data in seeds.values() for row in rep_data["gates"]]
    gate_table = gate_tex_table(primary)
    prior_table = prior_sensitivity_tex(sens_rows)
    pretrend_table = pretrend_tex(pretrend)
    intro, comparison_and_sensitivity, pre_summary = main_application_prose(primary, seeds, pretrend)

    main_text = MAIN_TEX.read_text(encoding="utf-8")
    supp_text = SUPP_TEX.read_text(encoding="utf-8")
    main_patched = patch_main(
        main_text, primary, seeds, pretrend, gate_table,
        PAPER_PRIOR_TEX.name, PAPER_PRETREND_TEX.name,
    )
    supp_patched = patch_supplement(supp_text)

    unit_job = next(
        row["job_id"] for row in primary["Primary: unit random effects"]
        if row["group"] == "All treated"
    )
    primary_catt = [row for row in seeds[0]["catt"] if row["job_id"] == unit_job]
    if len(primary_catt) != 291:
        raise IntegrationError(f"Primary CATT fit has {len(primary_catt)} rows, expected 291")

    # Build vector PDF once, then install identical bytes in versioned results
    # and the paper image target. Temporary work is outside manuscript/results.
    figure_path = work_dir / "fig_mpdta_gate_kde.pdf"
    make_figure(figure_path, primary, primary_catt)
    figure_bytes = figure_path.read_bytes()

    artifacts: dict[str, bytes] = {
        "main.tex": main_patched.encode("utf-8"),
        "supplementary_material.tex": supp_patched.encode("utf-8"),
        "paper_figure.pdf": figure_bytes,
        "paper_prior_sensitivity.tex": prior_table.encode("utf-8"),
        "paper_pretrend.tex": pretrend_table.encode("utf-8"),
        "central/application_primary_estimates.csv": files["application_primary_estimates.csv"].read_bytes(),
        "central/application_pretrend_diagnostics.csv": files["application_pretrend_diagnostics.csv"].read_bytes(),
        "central/application_prior_sensitivity.tex": prior_table.encode("utf-8"),
        "central/application_empirical_pretrend.tex": pretrend_table.encode("utf-8"),
        "central/fig_mpdta_gate_kde.pdf": figure_bytes,
    }
    for rep in EXPECTED_REPS:
        for table_name, artifact_prefix in (
            ("application_gates_rep", "application_gates"),
            ("application_catt_rep", "application_catt"),
            ("application_contrasts_rep", "application_contrasts"),
        ):
            key = f"{table_name}{rep}.csv"
            artifacts[f"central/{key}"] = files[key].read_bytes()
        sidecar = {
            "rep": rep,
            "bundle_role": "prespecified primary/comparison" if rep == 0 else "same-panel computational repeat",
            **seeds[rep],
        }
        artifacts[f"central/application_bundle_rep{rep}.json"] = (
            json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    for artifact_name, json_value in (
        ("central/application_integration_source.json", source_output_rows(files, summary, seeds, pretrend)),
        ("central/application_pretrend_9rows.json", pretrend),
    ):
        artifacts[artifact_name] = (json.dumps(json_value, ensure_ascii=False, indent=2,
                                               sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    report_lines = [
        "# Repaired-engine MPDTA application integration",
        "",
        "Complete compact application collection: 27/27 jobs, comprising 9 main-prior jobs and 18 prior-sensitivity jobs. The wider simulation campaign may remain incomplete.",
        "",
        "The rate-1 computational bundle 0 unit-random-effects fit is the prespecified primary estimate. Bundle 0 without random effects is a comparison. Bundles 1 and 2 reuse the same observed panel and remain separate computational repeats; posterior intervals are not pooled.",
        "",
        "The primary CATT distribution uses `catt_pct = 100 * (exp(catt_log) - 1)`, where `catt_log` is the posterior mean log effect. It shows cross-sectional dispersion, not posterior uncertainty. The 291 treated county-year rows come from 191 distinct counties.",
        "",
        "The pretrend table reports the exported one-sided posterior-tail probabilities separately from gate `p_bayes` summaries. The prespecified diagnostic cutoff is 0.025 and includes the Bonferroni-any decision.",
        "",
        "The full-precision source summaries and validation provenance are in `application_integration_source.json`; bundle-level compact data are in the three `application_bundle_rep*.json` files and corresponding CSVs.",
        "",
    ]
    artifacts["central/application_integration_report.md"] = "\n".join(report_lines).encode("utf-8")
    return artifacts


def destination_map(artifacts: dict[str, bytes]) -> dict[Path, bytes]:
    dest: dict[Path, bytes] = {}
    central_prefix = "central/"
    for relative, data in artifacts.items():
        if relative.startswith(central_prefix):
            dest[CENTRAL_OUTPUT / relative[len(central_prefix):]] = data
    dest[MAIN_TEX] = artifacts["main.tex"]
    dest[SUPP_TEX] = artifacts["supplementary_material.tex"]
    dest[PAPER_FIGURE] = artifacts["paper_figure.pdf"]
    dest[PAPER_PRIOR_TEX] = artifacts["paper_prior_sensitivity.tex"]
    dest[PAPER_PRETREND_TEX] = artifacts["paper_pretrend.tex"]
    return dest


def plan_lines(input_dir: Path) -> list[str]:
    return [
        f"Input collection: {input_dir}",
        f"Versioned output: {CENTRAL_OUTPUT}",
        f"Paper files staged only with --apply: {MAIN_TEX}, {SUPP_TEX}, {PAPER_FIGURE}, {PAPER_PRIOR_TEX}, {PAPER_PRETREND_TEX}",
        f"Backup directory on --apply: {PAPER / 'audit' / BACKUP_DIR_NAME}",
        "No sampling, model fitting, manuscript edits, or results writes occur in dry-run mode.",
    ]


def snapshot_and_write(destinations: dict[Path, bytes]) -> list[str]:
    backup_root = PAPER / "audit" / BACKUP_DIR_NAME
    if CENTRAL_OUTPUT.exists():
        raise IntegrationError(f"Refusing to overwrite existing versioned output directory: {CENTRAL_OUTPUT}")
    for path in (MAIN_TEX, SUPP_TEX, PAPER_FIGURE):
        if not path.is_file():
            raise IntegrationError(f"Expected existing manuscript target before integration: {path}")
    backup_root.mkdir(parents=True, exist_ok=True)
    backed_up = []
    for target in (MAIN_TEX, SUPP_TEX, PAPER_FIGURE, PAPER_PRIOR_TEX, PAPER_PRETREND_TEX):
        if not target.exists():
            continue
        relative = target.relative_to(PAPER)
        backup = backup_root / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise IntegrationError(f"Refusing to overwrite existing original snapshot: {backup}")
        shutil.copy2(target, backup)
        if target.read_bytes() != backup.read_bytes():
            raise IntegrationError(f"Snapshot verification failed for {target}")
        backed_up.append(str(backup))

    # Stage all generated bytes under the repository, then atomically replace
    # each target. The snapshot above is the recovery copy for any OS failure.
    with tempfile.TemporaryDirectory(prefix="application-integration-stage-", dir=REPO) as stage_name:
        stage_root = Path(stage_name)
        staged: list[tuple[Path, Path]] = []
        for target, data in destinations.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            staged_path = stage_root / f"{len(staged):04d}.staged"
            staged_path.write_bytes(data)
            staged.append((staged_path, target))
        for staged_path, target in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(target)
    return backed_up


def run(args: argparse.Namespace) -> int:
    input_dir = args.input.resolve()
    print("Application integration plan")
    for line in plan_lines(input_dir):
        print(f"  {line}")

    try:
        summary = collection_summary(input_dir)
    except (IntegrationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Source validation: BLOCKED: {exc}")
        print("No manuscript or results files were written.")
        return 2

    app_completed = sum(summary["phase_completed"].values())
    print(
        "Validated application counts: "
        f"{summary['phase_completed']['application']}/9 main-prior, "
        f"{summary['phase_completed']['application_sensitivity']}/18 sensitivity, "
        f"{app_completed}/27 total."
    )
    print(f"Validated application fit-record IDs: {len(summary['validated_application_record_hashes'])}/27.")
    if not summary["complete"]:
        missing = summary["missing_application_ids"]
        print("Source validation: BLOCKED: collection is not complete for manuscript integration.")
        if missing:
            print("Missing application IDs: " + ", ".join(missing))
        print("No manuscript or results files were written.")
        return 2

    with tempfile.TemporaryDirectory(prefix="didbcf-application-integration-") as temp_name:
        work_dir = Path(temp_name)
        try:
            artifacts = make_artifacts(input_dir, summary, work_dir)
        except (IntegrationError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            print(f"Validated compact-output integration: BLOCKED: {exc}")
            print("No manuscript or results files were written.")
            return 2
        destinations = destination_map(artifacts)
        print(f"Validated compact-output integration: ready; {len(destinations)} output files staged.")
        if not args.apply:
            print("Dry run complete. No manuscript or Results files were written.")
            return 0
        try:
            backups = snapshot_and_write(destinations)
        except (IntegrationError, OSError) as exc:
            print(f"Apply: BLOCKED: {exc}")
            print("No paper files were intentionally changed; inspect the stated snapshot path if staging began.")
            return 2
        print("Apply complete. Original paper targets were snapshotted:")
        for path in backups:
            print(f"  {path}")
        print(f"Versioned collection artifacts: {CENTRAL_OUTPUT}")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="complete collected output from helpers/collect.py")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="plan and validate without writing (default)")
    mode.add_argument("--apply", action="store_true", help="snapshot and then update manuscript/assets/results")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

