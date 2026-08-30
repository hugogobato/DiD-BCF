#!/usr/bin/env python3
"""Generate validation, 48-way family shards, BCF pilots, and mechanics audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_URL = "https://github.com/hugogobato/DiD-BCF.git"
BRANCH = "experiments/theory-calibration-colab"

PILOT_SPECS = {
    "correction": {
        "family": "correction_audit",
        "design": "serial",
        "degree": 2,
        "N": 800,
        "estimators": (
            "raw_structured",
            "current_hybrid_logit",
            "reference_fold_convolution_logit",
        ),
        "expected_fits": 3,
        "output": "correction_pilot",
    },
    "information": {
        "family": "information_ablation",
        "design": "staggered",
        "degree": 3,
        "N": 800,
        "estimators": (
            "full_panel_raw",
            "reduced_cell",
            "pooled_full_panel",
        ),
        "expected_fits": 11,
        "output": "information_pilot",
    },
    "oracle": {
        "family": "correction_audit",
        "design": "oracle_canonical",
        "degree": 2,
        "N": 800,
        "estimators": (
            "oracle_current_hybrid",
            "oracle_reference_fold_convolution",
        ),
        "expected_fits": 3,
        "output": "oracle_pilot",
    },
}


def cell(kind, source):
    return {"cell_type": kind, "metadata": {}, "source": source.splitlines(True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {})}


def notebook(*, family=None, shard=0, n_shards=1, validation=False, pilot=None):
    if validation:
        title = "Extra Theory Experiments lightweight validation"
        setup = f"""# Lightweight local/Colab validation. This runs two smoke tasks only.
REPO_URL = {REPO_URL!r}
BRANCH = {BRANCH!r}
"""
        run = """import os, sys, pathlib, subprocess, zipfile
target = pathlib.Path("DiD-BCF")
if not (target / ".git").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(target)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(target / "Extra_Theory_Experiments" / "requirements-colab.txt")], check=True)
sys.path.insert(0, str(target))
sys.path.insert(0, str(target / "Extra_Theory_Experiments" / "src"))
from extra_theory_experiments.manifest import build_manifest, manifest_frame
from extra_theory_experiments.runner import run_tasks
RESULTS_ROOT = target / "Extra_Theory_Experiments" / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
tasks = build_manifest("correction_audit", reps=2, n_shards=16, shard_id=0)[:2]
manifest_path = RESULTS_ROOT / "validation_manifest.csv"
summary_path = RESULTS_ROOT / "validation_summary.csv"
manifest_frame(tasks).to_csv(manifest_path, index=False)
summary = run_tasks(tasks, out_dir=RESULTS_ROOT / "validation_checkpoints", smoke=True, resume=True)
summary.to_csv(summary_path, index=False)
print("validated", len(summary), "rows")
output_file = str(RESULTS_ROOT / "validation_outputs.zip")
with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in [manifest_path, summary_path]:
        archive.write(path, arcname=path.name)
"""
        download = """try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
"""
        cells = [
            cell("markdown", f"# {title}\n\nThis smoke path avoids stochtree fits."),
            cell("code", setup),
            cell("code", run),
            cell("code", download),
        ]
        return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    pilot_spec = PILOT_SPECS.get(pilot) if pilot else None
    notebook_family = pilot_spec["family"] if pilot_spec else family
    title = (f"Extra Theory Experiments | {notebook_family} | {pilot} worst-case BCF pilot"
             if pilot else
             f"Extra Theory Experiments | {family} | shard {shard:02d}/{n_shards}")
    setup = f"""# Colab bootstrap: branch is pinned so this notebook is self-contained.
import os, sys, pathlib, subprocess
REPO_URL = {REPO_URL!r}
BRANCH = {BRANCH!r}
TARGET = pathlib.Path("DiD-BCF")
if not (TARGET / ".git").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, str(TARGET)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(TARGET / "Extra_Theory_Experiments" / "requirements-colab.txt")], check=True)
sys.path.insert(0, str(TARGET))
sys.path.insert(0, str(TARGET / "Extra_Theory_Experiments" / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
print("Using clone:", TARGET.resolve(), "| max workers: 2, default workers: 1")
"""
    bcf_literal = ("{'num_gfr': 2, 'num_mcmc': 8, 'keep_every': 2, 'num_chains': 1}"
                   if pilot else
                   "{'num_gfr': 50, 'num_mcmc': 500, 'keep_every': 5, 'num_chains': 3}")
    pilot_filter = ""
    if pilot_spec:
        pilot_filter = (
            f"PILOT_ESTIMATORS = {list(pilot_spec['estimators'])!r}\n"
            f"tasks = [task for task in tasks if task.design == {pilot_spec['design']!r} "
            f"and task.degree == {pilot_spec['degree']} and task.N == {pilot_spec['N']} "
            f"and task.rep == 0 and task.estimator in PILOT_ESTIMATORS]\n"
            f"assert len(tasks) == {len(pilot_spec['estimators'])}\n"
            "assert {task.estimator for task in tasks} == set(PILOT_ESTIMATORS)\n"
            f"print('Selected {pilot_spec['design']} degree {pilot_spec['degree']} "
            f"N={pilot_spec['N']} rep=0; expected cached sampler fits: "
            f"{pilot_spec['expected_fits']}')\n"
        )
    pilot_family = repr(notebook_family)
    pilot_shard = 0 if pilot else shard
    pilot_n_shards = 1 if pilot else n_shards
    wave_controls = (
        "N_WAVES = 1\nWAVE_ID = 0"
        if pilot else
        "N_WAVES = int(os.environ.get('ETE_N_WAVES', '1'))\n"
        "WAVE_ID = int(os.environ.get('ETE_WAVE_ID', '0'))\n"
        "if N_WAVES < 1 or not 0 <= WAVE_ID < N_WAVES:\n"
        "    raise ValueError('ETE_WAVE_ID must satisfy 0 <= ETE_WAVE_ID < ETE_N_WAVES')"
    )
    pilot_reps = 1 if pilot else 'int(os.environ.get("ETE_REPS", "100"))'
    pilot_smoke = False if pilot else 'os.environ.get("ETE_SMOKE", "0") == "1"'
    output_suffix = pilot_spec["output"] if pilot_spec else f"shard_{shard:02d}"
    pilot_value = repr(pilot)
    run = f"""from extra_theory_experiments.manifest import build_manifest, manifest_frame
from extra_theory_experiments.runner import run_tasks, write_provenance
import json, os, zipfile
FAMILY = {pilot_family}
SHARD_ID = {pilot_shard}
N_SHARDS = {pilot_n_shards}
REPS = {pilot_reps}
SMOKE = {pilot_smoke}
BCF_PARAMS = {bcf_literal}
{wave_controls}
OUT = str(TARGET / "Extra_Theory_Experiments" / "results" /
          f"extra_theory_{notebook_family}_{output_suffix}_wave_{{WAVE_ID:02d}}_of_{{N_WAVES:02d}}")
os.makedirs(OUT, exist_ok=True)
tasks = build_manifest(FAMILY, reps=REPS, n_shards=N_SHARDS, shard_id=SHARD_ID,
                       wave_id=WAVE_ID, n_waves=N_WAVES)
{pilot_filter}manifest_frame(tasks).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
summary = run_tasks(tasks, out_dir=os.path.join(OUT, "checkpoints"),
                    bcf_params=BCF_PARAMS, smoke=SMOKE, resume=True)
summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
try:
    summary.to_parquet(os.path.join(OUT, "summary.parquet"), index=False)
except Exception as exc:
    print("Parquet skipped:", exc)
write_provenance(os.path.join(OUT, "provenance.json"), tasks=tasks,
                 repo_root=str(TARGET), smoke=SMOKE,
                 bcf_params=BCF_PARAMS,
                 config={{"family": FAMILY, "reps": REPS, "shard": SHARD_ID,
                         "n_shards": N_SHARDS, "wave_id": WAVE_ID,
                         "n_waves": N_WAVES, "K": 2, "workers": 1,
                         "pilot": {pilot_value}}})
with open(os.path.join(OUT, "README_run.txt"), "w", encoding="utf-8") as handle:
    handle.write("Family=" + FAMILY + "; shard=" + str(SHARD_ID) + "/" + str(N_SHARDS) +
                 "; wave=" + str(WAVE_ID) + "/" + str(N_WAVES) + "; reps=" + str(REPS) +
                 "; smoke=" + str(SMOKE) + "\\n")
output_file = OUT + ".zip"
with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for root, _, names in os.walk(OUT):
        for name in names:
            path = os.path.join(root, name)
            archive.write(path, arcname=os.path.relpath(path, OUT))
print("Wrote archive:", output_file)
"""
    download = """try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
"""
    cells = [
        cell("markdown", f"# {title}\n\nUse at most two workers on a 12 GB Colab session; default is one. "
                         "This shard resumes local checkpoints and downloads one zip. "
                         + ("The pilot is fixed to wave 0 of 1."
                            if pilot else
                            "Set ETE_N_WAVES and ETE_WAVE_ID to reuse this notebook across waves; "
                            "the wave is included in the output directory and archive name.")
                         + (" The oracle pilot checks exact nuisance-column alignment, iid default errors, and homogeneous truth 3 through the production sampler."
                            if pilot == "oracle" else "")),
        cell("code", setup),
        cell("code", run),
        cell("code", download),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"},
            "colab": {"name": title}}, "nbformat": 4, "nbformat_minor": 5}


def controlled_mechanics_notebook():
    """Notebook for the no-BCF exact-m0 special-case mechanics audit."""
    setup = f"""# Controlled special-case mechanics diagnostic; no BCF training.
import os, sys, pathlib, subprocess
REPO_URL = {REPO_URL!r}
BRANCH = {BRANCH!r}
TARGET = pathlib.Path("DiD-BCF")
if not (TARGET / ".git").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, str(TARGET)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(TARGET / "Extra_Theory_Experiments" / "requirements-colab.txt")], check=True)
sys.path.insert(0, str(TARGET))
sys.path.insert(0, str(TARGET / "Extra_Theory_Experiments" / "src"))
"""
    run = """from extra_theory_experiments.controlled_mechanics import run_controlled_mechanics
import os
OUT = TARGET / "Extra_Theory_Experiments" / "results" / "controlled_mechanics"
per_rep, metrics, archive_path = run_controlled_mechanics(
    out_dir=OUT,
    reps=int(os.environ.get("ETE_MECH_REPS", "100")),
    n_draws=int(os.environ.get("ETE_MECH_DRAWS", "300")),
    n_units=int(os.environ.get("ETE_MECH_UNITS", "200")), K=2)
print("Controlled exact-m0 mechanics diagnostic, not a theorem proof")
print(metrics.to_string(index=False))
output_file = str(archive_path)
"""
    download = """try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
"""
    cells = [
        cell("markdown", "# Controlled exact-m0 mechanics diagnostic\n\n"
             "This special-case audit fixes every prognostic posterior draw at exact m0_oracle, "
             "compares full Algorithm-1 BB with K=2 fold convolution, and reports GATT/ATT "
             "bias, 95% coverage, interval length, and null rejection. It is not a theorem proof. "
             "Defaults are Colab-feasible and can be overridden with ETE_MECH_REPS, "
             "ETE_MECH_DRAWS, and ETE_MECH_UNITS."),
        cell("code", setup), cell("code", run), cell("code", download),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "colab": {"name": "Extra Theory Experiments | controlled mechanics"}},
        "nbformat": 4, "nbformat_minor": 5}


def generate(output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.ipynb").write_text(
        json.dumps(notebook(validation=True), indent=1), encoding="utf-8")
    for family in ("correction_audit", "information_ablation"):
        for shard in range(48):
            name = f"{family}_shard_{shard:02d}.ipynb"
            (out / name).write_text(
                json.dumps(notebook(family=family, shard=shard, n_shards=48), indent=1),
                encoding="utf-8")
    for pilot_name in ("correction", "information", "oracle"):
        (out / f"{pilot_name}_bcf_pilot.ipynb").write_text(
            json.dumps(notebook(pilot=pilot_name), indent=1), encoding="utf-8")
    (out / "controlled_mechanics.ipynb").write_text(
        json.dumps(controlled_mechanics_notebook(), indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).parents[1] / "notebooks"))
    generate(parser.parse_args().output_dir)
