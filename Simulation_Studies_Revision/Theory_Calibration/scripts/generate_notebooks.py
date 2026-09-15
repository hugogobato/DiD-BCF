#!/usr/bin/env python3
"""Generate validation, family shards, BCF pilots, mechanics, and completion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_URL = "https://github.com/hugogobato/DiD-BCF.git"
BRANCH = "main"

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


def notebook(*, family=None, shard=0, n_shards=1, validation=False, pilot=None,
             single_wave=False):
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
else:
    # A rerun can reuse an older disposable clone. Refresh tracked source
    # files while leaving untracked result checkpoints intact.
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(target), "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(target / "Simulation_Studies_Revision" / "Theory_Calibration" / "requirements-colab.txt")], check=True)
sys.path.insert(0, str(target))
sys.path.insert(0, str(target / "Simulation_Studies_Revision" / "Theory_Calibration" / "src"))
from extra_theory_experiments.manifest import build_manifest, manifest_frame
from extra_theory_experiments.runner import run_tasks
RESULTS_ROOT = target / "Simulation_Studies_Revision" / "Theory_Calibration" / "results"
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
    shard_digits = 3 if single_wave else 2
    title = (f"Extra Theory Experiments | {notebook_family} | {pilot} worst-case BCF pilot"
             if pilot else
             f"Extra Theory Experiments | {family} | shard "
             f"{shard:0{shard_digits}d}/{n_shards}")
    setup = f"""# Colab bootstrap: branch is pinned so this notebook is self-contained.
import os, sys, pathlib, subprocess
REPO_URL = {REPO_URL!r}
BRANCH = {BRANCH!r}
TARGET = pathlib.Path("DiD-BCF")
if not (TARGET / ".git").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, str(TARGET)], check=True)
else:
    # Refresh an existing disposable clone so reruns cannot keep stale code.
    subprocess.run(["git", "-C", str(TARGET), "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(TARGET), "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "requirements-colab.txt")], check=True)
sys.path.insert(0, str(TARGET))
sys.path.insert(0, str(TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "src"))
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
        if (pilot or single_wave) else
        "N_WAVES = int(os.environ.get('ETE_N_WAVES', '1'))\n"
        "WAVE_ID = int(os.environ.get('ETE_WAVE_ID', '0'))\n"
        "if N_WAVES < 1 or not 0 <= WAVE_ID < N_WAVES:\n"
        "    raise ValueError('ETE_WAVE_ID must satisfy 0 <= ETE_WAVE_ID < ETE_N_WAVES')"
    )
    if pilot:
        pilot_reps = "1"
    elif single_wave:
        # Single-wave completion notebooks keep the per-design replication
        # counts (null/PT at 200) unless ETE_REPS overrides them.
        pilot_reps = ("int(os.environ['ETE_REPS']) if os.environ.get('ETE_REPS', '').strip() "
                      "not in ('', '0') else None")
    else:
        pilot_reps = 'int(os.environ.get("ETE_REPS", "100"))'
    pilot_smoke = False if pilot else 'os.environ.get("ETE_SMOKE", "0") == "1"'
    output_suffix = (pilot_spec["output"] if pilot_spec else
                     f"shard_{shard:0{shard_digits}d}")
    if single_wave:
        out_path = f'f"extra_theory_{notebook_family}_{output_suffix}"'
    else:
        out_path = (f'f"extra_theory_{notebook_family}_{output_suffix}_wave_'
                    f'{{WAVE_ID:02d}}_of_{{N_WAVES:02d}}"')
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
OUT = str(TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "results" /
          {out_path})
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
                            "This is a single-wave notebook: run it once; no ETE_N_WAVES or "
                            "ETE_WAVE_ID configuration is needed. Replication counts follow each "
                            "design's config value (ETE_REPS overrides)."
                            if single_wave else
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


def completion_pair_notebook(group) -> dict:
    """One single-wave notebook that runs two original completion shards.

    The user already ran shards 0-53 as single-shard notebooks.  From shard 54
    on, each notebook runs two original shards sequentially into their original
    ``..._shard_XXX`` output directories, so the union of outputs is still all
    384 shard directories and ``aggregate_archives.py --n-shards 384`` needs no
    change.
    """
    first, second = int(group[0]), int(group[-1])
    n_shards = FAMILY_SHARDS["correction_completion"]
    title = (f"Extra Theory Experiments | correction_completion | "
             f"shards {first:03d}+{second:03d}/{n_shards}")
    setup = f"""# Colab bootstrap: branch is pinned so this notebook is self-contained.
import os, sys, pathlib, subprocess
REPO_URL = {REPO_URL!r}
BRANCH = {BRANCH!r}
TARGET = pathlib.Path("DiD-BCF")
if not (TARGET / ".git").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, str(TARGET)], check=True)
else:
    # Refresh an existing disposable clone so reruns cannot keep stale code.
    subprocess.run(["git", "-C", str(TARGET), "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(TARGET), "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "requirements-colab.txt")], check=True)
sys.path.insert(0, str(TARGET))
sys.path.insert(0, str(TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
print("Using clone:", TARGET.resolve(), "| max workers: 2, default workers: 1")
"""
    run_template = """from extra_theory_experiments.manifest import build_manifest, manifest_frame
from extra_theory_experiments.runner import run_tasks, write_provenance
import json, os, zipfile

import pandas as pd

FAMILY = 'correction_completion'
SHARD_GROUP = __SHARD_GROUP__
SHARD_IDS = list(SHARD_GROUP)
N_SHARDS = __N_SHARDS__
N_WAVES = 1
WAVE_ID = 0
REPS = int(os.environ['ETE_REPS']) if os.environ.get('ETE_REPS', '').strip() not in ('', '0') else None
SMOKE = os.environ.get("ETE_SMOKE", "0") == "1"
BCF_PARAMS = {'num_gfr': 50, 'num_mcmc': 500, 'keep_every': 5, 'num_chains': 3}

RESULTS_ROOT = TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
ARCHIVES = []
SUMMARIES = []
for SHARD_ID in SHARD_IDS:
    print(f"[correction_completion] running original shard {SHARD_ID:03d}/{N_SHARDS:03d}", flush=True)
    OUT = str(RESULTS_ROOT / f"extra_theory_correction_completion_shard_{SHARD_ID:03d}")
    os.makedirs(OUT, exist_ok=True)
    tasks = build_manifest(FAMILY, reps=REPS, n_shards=N_SHARDS, shard_id=SHARD_ID,
                           wave_id=WAVE_ID, n_waves=N_WAVES)
    manifest_frame(tasks).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
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
                     config={"family": FAMILY, "reps": REPS, "shard": SHARD_ID,
                             "n_shards": N_SHARDS, "wave_id": WAVE_ID,
                             "n_waves": N_WAVES, "K": 2, "workers": 1,
                             "pilot": None})
    with open(os.path.join(OUT, "README_run.txt"), "w", encoding="utf-8") as handle:
        handle.write("Family=" + FAMILY + "; shard=" + str(SHARD_ID) + "/" + str(N_SHARDS) +
                     "; wave=" + str(WAVE_ID) + "/" + str(N_WAVES) + "; reps=" + str(REPS) +
                     "; smoke=" + str(SMOKE) + "\\n")
    archive_path = OUT + ".zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _, names in os.walk(OUT):
            for name in names:
                path = os.path.join(root, name)
                archive.write(path, arcname=os.path.relpath(path, OUT))
    print("Wrote archive:", archive_path)
    ARCHIVES.append(archive_path)
    SUMMARIES.append(summary)
combined = pd.concat(SUMMARIES, ignore_index=True) if SUMMARIES else pd.DataFrame()
combined_path = str(RESULTS_ROOT /
                    f"extra_theory_correction_completion_shards_{SHARD_IDS[0]:03d}_{SHARD_IDS[-1]:03d}_summary.csv")
combined.to_csv(combined_path, index=False)
print("Wrote combined summary:", combined_path)
OUTPUT_FILES = ARCHIVES + [combined_path]
"""
    run = (run_template
           .replace("__SHARD_GROUP__", repr(tuple(int(x) for x in group)))
           .replace("__N_SHARDS__", str(n_shards)))
    download = """try:
    from google.colab import files
    for output_file in OUTPUT_FILES:
        files.download(output_file)
        print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
"""
    cells = [
        cell("markdown", f"# {title}\n\nUse at most two workers on a 12 GB Colab session; "
                         "default is one. This single-wave notebook runs two original shards "
                         f"sequentially ({first:03d} then {second:03d}), each into its own "
                         "original shard output directory, so aggregation with "
                         "--n-shards 384 is unchanged and each notebook downloads its shard "
                         "archives plus one combined summary. Run the notebook once; no "
                         "ETE_N_WAVES or ETE_WAVE_ID configuration is needed. Replication "
                         "counts follow each design's config value (ETE_REPS overrides)."),
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
else:
    # Refresh an existing disposable clone so reruns cannot keep stale code.
    subprocess.run(["git", "-C", str(TARGET), "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(TARGET), "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "requirements-colab.txt")], check=True)
sys.path.insert(0, str(TARGET))
sys.path.insert(0, str(TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "src"))
"""
    run = """from extra_theory_experiments.controlled_mechanics import run_controlled_mechanics
import os
OUT = TARGET / "Simulation_Studies_Revision" / "Theory_Calibration" / "results" / "controlled_mechanics"
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


FAMILY_SHARDS = {
    "correction_audit": 48,
    "information_ablation": 48,
    # The completion family is emitted as one single wave of 384 shards, so no
    # notebook needs ETE_N_WAVES/ETE_WAVE_ID edits.
    "correction_completion": 384,
}
# Shards 0..53 already ran as one notebook per shard.  From shard 54 on each
# notebook runs a pair of original shards sequentially; notebook j covers
# 2*j - 54 and 2*j - 53, for 54 <= j <= 218.
COMPLETION_SINGLE_SHARDS = 54
COMPLETION_PAIRED_LAST = 218
FAMILIES = ("all", "correction_audit", "information_ablation",
            "correction_completion")
# The generated notebooks live with the final revision artifacts.  The package
# itself (src/, scripts/, configs/, results/, tests/) stays under
# Theory_Calibration/; only the uploadable notebooks moved.
NOTEBOOK_ROOT = Path(__file__).resolve().parents[2] / "DiD_BCF"
THEORY_NOTEBOOKS = NOTEBOOK_ROOT / "Theory_Calibration"
COMPLETION_NOTEBOOKS = NOTEBOOK_ROOT / "Correction_Completion_Notebooks"


def completion_shard_group(notebook_index: int) -> tuple[int, int]:
    """Original shard pair run by completion notebook ``notebook_index``."""
    index = int(notebook_index)
    if not COMPLETION_SINGLE_SHARDS <= index <= COMPLETION_PAIRED_LAST:
        raise ValueError(
            f"paired completion notebook index must lie in "
            f"[{COMPLETION_SINGLE_SHARDS}, {COMPLETION_PAIRED_LAST}], got {index}")
    return (2 * index - 54, 2 * index - 53)


def _write_shards(out: Path, family: str) -> None:
    n_shards = FAMILY_SHARDS[family]
    single_wave = family == "correction_completion"
    digits = 3 if single_wave else 2
    for shard in range(n_shards):
        name = f"{family}_shard_{shard:0{digits}d}.ipynb"
        (out / name).write_text(
            json.dumps(notebook(family=family, shard=shard, n_shards=n_shards,
                                single_wave=single_wave), indent=1),
            encoding="utf-8")


def _write_completion_shards(out: Path) -> None:
    n_shards = FAMILY_SHARDS["correction_completion"]
    for shard in range(COMPLETION_SINGLE_SHARDS):
        name = f"correction_completion_shard_{shard:03d}.ipynb"
        (out / name).write_text(
            json.dumps(notebook(family="correction_completion", shard=shard,
                                n_shards=n_shards, single_wave=True), indent=1),
            encoding="utf-8")
    for index in range(COMPLETION_SINGLE_SHARDS, COMPLETION_PAIRED_LAST + 1):
        name = f"correction_completion_shard_{index:03d}.ipynb"
        (out / name).write_text(
            json.dumps(completion_pair_notebook(completion_shard_group(index)),
                       indent=1),
            encoding="utf-8")


def generate(output_dir: str | Path | None = None, family: str = "all"):
    """Write the selected notebook family (or every family) to disk.

    Without ``output_dir`` the theory notebooks land in
    ``Simulation_Studies_Revision/DiD_BCF/Theory_Calibration`` and the
    completion shards in ``Simulation_Studies_Revision/DiD_BCF/Correction_Completion_Notebooks``.
    A caller-supplied ``output_dir`` collects the selected notebooks flat in
    that directory.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown notebook family {family!r}; expected one of {FAMILIES}")
    explicit = output_dir is not None
    flat = Path(output_dir) if explicit else None
    theory = flat if explicit else THEORY_NOTEBOOKS
    completion = flat if explicit else COMPLETION_NOTEBOOKS
    if family in ("all", "correction_audit", "information_ablation"):
        theory.mkdir(parents=True, exist_ok=True)
        (theory / "validation.ipynb").write_text(
            json.dumps(notebook(validation=True), indent=1), encoding="utf-8")
        (theory / "controlled_mechanics.ipynb").write_text(
            json.dumps(controlled_mechanics_notebook(), indent=1), encoding="utf-8")
    if family in ("all", "correction_audit"):
        _write_shards(theory, "correction_audit")
        for pilot_name in ("correction", "oracle"):
            (theory / f"{pilot_name}_bcf_pilot.ipynb").write_text(
                json.dumps(notebook(pilot=pilot_name), indent=1), encoding="utf-8")
    if family in ("all", "information_ablation"):
        _write_shards(theory, "information_ablation")
        (theory / "information_bcf_pilot.ipynb").write_text(
            json.dumps(notebook(pilot="information"), indent=1), encoding="utf-8")
    if family in ("all", "correction_completion"):
        completion.mkdir(parents=True, exist_ok=True)
        _write_completion_shards(completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None,
                        help="collect the selected notebooks flat in this directory")
    parser.add_argument("--family", default="all", choices=FAMILIES)
    args = parser.parse_args()
    generate(args.output_dir, args.family)
