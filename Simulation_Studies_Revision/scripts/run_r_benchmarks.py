#!/usr/bin/env python3
"""Run R simulation study benchmarks in parallel with on-the-fly data generation.

This script executes the R benchmark scripts (did_dr_new.R, did2s.R,
DoubleML_did.R, synthdid.R) for each scenario under R_code/.

To prevent disk clutter and avoid saving thousands of CSV files to disk:
1. It generates the datasets on-the-fly in a temporary directory.
2. It uses the exact same seeds per replication as the Python notebooks,
   ensuring an apples-to-apples comparison.
3. It copies the R scripts into the temporary directory, executes them there,
   and copies the final output files (*_GATE_and_PValues_*.xlsx, etc.) back to
   the persistent R_code/<scenario>_datasets/ folders.
4. It cleans up the temporary directory immediately after run.

It processes 2 folders at a time (8 concurrent R processes) to maximize CPU utilization.
"""

import os
import sys
import shutil
import tempfile
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from did_bcf_revision.config import get_experiment, degrees_for, all_experiments
from did_bcf_revision.dgps import generate_canonical_did, generate_staggered_did
from did_bcf_revision.exports import to_r_frame

R_CODE_DIR = os.path.join(ROOT, "R_code")

# Derived from the experiment grid rather than hard-coded, so a scenario added to
# config.py (the D_ramp_* family, for instance) is picked up automatically.
FOLDERS = [f"{e.name}_datasets" for e in all_experiments()]

# Workstream PT estimates the pre-trend placebo rather than the ATT, so it has
# its own four scripts.  They are the benchmark counterparts of
# did_bcf_revision/pretrend.py and emit the same PRE / PRE_SUBC estimands, which
# is why they cannot share a file with the ATT benchmarks.
ATT_SCRIPTS = [
    "did_dr_new.R",
    "did2s.R",
    "DoubleML_did.R",
    "synthdid.R"
]
PT_SCRIPTS = [
    "did_dr_pretrend.R",
    "did2s_pretrend.R",
    "synthdid_pretrend.R",
    "DoubleML_pretrend.R",
    # grf-DiD. Set GRF_THREADS=1 in the environment when several of these run
    # concurrently, otherwise each one grabs every core.
    "wang_pretrend.R",
]
ALL_SCRIPTS = ATT_SCRIPTS + PT_SCRIPTS
R_SCRIPTS = ATT_SCRIPTS


def scripts_for(scenario):
    """The estimator scripts that apply to ``scenario``, honouring --scripts."""
    ws = get_experiment(scenario).workstream
    family = PT_SCRIPTS if ws == "PT" else ATT_SCRIPTS
    return [x for x in family if x in R_SCRIPTS]

def generate_temp_data(scenario, reps, temp_dir):
    """Generate iteration CSV files on-the-fly directly in the temp directory."""
    exp = get_experiment(scenario)
    base_N = exp.n_values[0]
    
    # Determine the correct data generator
    if exp.dgp == "canonical":
        GEN = generate_canonical_did
    else:
        GEN = generate_staggered_did

    # Sweep the workstream's linearity degrees (PT runs 1 and 3 only)
    for d in degrees_for(exp.workstream):
        base_dir = os.path.join(temp_dir, f"linearity_degree={d}")
        os.makedirs(base_dir, exist_ok=True)
        
        for rep in range(reps):
            df = GEN(seed=int(rep), **{**exp.dgp_params, "n_units": int(base_N), "linearity_degree": int(d)})
            to_r_frame(df).to_csv(os.path.join(base_dir, f"iteration_{rep}.csv"), index=False)
            
        # If it's a sample-size sweep scenario, generate data for other N values
        if "sweep" in scenario and len(exp.n_values) > 1:
            for N in exp.n_values:
                nd = os.path.join(temp_dir, f"N={N}", f"linearity_degree={d}")
                os.makedirs(nd, exist_ok=True)
                for rep in range(reps):
                    df = GEN(seed=int(rep), **{**exp.dgp_params, "n_units": int(N), "linearity_degree": int(d)})
                    to_r_frame(df).to_csv(os.path.join(nd, f"iteration_{rep}.csv"), index=False)

def run_script(folder, script, temp_folder_path, dest_folder_path):
    """Run a single R script inside the temp folder and copy outputs back."""
    script_path = os.path.join(temp_folder_path, script)
    if not os.path.exists(script_path):
        print(f"Skipping {script} in {folder} (does not exist).")
        return
        
    print(f"[START] Running {script} in {folder}...")
    try:
        # Run Rscript and set working directory to the temporary folder containing the generated data.
        res = subprocess.run(["Rscript", script], cwd=temp_folder_path, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[SUCCESS] Finished {script} in {folder}.")
        else:
            print(f"[ERROR] Failed {script} in {folder} (Exit code: {res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    except Exception as e:
        print(f"[EXCEPTION] Failed {script} in {folder}: {e}")


def collect_outputs(temp_folder_path, dest_folder_path):
    """Copy every result file the R scripts wrote back to the scenario folder.

    Called once per scenario, *after* all four scripts have finished, so no file
    is copied while another process is still writing it.

    The generated panels live in ``linearity_degree=*/`` and ``N=*/``
    sub-folders, so anything at the root of the temp directory that is not an R
    script is an output.  An earlier version filtered on the script name or on
    the substrings GATE/Metrics/Estimates/output, which matched none of the
    ``summaries_<method>_<scenario>_lin_<d>.csv`` files the scripts actually
    write -- so the runner completed successfully and copied nothing.
    """
    copied = 0
    for f in sorted(os.listdir(temp_folder_path)):
        src = os.path.join(temp_folder_path, f)
        if not os.path.isfile(src) or f.endswith(".R"):
            continue
        if not f.endswith((".csv", ".xlsx", ".txt")):
            continue
        shutil.copy2(src, os.path.join(dest_folder_path, f))
        copied += 1
    return copied

def process_scenario_batch(batch_folders, reps, max_workers=8):
    """Process a batch of 2 scenarios in parallel using temp directories."""
    temp_dirs = {}
    
    try:
        # Step 1: Create temp directories & generate data / copy scripts
        for folder in batch_folders:
            scenario_name = folder.replace("_datasets", "")
            persistent_folder = os.path.join(R_CODE_DIR, folder)
            
            # Create a secure temp directory
            td = tempfile.mkdtemp()
            temp_dirs[folder] = td
            
            print(f"[{scenario_name}] Generating datasets in temp dir: {td}...")
            generate_temp_data(scenario_name, reps, td)
            
            # Copy R scripts into the temp directory
            for script in scripts_for(scenario_name):
                src_script = os.path.join(persistent_folder, script)
                if os.path.exists(src_script):
                    shutil.copy2(src_script, os.path.join(td, script))
                    
        # Step 2: Build parallel tasks
        tasks = []
        for folder in batch_folders:
            persistent_folder = os.path.join(R_CODE_DIR, folder)
            td = temp_dirs[folder]
            for script in scripts_for(folder.replace("_datasets", "")):
                tasks.append((folder, script, td, persistent_folder))
                
        # Step 3: Run all 8 script tasks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_script, folder, script, td, dest) for folder, script, td, dest in tasks]
            for future in futures:
                future.result()

        # Step 3b: collect results once every script for a scenario is done
        for folder in batch_folders:
            n = collect_outputs(temp_dirs[folder],
                                os.path.join(R_CODE_DIR, folder))
            print(f"[{folder.replace('_datasets', '')}] copied {n} result files")

    finally:
        # Step 4: Clean up temp directories
        for folder, td in temp_dirs.items():
            scenario_name = folder.replace("_datasets", "")
            print(f"[{scenario_name}] Cleaning up temp directory: {td}...")
            shutil.rmtree(td, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=None, help="Override default replication count (e.g. 10 for quick smoke test)")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="Scenario names to run (default: all). Accepts either "
                         "'D_ramp_nt05' or 'D_ramp_nt05_datasets'.")
    global R_SCRIPTS
    ap.add_argument("--scripts", nargs="+", default=None,
                    choices=ALL_SCRIPTS + [x[:-2] for x in ALL_SCRIPTS],
                    help="Which estimators to run (default: all four). Use this "
                         "to keep the slow ones off this machine, e.g. "
                         "--scripts did_dr_new did2s synthdid leaves DoubleML "
                         "for the Colab notebooks in DoubleML_Colab/.")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Scenarios processed concurrently; each runs 4 R "
                         "processes, so the peak is 4x this. Lower it on a "
                         "memory-constrained machine (default 2 = 8 processes).")
    args = ap.parse_args()

    if args.scripts:
        want = {x if x.endswith(".R") else x + ".R" for x in args.scripts}
        R_SCRIPTS = [x for x in ALL_SCRIPTS if x in want]
    else:
        R_SCRIPTS = ALL_SCRIPTS

    folders = FOLDERS
    if args.scenarios:
        want = {s if s.endswith("_datasets") else s + "_datasets"
                for s in args.scenarios}
        unknown = want - set(FOLDERS)
        if unknown:
            ap.error(f"unknown scenario(s): {sorted(unknown)}")
        folders = [f for f in FOLDERS if f in want]

    print("Starting R benchmarks parallel runner with on-the-fly data generation.")
    print("Comparing apples-to-apples using identical seeds as python notebooks.")
    print(f"Scenarios: {', '.join(f.replace('_datasets', '') for f in folders)}")
    print(f"Estimators: {', '.join(R_SCRIPTS)}")
    print("  (per scenario: workstream PT runs the *_pretrend.R family, the "
          "rest run the ATT family)")

    for i in range(0, len(folders), args.batch_size):
        batch_folders = folders[i:i + args.batch_size]
        print(f"\n==========================================")
        print(f"Executing Batch: {', '.join(batch_folders)}")
        print(f"==========================================")
        
        # Determine replication count
        # For batch, check the configs of the scenarios
        reps_to_use = args.reps
        if reps_to_use is None:
            # Get max default reps in this batch
            reps_list = []
            for folder in batch_folders:
                scen = folder.replace("_datasets", "")
                reps_list.append(get_experiment(scen).reps)
            reps_to_use = max(reps_list)
            
        print(f"Running iterations: {reps_to_use}")
        process_scenario_batch(batch_folders, reps_to_use,
                               max_workers=args.batch_size * len(R_SCRIPTS))
        print(f"Finished Batch: {', '.join(batch_folders)}")
        
    print("\nAll R benchmarks execution completed.")

if __name__ == "__main__":
    main()
