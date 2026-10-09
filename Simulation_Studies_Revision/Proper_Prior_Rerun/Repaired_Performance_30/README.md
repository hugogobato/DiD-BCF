# Repaired DiD-BCF: remaining 30-replication comparison on 48 Colab runtimes

Run the 48 loose notebooks in `Colab/remaining_00.ipynb` through
`remaining_47.ipynb`, once each. They replace the earlier local execution plan
for this study. The local run was stopped at the user's request; its nine
verified completed chains are preserved in `local_completed_outputs/` and
excluded from the 1,881 chains assigned to these notebooks. Original partial
attempts remain in the original local run directory for diagnosis.

This is the approved nine-cell, 30-replication comparison, not the full paper
campaign. It compares the repaired engine at the earlier budget and at B3 with
saved proper-prior results on exactly matched datasets. DGPs, priors, estimands,
model architecture, chain seeds, retained draws, run IDs and the statistical
manifest remain unchanged. Both arms use the repaired native engine with the
IG(2,1) variance prior. Paper metrics determine the performance comparison;
diagnostics are reported as supporting evidence, without discarding draws or
replications. A favorable result does not by itself release the full campaign
or establish asymptotic theory for a finite simulation budget.

| Arm | GFR | MCMC burn-in | Retained per chain | Thinning | Chains |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier budget, repaired engine | 50 | 0 | 500 | 5 | 3 |
| B3, repaired engine | 50 | 2,500 | 2,000 | 5 | 4 |

## Run on Colab

Use a CPU runtime with Python 3.12 through 3.14. Upload the numbered notebooks
as individual `.ipynb` files, or open them from GitHub. Each notebook clones the
repository and checks out the full source revision pinned in its first cell.
It installs exact Python dependency versions, verifies the frozen files and
assignment, downloads SHA256-verified source archives, builds the repaired
stochtree engine into an isolated site directory with one compilation worker,
and runs native and eligible-feature controls before sampling. There is no
global stochtree installation and no GPU requirement.

Run all cells in order. Up to two independent sampling workers are admitted
according to CPU quota and available RAM, with one numeric/OpenMP thread each.
Remaining chains for both budgets on a given dataset stay in the same shard.
Each shard has approximately 6.3–6.7 estimated CPU hours of sampling. Allow
roughly 3–5 wall-clock hours with two workers or 6–8 hours with one; Colab speed
and setup time vary. These are estimates from local measurements and historical
timing proxies, not observed Colab runtimes. The 9.5-hour session window starts
in the configuration cell and includes setup, with 45 minutes reserved for
checkpoint/export. Units that do not fit are deferred for a later session.

Leave `RESUME_ZIP=None` for a clean run. Setting `DRIVE_CHECKPOINT_DIR` to a
directory such as `/content/drive/MyDrive/didbcf_30` enables durable checkpoint
copies every 20 minutes; this is optional and each notebook uses a unique
filename. Local `/content` files can be lost on a forced runtime reset. The
last cell automatically downloads **one result ZIP per notebook**, including
draws, traces, checksums, source provenance, the verified engine and run status.
The notebooks themselves are distributed as loose files, not a ZIP.

To continue a deferred or interrupted shard, upload its previous output ZIP,
set `RESUME_ZIP` to its path, and rerun that notebook with the same Python
version. Completed chains are verified and skipped. Incomplete attempts are
preserved and restarted in a new directory from their original seed. They are
never appended as if they were a continued chain. The verified binary in that
shard's archive is reused and native controls are rerun. Keep the **newest
final result ZIP for each shard** for collection; older partial snapshots
contain repeated completions and must not be collected alongside newer ones.

Saved unpatched proper-prior fits used Python 3.13.15; the nine local repaired
chains used Python 3.12.3. Colab environments are recorded for every chain,
including exact dependency and engine hashes. Different platforms/interpreters
need not produce identical MCMC paths. Thus the reference comparison includes
environment differences; the two repaired budgets provide the closer
within-engine comparison. Each compiled platform must pass its own controls.

## Collect and compare paper metrics

Clone the repository, install the pinned dependencies shown in any notebook
into a scientific Python environment, and put the 48 newest final result ZIPs
in a directory called `incoming`. From the Git repository root run:

```sh
python Simulation_Studies_Revision/Proper_Prior_Rerun/Repaired_Performance_30/helpers/portable_runtime.py collect --incoming incoming --out performance_comparison_returned
```

The collector automatically includes the nine committed local chains and
verifies every archive's engine, identities, seeds, priors, shapes and output
checksums. It rejects duplicate/conflicting completions and reports missing
units. It assembles all chains in their original order and recomputes scalar
and effect-surface metrics. The complete paired report is generated only when
all **1,890 chains and 540 job/budget pairs** are verified, with exactly 30
matched replications per cell. Partial collection returns code 2 and preserves
the integrity report; failed or missing replications are never silently dropped.

The report uses the unchanged, frozen analysis script and saved reference
data. It reports bias, RMSE, MAE, interval coverage and length, surface metrics,
Monte Carlo uncertainty and paired dataset-level bootstrap contrasts, with
supporting chain diagnostics. There is no prespecified equivalence margin or
automatic full-campaign release. Full-study files are under
`performance_comparison_returned/report/` after successful collection.

## Provenance and preparation

`transfer.json` pins the unchanged statistical manifest, portable helpers,
preserved local files and all 48 remaining assignments. `scheduling.csv` maps
every remaining chain to its notebook; `shard_summary.csv` lists estimated
loads and output sizes. `frozen_workspace/` mirrors the original source paths,
so validated worker and analysis files remain byte-for-byte identical.

The package intentionally includes experiment results and the native binary
needed to verify their provenance. These are not ignored by Git. The frozen
original protocol documents its original execution plan; this transfer changes
only execution location, sharding and concurrency, as authorized by the user.
Do not run the old unpatched `Proper_Prior_Rerun/Colab/proper_ig_shard_*`
notebooks or the original 50 embedded comparison notebooks to continue this
48-notebook assignment.

`helpers/prepare_transfer.py` recreates the package from the original stopped
workspace without new sampling. `helpers/build_git_notebooks.py` generates
the loose notebooks with a full committed source revision. Helper sources are
committed first, then the generated notebooks are committed with that revision
pinned in each file. Validation records are in `validation/`.
