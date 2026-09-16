# Proper-prior DiD-BCF reruns

This campaign replaces the improper IG(0,0) noise-variance prior by IG(2,1)
on standardized outcomes. See `JBES_Submission/audit/PROPER_PRIOR_REPAIR.md`
for the proof and scope. The shape/second-parameter convention is
`p(v) proportional to v^(-a-1) exp(-b/v)`.

Historical `Results/` files, embedded-code notebooks and submission tables
have NOT been updated to this estimator. Do not run the old notebooks for the
new campaign: their embedded code can retain the improper default. Do not use
the old aggregators, which can mix historical and new files.

## Prepared scope

The manifest contains main simulations (5,100 fits), pretrend diagnostics
(4,000), effects under the pretrend scenarios (4,000), application fits and
diagnostics at seeds 0,1,2 (9), simulation prior sensitivity (1,200), and
application prior sensitivity (18). Only the main simulations and application
are needed to replace the headline estimates; the other phases replace the
supplementary diagnostics and provide prespecified robustness checks.

Main simulations use the historical structured/no-random-effect specification;
the application includes both that specification and unit random intercepts.
No posterior augmentation/correction is run. Cohort-specific effects remain
enabled. The adapter now actually forwards `effect_by_cohort` (its default
was already True, so this forwarding fix does not alter the historical default).
The sampler budget is unchanged: 50 GFR iterations, no extra MCMC burn-in,
500 retained draws per chain, thinning 5, three independent chains. Adequacy
of this budget must be checked, not assumed. Convergence failures require a
new, documented budget and manifest, not selective replacement of bad runs.

The main grid includes baseline, strong confounding, AR(1), observable
selection, sharp null, both sample-size sweeps and staggered adoption, with
degrees 1,2,3. Null and diagnostic cells use 200 repetitions; other main cells
use 100. All ten pretrend settings retain degrees 1,3. Excluded contamination
and ramp designs are not silently reintroduced. Scale sensitivity fixes a=2
and b in {0.1,10}, alongside the main b=1, for baseline/AR(1), N=50,200,800,
degree 1, 100 matched seeds, and the application. This tests a substantive
scale choice, not an assertion that IG(2,1) is noninformative.

## Local commands (from repository root)

```sh
rtk proxy env MPLCONFIGDIR=/tmp/didbcf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python DiD-BCF/Simulation_Studies_Revision/Proper_Prior_Rerun/campaign.py smoke
rtk proxy env MPLCONFIGDIR=/tmp/didbcf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python DiD-BCF/Simulation_Studies_Revision/Proper_Prior_Rerun/campaign.py run --phase main --shard 0 --max-jobs 1
```

The second command starts an actual production fit; it has NOT been run in
preparing this package. Remove `--max-jobs 1` only after checking runtime,
available memory and chain diagnostics. Shards are numbered 0 through 47.
Each process runs serially with one BLAS/OpenMP thread. Start with two processes
locally, then increase only after measuring peak memory and other active work;
do not allocate all 20 logical CPUs blindly. Colab jobs use one worker each.
`--phase` accepts a comma-separated subset of the manifest's phase names.
`--hours 8` stops between fits, not inside a fit, so it is not a hard deadline.

`prepare --manifest PATH` writes a new manifest exclusively and performs no
fits. A prepared manifest pins source/data hashes, installed dependency
versions, sampler settings and every job. Source or dependency drift causes an
explicit error; prepare a new campaign rather than editing the existing
manifest. Python version is recorded but not enforced across platforms;
floating-point/MCMC paths may differ across platforms despite identical seeds.

Completed jobs contain `complete.json`, hashed summaries and draw arrays.
Repeated jobs are skipped only after identity and file checks. A partial job
is deliberately not overwritten: use a new `--out` directory to rerun it and
retain the partial directory for diagnosis. Never collect both copies of a job.
Smoke tests are separate, flagged, and use short chains (not usable inference).

## Colab and collection

`build_colab.py` creates 48 self-contained notebooks, each embedding the same
source/data/manifest bundle and installing the pinned numerical dependencies.
Choose `PHASE` in its first code cell. Downloaded output is a single zip with
an automatic Colab download and a non-Colab fallback. The notebooks stop
between jobs after eight hours and preserve already completed jobs if a later
fit raises an exception. A forced runtime termination cannot execute the
download fallback; download/checkpoint before the runtime limit.

Simulation scalar draws are saved as `(chain, retained_draw)` arrays for ATT,
GATT, event-time effects and variance. Application and diagnostic jobs also
save full effect draws and their panel for reconstruction; allow substantial
disk space (uncompressed diagnostic draws alone can exceed 75 GB over the
whole campaign). This favors auditability over minimal storage. Do not start
all phases at once without a storage plan. No GPU is needed.

Before rebuilding paper tables, collect completed production jobs by manifest
and job ID, reject duplicates/conflicts and smoke files, verify all required
replications, and assess mixing, interval coverage with Monte Carlo uncertainty,
and prior sensitivity. Historical comparator reuse requires matched data seeds,
DGP parameters and estimands. New null reps 100:199 may require comparator
extensions. No claim of submission readiness follows from this repair alone.
