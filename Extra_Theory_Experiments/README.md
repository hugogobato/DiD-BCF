# Extra Theory Experiments

This folder is a disposable, self-contained calibration package. It scientifically
separates six questions that were confounded in the earlier simulation driver:

1. Does the raw structured DiD-BCF posterior behave differently from post-processing?
2. How much changes under the current same-sample hybrid correction?
3. Does a fixed-K, genuinely off-fold reference construction behave differently?
4. Are extreme inverse odds driving results, and what is the effect of clipping or
   a different propensity learner?
5. Can an exact nuisance oracle be used for a DGP/cell, or must that comparison be
   marked unavailable?
6. How much information is lost by reducing the full panel to a two-group,
   two-period cell or by pooling cohorts?

The package reuses the production sampler through imports, but all adapters,
manifests, metrics, notebooks, and result files here are local additions.

## Estimator definitions and labels

"raw_structured" fits the production structured sampler on the full panel and
reports its posterior treatment-effect draws. The word structured describes
the two-way restricted BCF likelihood. It is not the same word as
DR-corrected or posterior-corrected.

"current_hybrid" is the local copy of the current cell correction. For cell
(g,t), let delta=1{G=g}, DeltaY=Y_t-Y_(g-1), M[i,s] be the posterior
control-arm long difference, barpi=mean(delta), and W[i,s] normalized
unit-level Exp(1) weights. It computes
theta_s = sum_i W[i,s] [delta_i-(1-delta_i)pi_i/(1-pi_i)] [DeltaY_i-M[i,s]]
/ sum_i W[i,s]delta_i, then subtracts
b_s = n_g^(-1) sum_i [(delta_i-pi_i)/((1-pi_i)barpi)] [mean_s M[i,s]-M[i,s]].
This is deliberately labeled same-sample "current_hybrid", not
theorem-faithful.

"reference_fold_convolution" partitions original units once. Every row of a unit
stays in its fold. For each cell and fold, the BCF posterior sees only that fold's
two groups and all their panel rows. Outcome, propensity, and bar-pi pilots use
only complementary cell units. Fold posterior and Bayesian-bootstrap draws have
independent deterministic seeds, with one cached posterior per (cohort, fold)
and one global BB multiplier map per fold reused across calendar cells. Fold
draws are convolved using exact selected two-group cell sizes N_{g,k}=|I_k
intersection S_g|, not treated counts. The posterior median is the theorem-aligned point summary;
the mean is retained as a diagnostic. Fixed sampler draws are finite-sample
approximations, so this name never means theorem-validated.

Aggregate ATT and event-study weights are empirical treated-cell weights. Their
use is a diagnostic aggregation and is not automatically theorem-covered.

Propensity choices are "logit", "rf", and "intercept". The reference default
uses a flexible off-fold random-forest outcome pilot; "intercept" is retained
as a diagnostic. Clipping reports raw and
clipped control odds, effective sample size, maximum normalized control weight,
number clipped, and fold sizes. Clipping is an exploratory stabilization and is
not theoretically neutral.

Oracle requests require exact "m0_oracle", "pi_oracle", and "barpi_oracle"
columns. The production baseline, null, and serial DGPs include logistic-normal
assignment noise, so their exact propensity is not exposed; their oracle tasks
remain explicitly "unavailable", rather than silently using a pilot. The
additional local `oracle_canonical` design is oracle-ready: treatment is sampled
from the known Bernoulli probability `pi(X)`, the control long-difference mean is
the stored `m0_oracle(X,g,t)`, and `barpi_oracle` is the exact population cell
share 0.5, not a sample mean of evaluation propensities. Specifically,
X1 is Bernoulli with z1=2X1-1, X2:X5 are iid Uniform[-1,1], and the joint
sign-flip T(X)=(1-X1,-X2,...,-X5) gives `pi(TX)=1-pi(X)`. Therefore
E[pi(X)]=0.5, while the bounded degree-1 and degree-2 indices keep
`0.001 < pi(X) < 0.999` without clipping. It keeps the same GATT/ES/ATT
definitions and panel columns, so its two oracle estimators actually run without
changing the estimand. The degree-1 and degree-2 formulas are recorded in the
DGP module and its metadata. The default has `alpha_sd=0` and iid Gaussian row
errors, so it stays within the intended conditional-independence error setup;
nonzero alpha is allowed only as an explicitly flagged `non_theorem_stress=True`
option. Heterogeneous effects are always refused because this adapter does not
implement their exact population GATT. The oracle DGP stores
`gatt_population_oracle=3` and the runner labels joined truths with
`truth_source="gatt_population_oracle"`, asserting that every realized
GATT/ES/ATT truth remains exactly 3.

Information ablations compare "full_panel_raw", "reduced_cell", and
"pooled_full_panel". The reduced fit uses only cohort g and never-treated units
at periods g-1 and t. The local production adapter bypasses the production
wrapper and directly calls `StructuredDiDBCF.sample`, passing
`effect_by_cohort=False` for `pooled_full_panel` (and `True` for the raw
full-panel fit), so pooling is an actual fit variant. Pooling remains
exploratory because its target comparison is not theorem-covered. No full-panel
EIF is invented. Center-preserving multiplier intervals are disabled by default
and raise an error if requested.

## Metrics and truth

Truth is joined by replication, estimand type, and estimand ID. A GATT truth is
never broadcast into a CATT surface. For each point summary (mean and median),
scalar Monte Carlo empirical SD is SD(point - true) and is reported as
empirical_sd_error/emp_sd; raw SD(point) is separately named raw_sd_est. Bias MCSE is
empirical_sd_error/sqrt(n_reps). The stored two-sided Bayesian p-value is
min(1, 2 min(P(draw >= 0), P(draw <= 0))); the one-tail minimum is retained as
p_bayes_tail_min.

Simulations motivate but do not prove any theorem. Results can change with
finite posterior and bootstrap draw budgets, and posterior mean versus median
can matter. Extreme inverse odds are always recorded even in clipped variants.
Real-data mpdta comparisons are out of scope.

## Configurations and shard accounting

configs/correction_audit.json contains B1 baseline, null, and serial-correlation
designs (including the oracle-ready `oracle_canonical` design), degrees 1 and 2,
N in {200,800}, 100 replications, raw/current/reference,
intercept/logit/RF reference pilots, stabilized current logit variants at .01
and .05, and explicit unavailable oracle rows. It expands to 13,200 manifest
tasks for the original three production designs, plus 4,400 additional
oracle-canonical/production rows, for 17,600 total tasks. Of the 3,200 oracle
requests, 800 oracle-canonical rows run and 2,400 production-DGP rows are
explicitly unavailable. All methods for a
design-degree-N-replication bundle share one shard and paired full-panel fit
cache where applicable.

configs/information_ablation.json contains baseline, serial, and staggered
designs at selected degrees and N values, with full-panel, reduced-cell, and
pooling variants. It expands to 3,600 tasks. Use build_manifest(..., reps=2)
for a pilot. Sharding is deterministic after sorting by design, degree, N,
and replication, with every method in one bundle on the same shard. There are
48 generated compute notebooks per family, plus one lightweight validation
notebook, three one-replication worst-case real-BCF pilot notebooks, and one
controlled-mechanics notebook (101 notebooks total). Each compute notebook
accepts `ETE_N_WAVES` and `ETE_WAVE_ID`; the same 48 notebooks can be reused
for successive waves without overwriting outputs, because wave identifiers are
included in output directories, archives, manifests, and provenance. Pilots
remain fixed at wave 0 of 1.
The correction pilot selects exactly serial, degree 2, N=800,
rep 0, with raw, current-logit, and reference-logit estimators. The information
pilot selects exactly staggered, degree 3, N=800, rep 0, with full-panel,
reduced-cell, and pooled estimators. The dedicated oracle pilot selects exactly
`oracle_canonical`, degree 2, N=800, rep 0, with the two oracle estimators; it
checks the production-sampler path against the exact nuisance columns, iid
default errors, and homogeneous truth 3 at the tiny budget. Each compute notebook uses
at most two workers, defaults to one, resumes per-task CSV checkpoints, and writes
a manifest, CSV/Parquet summary, provenance JSON, and one zip archive. After
timing pilots, choose correction and information wave counts separately so a
wave-shard has a conservative predicted runtime below nine hours. The
approximate cached fit load per notebook is 100/n_waves for correction and
about 192/n_waves for information. The 48 notebooks are reused for every wave,
minimizing uploads.

Approximate wall time is 1 to 5 minutes per small N=200 structured fit and 5 to
20 minutes per N=800 fit with a tiny sampler budget; production budgets can be
materially longer. Reference fits are cached once per cohort-fold and are not
multiplied by the number of calendar cells, although they remain a dominant
cost. The pilot gate is mandatory: run all three representative worst-case pilots
with tiny budgets,
inspect convergence, fold diagnostics, odds/ESS, and memory, then increase
budgets and only proceed if one shard fits within Colab's 10-hour limit.
Sessions are assumed to have about two usable CPU cores and 12 GB RAM. GPUs are
not assumed to accelerate the sampler.

Compute notebooks expose the production study budget, num_gfr=50,
num_mcmc=500, keep_every=5, num_chains=3. Each worst-case pilot uses one
replication with a tiny budget. The correction and dedicated oracle pilots
should each yield three cached sampler fits, while the staggered information
pilot should yield eleven. Their timing is a hard gate: do not launch the
100-replication waves until all three pilots demonstrate that the assigned work
fits within the ten-hour Colab limit.

## Controlled mechanics special case

`controlled_mechanics.ipynb` and `scripts/run_controlled_mechanics.py` isolate
algebra, Bayesian-bootstrap, and fold-convolution mechanics without fitting BCF.
For both a homogeneous signal (`tau=3`) and a homogeneous null (`tau=0`), every
prognostic posterior draw is fixed at the exact `m0_oracle`. The diagnostic
compares the full-sample Algorithm-1 BB law with the K=2
`reference_fold_convolution` law, reusing original-unit multipliers across
calendar cells and using selected two-group cell-size weights. Default settings
are 100 replications, 300 draws, and N=200, with `ETE_MECH_REPS`,
`ETE_MECH_DRAWS`, and `ETE_MECH_UNITS` overrides for Colab or pilots. It writes
per-replication and aggregate CSV files reporting GATT(g,t)/ATT bias, 95%
coverage, interval length, and null rejection, together with manifest,
provenance, and one zip archive. This is a controlled special-case diagnostic,
not a theorem proof.

## Colab

Upload any generated notebook. It clones the GitHub repository at branch
experiments/theory-calibration-colab, installs requirements-colab.txt, adds
this folder's src to sys.path, runs the selected shard, resumes local
checkpoints, and zips every artifact. Since a run creates five or more artifacts,
only the single zip is downloaded. The exact fallback used by each notebook is:

    try:
        from google.colab import files
        files.download(output_file)
        print("Downloaded:", output_file)
    except Exception as e:
        print("(Not on Colab / download skipped):", e)

Locally, set PYTHONPATH=Extra_Theory_Experiments/src:. and run the smoke path
from scripts/run_smoke.py. The notebooks do not rely on files outside the
clone. After downloads, aggregate locally with
python Extra_Theory_Experiments/scripts/aggregate_archives.py
<download-directory> --output-dir aggregated --n-shards 48. The aggregator
checks archive paths, manifests, config hashes, shard and wave coverage,
duplicate exact task/estimand/method keys, and missing bundles, then emits
separate mean and median error metrics. It accepts `--n-waves` when auditing a
collection, and infers the declared count from wave-aware manifests. Archives
without wave fields remain valid as legacy one-wave inputs. Oracle-canonical rows are aggregated as ordinary
estimates; explicit oracle-unavailable rows are retained as unavailable rather
than treated as failed estimates. Delete this folder to remove every added
package, notebook, result, and checkpoint.

Aggregate correction and information archives separately (use separate download
directories or `--family`), because the two families can legitimately use
different wave counts and config hashes. Do not mix pilot or controlled-mechanics
archives into a 48-shard aggregation directory.
