# Experiment matrix and pilot gate

The matrix is defined by the JSON configs, so this table records expected fit
multipliers and the scheduling assumptions.

| family | grid | estimator classes | approximate fits per replication | shard plan |
|---|---|---|---:|---|
| correction audit | 4 designs x 2 degrees x 2 N | 1 raw, 5 current pilots, 3 reference propensity variants, 2 oracle variants (oracle_canonical available; production DGP rows unavailable) | 3 unique sampler fits after cache (1 full + 2 cohort-fold fits), with oracle rows using the same cached fits | 48 shards per family |
| information ablation | 3 designs x selected degrees x 2 N | full (1), pooled (1), reduced (4 cells for canonical, 9 for staggered) | 6 to 11 fits | 48 shards per family |

The correction audit has 17,600 attempted task rows: 800 oracle-canonical oracle
rows run, while 2,400 oracle rows for the production DGPs are explicitly
unavailable. The default oracle-canonical rows have iid Gaussian row errors and
homogeneous population GATT/ES/ATT truth equal to 3. The available sampler
workload is about 4,800 fits after caching
one full fit and one posterior per (cohort, fold) across reference propensity
variants. The information family has 3,600 attempted rows and about
9,200 sampler fits.
Reference cell counts depend on adoption timing, so staggered designs can have
more than four cells. Reference posterior fits are cached once per cohort-fold,
not multiplied by the number of calendar cells.

The 48 compute notebooks are reusable across waves. After the timing pilots,
choose `n_waves` separately by family so each wave-shard has a conservative
predicted runtime below nine hours. The approximate cached fit load per
notebook is 100/n_waves for correction and about 192/n_waves for information.
Set `ETE_N_WAVES` and `ETE_WAVE_ID` in a notebook, and aggregate all waves
together. Bundle allocation is deterministic over the wave-shard Cartesian
product, so no replication is split across methods or duplicated across waves.

Run the three representative worst-case pilots with one replication and tiny
sampler settings such as num_gfr=2, num_mcmc=8, keep_every=2, num_chains=1.
The correction pilot is serial, degree 2, N=800, rep 0 with raw/current-logit/
reference-logit and should use three cached sampler fits. The information pilot
is staggered, degree 3, N=800, rep 0 with full/reduced/pooled and should use
eleven cached sampler fits. The dedicated oracle pilot is oracle_canonical,
degree 2, N=800, rep 0 with the two oracle estimators and should use three
cached sampler fits. Verify that every requested method has a row or an
explicit unavailable record, that no unit appears in multiple folds, and that
propensity diagnostics expose raw odds, clipped odds, ESS, normalized weights,
and fold sizes. Compare posterior median and mean, and inspect the
checkpoint/resume result before scaling.

Use one worker for memory-heavy structured/reference fits. At most two workers
are allowed on a Colab session with approximately 12 GB RAM. A tiny N=200 fit
is usually 1 to 5 minutes; N=800 is often 5 to 20 minutes with the tiny budget,
and full production budgets may be much longer. Reference fits are cached once
per cohort-fold rather than repeated across calendar cells, and are reused
across propensity variants. Stop or rebalance a shard if any pilot
suggests it could exceed the 10-hour session cap. GPUs are not assumed to help
the sampler.

The matrix is exploratory evidence only. It does not establish identification,
a Bernstein-von Mises result, efficiency, or any theorem. Clipping and finite
posterior/BB draws must be reported with the outputs.

The separate controlled mechanics audit is not part of the sampler fit counts.
It fixes posterior prognostic draws at exact m0 and reports the full-sample BB
and K=2 fold-convolution laws for homogeneous signal and null scenarios. It is a
special-case algebra diagnostic, not a theorem proof.
