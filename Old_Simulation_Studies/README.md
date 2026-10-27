# Development history and superseded experiments

This folder preserves earlier DiD-BCF implementations, intermediate experiments, smoke tests, correction studies and superseded results for transparency. A file's presence here does not imply that its estimator, assumptions, numerical claims or intervals were accepted for the final paper. Earlier READMEs retain their original, sometimes outdated, descriptions.

For current results and model use, start with the [repository README](../README.md), [final result guide](../RESULTS_VISUALIZATION_HANDOFF.md), and [model instructions](../MODEL_USAGE.md). Current repaired main and non-main campaigns remain under [`Simulation_Studies_Revision/DiD_BCF/`](../Simulation_Studies_Revision/DiD_BCF/).

## Directory map

| Folder | Contents and interpretation |
|---|---|
| [`Original/`](Original/) | Original simulation generation: DGPs, DiD-BCF and benchmark notebooks, and saved results before the revision. |
| [`Tau=0/`](Tau=0/) | Earlier sharp-null experiments and associated code/results. |
| [`Revision_DiD_BCF/Proper_Prior_Rerun/`](Revision_DiD_BCF/Proper_Prior_Rerun/) | Earlier campaign definitions, proper-prior checks and verification records. These are not the final repaired-engine production entry point. The `Paper_Rerun_20261011` compatibility link points to the current main package. |
| [`Revision_DiD_BCF/DiD_BCF/Theory_Calibration/`](Revision_DiD_BCF/DiD_BCF/Theory_Calibration/) | Earlier correction-audit, information-ablation and validation notebooks. |
| [`Revision_DiD_BCF/DiD_BCF/Correction_Completion_Notebooks/`](Revision_DiD_BCF/DiD_BCF/Correction_Completion_Notebooks/) | Earlier completion notebooks for reduced-data/reference-fold comparisons. Some resulting comparator metrics are still used, explicitly labeled, in the paper. |
| [`Revision_DiD_BCF/Reruns/`](Revision_DiD_BCF/Reruns/) | Earlier proper-prior smoke and rerun outputs, including diagnostic evidence. |
| [`Revision_DiD_BCF/Pretrend/`](Revision_DiD_BCF/Pretrend/) | Superseded pretrend notebooks and replication-block results. `_retired/` retains overlapping/superseded blocks. Use the final non-main campaign for current diagnostic performance. |
| [`Revision_DiD_BCF/Theory_Calibration/`](Revision_DiD_BCF/Theory_Calibration/) | Earlier smoke results and the historical reconciliation document with its generated tables and build files. Current paper comparator inputs remain in `../Simulation_Studies_Revision/Theory_Calibration/results/`. |
| [`Revision_DiD_BCF/scripts/`](Revision_DiD_BCF/scripts/) | Superseded simulation and empirical notebook generators and unpatched runner entry points. |
| [`Revision_DiD_BCF/Results_development/`](Revision_DiD_BCF/Results_development/) | Earlier identification discussion and metric-maintenance notes, retained as development records. |
| [`Empirical_Study/`](Empirical_Study/) | Earlier application notebooks, fitted-effect CSVs and plots. The application dataset remains at `../Empirical_Study/mpdta.csv`; current results are in the repaired empirical result directory. |
| [`_moved_from_repo/`](_moved_from_repo/) | Earlier revision-generation notebooks/results and verification snapshots, organized by their former repository paths. In particular, `Simulation_Studies_Revision/DiD_BCF_old_generation/` contains the old raw/corrected studies. |
| [`Documentation_before_20261027/`](Documentation_before_20261027/) | Previous repository, workflow, result, model and archive-map documentation, retained before the current navigation cleanup. |

## Understanding the successive implementations

The original unrestricted prognostic specification, the additive level/trend structured specification, changes to the variance prior, and the repaired native tree transition are distinct changes. An older result labeled `plain`, `structured`, `corrected` or `proper_ig` does not by itself identify the final computation. Check its manifest, source snapshot, prior, engine and budget. In particular, the historical posterior-correction/reference-fold estimator is a different procedure from repairing the native sampler.

The final production campaigns use the additive structured Python model, the frozen repaired native engine, a proper IG(2,1) standardized noise-variance prior, 50 GFR sweeps, no additional burn-in, 500 retained draws per chain, thinning five, and three chains. Rates 0.1 and 10 are separate sensitivity settings. The short production budget was selected under the paper-performance policy; it does not establish convergence of every finite MCMC run or universal nominal interval coverage. Supporting diagnostic shortcomings remain part of the record.

Historical benchmark and reference-fold aggregates needed by the paper intentionally remain in the active `Results/aggregated/` and `Theory_Calibration/results/` locations. Their role is documented there. Moving a workflow here does not relabel its results as current DiD-BCF.

## Preservation and reproducibility

The [2026-10-27 reorganization record](REORGANIZATION_20261027.json) lists source/destination paths, byte counts and SHA256 hashes for moved files. The [earlier migration record](Revision_DiD_BCF/MIGRATION_RECORD_20261014.json) describes the preceding folder migration. Experiment payloads were moved intact, not deleted or gitignored. Frozen final campaign sources, assignments, reuse records and collected metrics were left unchanged.

These old notebooks preserve their original import paths and Git pins. When an old workflow assumes its former layout, use the recorded historical Git revision rather than treating the archived path as a new supported command. Current experiment notebooks remain runnable from their frozen packages.

Some much larger pilot posterior-array ZIPs had already been kept in the author's separate local archive before this cleanup. This folder is not a claim that every local runtime artifact has been uploaded. The [archive map](../EXPERIMENT_ARCHIVE_MAP.md) records that earlier storage decision. All final paper metric collections and source packages are committed; none of the final experiment results is ignored.
