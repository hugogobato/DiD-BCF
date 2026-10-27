# Local intermediate experiment archive

On 2026-10-11 the user requested that intermediate and obsolete experiment outputs be excluded from future repository publication. They were moved intact into `.local_experiment_archive/intermediate/20261011/`, with atomic renames on the same filesystem. The detailed file counts, byte totals and original-to-archive mappings are in that local directory's `ARCHIVE_MAP.json`.

The moved materials are the repaired 30-replication comparison package and its returned ZIPs, the earlier unpatched proper-prior Colab package and root returned ZIPs, its returned-results analysis and longer-chain pilot outputs, and `Simulation_Studies_Revision/DiD_BCF/Correction_Completion_Results/`. The correction source notebooks and existing aggregated theory-calibration results remain available. Historical manuscript comparator assets were preserved. Git history was not rewritten.

Only `/.local_experiment_archive/` is excluded from Git. The current [paper rerun package](Simulation_Studies_Revision/DiD_BCF/Paper_Rerun_20261011/README.md), its sealed source snapshot, verified reuse, and returned results remain publishable. Earlier proper-prior campaign definitions and correction workflows are preserved under `Old_Simulation_Studies/Revision_DiD_BCF/`.

Local comparison reporting scripts use the archived package location. Frozen historical records retain their original input paths as provenance; the local archive map resolves those locations.

## Experiment layout migration (2026-10-14)

The current repaired campaign moved intact to `Simulation_Studies_Revision/DiD_BCF/Paper_Rerun_20261011/`. The remaining proper-prior campaign files moved to `Old_Simulation_Studies/Revision_DiD_BCF/Proper_Prior_Rerun/`, and the earlier correction and completion notebook folders moved to `Old_Simulation_Studies/Revision_DiD_BCF/DiD_BCF/`. The broad ignore for `Old_Simulation_Studies/` was removed so these historical files and experiment results remain visible to Git. `.local_experiment_archive/` remains separately ignored.

Two compatibility symlinks preserve historical entry points: `Simulation_Studies_Revision/Proper_Prior_Rerun/` resolves to the archived proper-prior campaign, and its `Paper_Rerun_20261011/` child resolves to the current campaign. File counts, byte totals, and per-file SHA-256 inventories are recorded in `Old_Simulation_Studies/Revision_DiD_BCF/MIGRATION_RECORD_20261014.json`.

The migration preserved these payloads: the current paper rerun has 1,431 files totaling 173,186,849 bytes; the earlier proper-prior campaign remainder has 10 files totaling 24,176,495 bytes; the correction-completion notebook folder has 219 files totaling 1,398,766 bytes; and the theory-calibration notebook folder has 101 files totaling 559,260 bytes. These totals exclude symlinks and directories. The migration record contains full per-file inventories and hashes.
