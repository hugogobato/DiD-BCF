# Local intermediate experiment archive

On 2026-10-11 the user requested that intermediate and obsolete experiment outputs be excluded from future repository publication. They were moved intact into `.local_experiment_archive/intermediate/20261011/`, with atomic renames on the same filesystem. The detailed file counts, byte totals and original-to-archive mappings are in that local directory's `ARCHIVE_MAP.json`.

The moved materials are the repaired 30-replication comparison package and its returned ZIPs, the earlier unpatched proper-prior Colab package and root returned ZIPs, its returned-results analysis and longer-chain pilot outputs, and `Simulation_Studies_Revision/DiD_BCF/Correction_Completion_Results/`. The correction source notebooks and existing aggregated theory-calibration results remain available. Historical manuscript comparator assets were preserved. Git history was not rewritten.

The only added ignore rule is `/.local_experiment_archive/`. The final [paper rerun package](Simulation_Studies_Revision/Proper_Prior_Rerun/Paper_Rerun_20261011/README.md), its compact verified reuse, and its returned `Results/` folder remain publishable and are not ignored. The original proper-prior job manifest and model source remain available as the campaign definitions.

Local comparison reporting scripts use the archived package location. Frozen historical records retain their original input paths as provenance; the local archive map resolves those locations. Other clones will contain the final compact evidence and code rather than the unpublished intermediate archives.
