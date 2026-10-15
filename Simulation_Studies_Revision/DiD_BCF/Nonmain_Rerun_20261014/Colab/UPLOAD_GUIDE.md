# Non-main Colab upload guide

Upload each generated `.ipynb` directly into a CPU Colab runtime and choose **Run all**. The notebooks clone and verify immutable helper commit `30b165b27610dc4b2c406a9d64281d77e660257a` in the dedicated `unique_nonmain` folder. The package pins Python dependencies and the repaired engine. No global stochtree package is installed.

This guide is generated from the frozen assignments and manifest. It contains 132 Colab notebooks covering 9,197 assigned fits across wave_01: 48 notebooks, wave_02: 48 notebooks, wave_03: 36 notebooks. Run up to 48 notebooks concurrently within a wave. Each notebook uses at most two single-threaded workers, with memory-based admission reducing that number when needed. The package carries 8 verified reused fit(s), which do not create Colab jobs. The local assignments contain 22 fits (application: 6, application_sensitivity: 16); those jobs do not have Colab notebooks and their returned metrics ZIP can be included in the collector.

The manifest contains application: 9, application_sensitivity: 18, diagnostic: 4,000, pt_impact: 4,000, sensitivity: 1,200 fits by study phase. The assignments schedule diagnostic: 3,999, pt_impact: 4,000, sensitivity: 1,198 on Colab and application: 6, application_sensitivity: 16 locally. This describes the campaign workload, not completion status. The collector reports collected and missing fits separately for every phase.

Each notebook targets its scheduled wall time, usually about 8 hours, and stops within 9.5 hours including setup. Enable `USE_DRIVE` before running to copy a metrics-only ZIP checkpoint to Drive every 20 minutes. To resume, run the same notebook and provide its ZIP through `RESUME_ZIP`; Drive checkpoints are detected automatically when enabled. Only verified complete fits resume. An interrupted fit restarts from its original seed. Native arrays and build products stay on the runtime disk, and each notebook downloads one metrics ZIP.

Upload the simulation and local application ZIPs to `collect_nonmain_metrics.ipynb`. To provide a ZIP already present in the Colab runtime, add its path to `INPUT_PATHS`. The collector validates the non-main campaign, reports missing jobs by phase, and downloads one ZIP of collected tables. Partial collections remain available with explicit missing-fit counts.

Planning estimates from returned complete-fit worker wall times. Two-worker Colab; hardware variation or memory constraints may require resume.

| Wave and assignment | Study phases | Fits | Estimated hours with two workers |
| --- | --- | ---: | ---: |
| [nonmain_wave_01_shard_00](wave_01/nonmain_wave_01_shard_00.ipynb) | diagnostic: 28, pt_impact: 30, sensitivity: 12 | 70 | 8.0 |
| [nonmain_wave_01_shard_01](wave_01/nonmain_wave_01_shard_01.ipynb) | diagnostic: 28, pt_impact: 30, sensitivity: 12 | 70 | 8.0 |
| [nonmain_wave_01_shard_02](wave_01/nonmain_wave_01_shard_02.ipynb) | diagnostic: 28, pt_impact: 30, sensitivity: 12 | 70 | 8.0 |
| [nonmain_wave_01_shard_03](wave_01/nonmain_wave_01_shard_03.ipynb) | diagnostic: 28, pt_impact: 30, sensitivity: 12 | 70 | 8.0 |
| [nonmain_wave_01_shard_04](wave_01/nonmain_wave_01_shard_04.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_05](wave_01/nonmain_wave_01_shard_05.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_06](wave_01/nonmain_wave_01_shard_06.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_07](wave_01/nonmain_wave_01_shard_07.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_08](wave_01/nonmain_wave_01_shard_08.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_09](wave_01/nonmain_wave_01_shard_09.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_10](wave_01/nonmain_wave_01_shard_10.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_11](wave_01/nonmain_wave_01_shard_11.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_12](wave_01/nonmain_wave_01_shard_12.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_13](wave_01/nonmain_wave_01_shard_13.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_14](wave_01/nonmain_wave_01_shard_14.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_15](wave_01/nonmain_wave_01_shard_15.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_16](wave_01/nonmain_wave_01_shard_16.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_17](wave_01/nonmain_wave_01_shard_17.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_18](wave_01/nonmain_wave_01_shard_18.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_19](wave_01/nonmain_wave_01_shard_19.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_20](wave_01/nonmain_wave_01_shard_20.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_21](wave_01/nonmain_wave_01_shard_21.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_22](wave_01/nonmain_wave_01_shard_22.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_23](wave_01/nonmain_wave_01_shard_23.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_24](wave_01/nonmain_wave_01_shard_24.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_25](wave_01/nonmain_wave_01_shard_25.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_26](wave_01/nonmain_wave_01_shard_26.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_27](wave_01/nonmain_wave_01_shard_27.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_28](wave_01/nonmain_wave_01_shard_28.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_29](wave_01/nonmain_wave_01_shard_29.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_30](wave_01/nonmain_wave_01_shard_30.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_31](wave_01/nonmain_wave_01_shard_31.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_01_shard_32](wave_01/nonmain_wave_01_shard_32.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_33](wave_01/nonmain_wave_01_shard_33.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_34](wave_01/nonmain_wave_01_shard_34.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_35](wave_01/nonmain_wave_01_shard_35.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_36](wave_01/nonmain_wave_01_shard_36.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_37](wave_01/nonmain_wave_01_shard_37.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_38](wave_01/nonmain_wave_01_shard_38.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_39](wave_01/nonmain_wave_01_shard_39.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_40](wave_01/nonmain_wave_01_shard_40.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_01_shard_41](wave_01/nonmain_wave_01_shard_41.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_01_shard_42](wave_01/nonmain_wave_01_shard_42.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_01_shard_43](wave_01/nonmain_wave_01_shard_43.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_01_shard_44](wave_01/nonmain_wave_01_shard_44.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_01_shard_45](wave_01/nonmain_wave_01_shard_45.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_01_shard_46](wave_01/nonmain_wave_01_shard_46.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_01_shard_47](wave_01/nonmain_wave_01_shard_47.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_02_shard_00](wave_02/nonmain_wave_02_shard_00.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_02_shard_01](wave_02/nonmain_wave_02_shard_01.ipynb) | diagnostic: 31, pt_impact: 30, sensitivity: 10 | 71 | 8.0 |
| [nonmain_wave_02_shard_02](wave_02/nonmain_wave_02_shard_02.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_03](wave_02/nonmain_wave_02_shard_03.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_04](wave_02/nonmain_wave_02_shard_04.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_05](wave_02/nonmain_wave_02_shard_05.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_06](wave_02/nonmain_wave_02_shard_06.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_07](wave_02/nonmain_wave_02_shard_07.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_08](wave_02/nonmain_wave_02_shard_08.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_09](wave_02/nonmain_wave_02_shard_09.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_10](wave_02/nonmain_wave_02_shard_10.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_11](wave_02/nonmain_wave_02_shard_11.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_12](wave_02/nonmain_wave_02_shard_12.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_13](wave_02/nonmain_wave_02_shard_13.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_14](wave_02/nonmain_wave_02_shard_14.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_15](wave_02/nonmain_wave_02_shard_15.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_16](wave_02/nonmain_wave_02_shard_16.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_17](wave_02/nonmain_wave_02_shard_17.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_18](wave_02/nonmain_wave_02_shard_18.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_19](wave_02/nonmain_wave_02_shard_19.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_20](wave_02/nonmain_wave_02_shard_20.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_21](wave_02/nonmain_wave_02_shard_21.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_22](wave_02/nonmain_wave_02_shard_22.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_23](wave_02/nonmain_wave_02_shard_23.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_24](wave_02/nonmain_wave_02_shard_24.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_25](wave_02/nonmain_wave_02_shard_25.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_26](wave_02/nonmain_wave_02_shard_26.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_27](wave_02/nonmain_wave_02_shard_27.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_28](wave_02/nonmain_wave_02_shard_28.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_29](wave_02/nonmain_wave_02_shard_29.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_30](wave_02/nonmain_wave_02_shard_30.ipynb) | diagnostic: 31, pt_impact: 31, sensitivity: 7 | 69 | 8.0 |
| [nonmain_wave_02_shard_31](wave_02/nonmain_wave_02_shard_31.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_32](wave_02/nonmain_wave_02_shard_32.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_33](wave_02/nonmain_wave_02_shard_33.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_34](wave_02/nonmain_wave_02_shard_34.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_35](wave_02/nonmain_wave_02_shard_35.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_36](wave_02/nonmain_wave_02_shard_36.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_37](wave_02/nonmain_wave_02_shard_37.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_38](wave_02/nonmain_wave_02_shard_38.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_39](wave_02/nonmain_wave_02_shard_39.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_40](wave_02/nonmain_wave_02_shard_40.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_41](wave_02/nonmain_wave_02_shard_41.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_42](wave_02/nonmain_wave_02_shard_42.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_43](wave_02/nonmain_wave_02_shard_43.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_44](wave_02/nonmain_wave_02_shard_44.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_45](wave_02/nonmain_wave_02_shard_45.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_46](wave_02/nonmain_wave_02_shard_46.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_02_shard_47](wave_02/nonmain_wave_02_shard_47.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_00](wave_03/nonmain_wave_03_shard_00.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_01](wave_03/nonmain_wave_03_shard_01.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_02](wave_03/nonmain_wave_03_shard_02.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_03](wave_03/nonmain_wave_03_shard_03.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_04](wave_03/nonmain_wave_03_shard_04.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_05](wave_03/nonmain_wave_03_shard_05.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_06](wave_03/nonmain_wave_03_shard_06.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_07](wave_03/nonmain_wave_03_shard_07.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_08](wave_03/nonmain_wave_03_shard_08.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_09](wave_03/nonmain_wave_03_shard_09.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_10](wave_03/nonmain_wave_03_shard_10.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_11](wave_03/nonmain_wave_03_shard_11.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_12](wave_03/nonmain_wave_03_shard_12.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_13](wave_03/nonmain_wave_03_shard_13.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_14](wave_03/nonmain_wave_03_shard_14.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_15](wave_03/nonmain_wave_03_shard_15.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_16](wave_03/nonmain_wave_03_shard_16.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_17](wave_03/nonmain_wave_03_shard_17.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_18](wave_03/nonmain_wave_03_shard_18.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_19](wave_03/nonmain_wave_03_shard_19.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 10 | 70 | 8.0 |
| [nonmain_wave_03_shard_20](wave_03/nonmain_wave_03_shard_20.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_21](wave_03/nonmain_wave_03_shard_21.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_22](wave_03/nonmain_wave_03_shard_22.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_23](wave_03/nonmain_wave_03_shard_23.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_24](wave_03/nonmain_wave_03_shard_24.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_25](wave_03/nonmain_wave_03_shard_25.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_26](wave_03/nonmain_wave_03_shard_26.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_27](wave_03/nonmain_wave_03_shard_27.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_28](wave_03/nonmain_wave_03_shard_28.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_29](wave_03/nonmain_wave_03_shard_29.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_30](wave_03/nonmain_wave_03_shard_30.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_31](wave_03/nonmain_wave_03_shard_31.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_32](wave_03/nonmain_wave_03_shard_32.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_33](wave_03/nonmain_wave_03_shard_33.ipynb) | diagnostic: 30, pt_impact: 30, sensitivity: 9 | 69 | 8.0 |
| [nonmain_wave_03_shard_34](wave_03/nonmain_wave_03_shard_34.ipynb) | diagnostic: 30, pt_impact: 31, sensitivity: 9 | 70 | 8.0 |
| [nonmain_wave_03_shard_35](wave_03/nonmain_wave_03_shard_35.ipynb) | diagnostic: 30, pt_impact: 31, sensitivity: 9 | 70 | 8.0 |
