# Upload guide

These are loose notebooks. Upload each `.ipynb` directly to a CPU Colab runtime and select **Run all**. They clone immutable helper commit `3078e2ca1155b932feab92c722d697803370b8c7` and compile an isolated repaired engine. No global stochtree package is installed.

Run the 48 notebooks in `wave_01/`, followed by the 12 in `wave_02/`. Each main notebook targets about eight hours with two workers. Run the four `Bridge_Checks/` notebooks as well; these cover eight non-main anchors and compare against saved B3 summaries. They are shorter and can run alongside the main campaign.

Each notebook downloads one small metrics ZIP. Move returned ZIPs to `../Results/` for local collection. Enable `USE_DRIVE` before execution to persist a metrics checkpoint every 20 minutes. For recovery, run the same notebook again, upload its ZIP through the Files pane and set `RESUME_ZIP`; Drive checkpoints are detected automatically when enabled. If the runtime ends abruptly without Drive or a downloaded checkpoint, its local files are lost. A resumed notebook rebuilds its engine and skips verified complete fits. An interrupted fit restarts from its original seeds.

Collect with `collect_paper_metrics.ipynb`, or locally:

```text
python helpers/collect.py Results/didbcf_paper_*.zip --out Results/collected --require-main
```

The collector includes the 264 reused main fits. A complete main collection has 5,100 fits. Diagnostics do not select or exclude results. The remaining full paper phases still need budget decisions after the eight checks. Existing manuscript numbers are unchanged until final results are available.

| Notebook | Estimated hours with two workers |
| --- | ---: |
| [bridge_checks_00](Bridge_Checks/bridge_checks_00.ipynb) | 1.5 |
| [bridge_checks_01](Bridge_Checks/bridge_checks_01.ipynb) | 2.0 |
| [bridge_checks_02](Bridge_Checks/bridge_checks_02.ipynb) | 0.5 |
| [bridge_checks_03](Bridge_Checks/bridge_checks_03.ipynb) | 1.5 |
| [main_wave_01_shard_00](wave_01/main_wave_01_shard_00.ipynb) | 8.0 |
| [main_wave_01_shard_01](wave_01/main_wave_01_shard_01.ipynb) | 8.0 |
| [main_wave_01_shard_02](wave_01/main_wave_01_shard_02.ipynb) | 8.0 |
| [main_wave_01_shard_03](wave_01/main_wave_01_shard_03.ipynb) | 8.0 |
| [main_wave_01_shard_04](wave_01/main_wave_01_shard_04.ipynb) | 8.0 |
| [main_wave_01_shard_05](wave_01/main_wave_01_shard_05.ipynb) | 8.0 |
| [main_wave_01_shard_06](wave_01/main_wave_01_shard_06.ipynb) | 8.0 |
| [main_wave_01_shard_07](wave_01/main_wave_01_shard_07.ipynb) | 8.0 |
| [main_wave_01_shard_08](wave_01/main_wave_01_shard_08.ipynb) | 8.0 |
| [main_wave_01_shard_09](wave_01/main_wave_01_shard_09.ipynb) | 8.0 |
| [main_wave_01_shard_10](wave_01/main_wave_01_shard_10.ipynb) | 8.0 |
| [main_wave_01_shard_11](wave_01/main_wave_01_shard_11.ipynb) | 8.0 |
| [main_wave_01_shard_12](wave_01/main_wave_01_shard_12.ipynb) | 8.0 |
| [main_wave_01_shard_13](wave_01/main_wave_01_shard_13.ipynb) | 8.0 |
| [main_wave_01_shard_14](wave_01/main_wave_01_shard_14.ipynb) | 8.0 |
| [main_wave_01_shard_15](wave_01/main_wave_01_shard_15.ipynb) | 8.0 |
| [main_wave_01_shard_16](wave_01/main_wave_01_shard_16.ipynb) | 8.0 |
| [main_wave_01_shard_17](wave_01/main_wave_01_shard_17.ipynb) | 8.0 |
| [main_wave_01_shard_18](wave_01/main_wave_01_shard_18.ipynb) | 8.0 |
| [main_wave_01_shard_19](wave_01/main_wave_01_shard_19.ipynb) | 8.0 |
| [main_wave_01_shard_20](wave_01/main_wave_01_shard_20.ipynb) | 8.0 |
| [main_wave_01_shard_21](wave_01/main_wave_01_shard_21.ipynb) | 8.0 |
| [main_wave_01_shard_22](wave_01/main_wave_01_shard_22.ipynb) | 8.0 |
| [main_wave_01_shard_23](wave_01/main_wave_01_shard_23.ipynb) | 8.0 |
| [main_wave_01_shard_24](wave_01/main_wave_01_shard_24.ipynb) | 8.0 |
| [main_wave_01_shard_25](wave_01/main_wave_01_shard_25.ipynb) | 8.0 |
| [main_wave_01_shard_26](wave_01/main_wave_01_shard_26.ipynb) | 8.0 |
| [main_wave_01_shard_27](wave_01/main_wave_01_shard_27.ipynb) | 8.0 |
| [main_wave_01_shard_28](wave_01/main_wave_01_shard_28.ipynb) | 8.0 |
| [main_wave_01_shard_29](wave_01/main_wave_01_shard_29.ipynb) | 8.0 |
| [main_wave_01_shard_30](wave_01/main_wave_01_shard_30.ipynb) | 8.0 |
| [main_wave_01_shard_31](wave_01/main_wave_01_shard_31.ipynb) | 8.0 |
| [main_wave_01_shard_32](wave_01/main_wave_01_shard_32.ipynb) | 8.0 |
| [main_wave_01_shard_33](wave_01/main_wave_01_shard_33.ipynb) | 8.0 |
| [main_wave_01_shard_34](wave_01/main_wave_01_shard_34.ipynb) | 8.0 |
| [main_wave_01_shard_35](wave_01/main_wave_01_shard_35.ipynb) | 8.0 |
| [main_wave_01_shard_36](wave_01/main_wave_01_shard_36.ipynb) | 8.0 |
| [main_wave_01_shard_37](wave_01/main_wave_01_shard_37.ipynb) | 8.0 |
| [main_wave_01_shard_38](wave_01/main_wave_01_shard_38.ipynb) | 8.0 |
| [main_wave_01_shard_39](wave_01/main_wave_01_shard_39.ipynb) | 8.0 |
| [main_wave_01_shard_40](wave_01/main_wave_01_shard_40.ipynb) | 8.0 |
| [main_wave_01_shard_41](wave_01/main_wave_01_shard_41.ipynb) | 8.0 |
| [main_wave_01_shard_42](wave_01/main_wave_01_shard_42.ipynb) | 8.0 |
| [main_wave_01_shard_43](wave_01/main_wave_01_shard_43.ipynb) | 8.0 |
| [main_wave_01_shard_44](wave_01/main_wave_01_shard_44.ipynb) | 8.0 |
| [main_wave_01_shard_45](wave_01/main_wave_01_shard_45.ipynb) | 8.0 |
| [main_wave_01_shard_46](wave_01/main_wave_01_shard_46.ipynb) | 8.0 |
| [main_wave_01_shard_47](wave_01/main_wave_01_shard_47.ipynb) | 8.0 |
| [main_wave_02_shard_00](wave_02/main_wave_02_shard_00.ipynb) | 8.0 |
| [main_wave_02_shard_01](wave_02/main_wave_02_shard_01.ipynb) | 8.0 |
| [main_wave_02_shard_02](wave_02/main_wave_02_shard_02.ipynb) | 8.0 |
| [main_wave_02_shard_03](wave_02/main_wave_02_shard_03.ipynb) | 8.0 |
| [main_wave_02_shard_04](wave_02/main_wave_02_shard_04.ipynb) | 8.0 |
| [main_wave_02_shard_05](wave_02/main_wave_02_shard_05.ipynb) | 8.0 |
| [main_wave_02_shard_06](wave_02/main_wave_02_shard_06.ipynb) | 8.0 |
| [main_wave_02_shard_07](wave_02/main_wave_02_shard_07.ipynb) | 8.0 |
| [main_wave_02_shard_08](wave_02/main_wave_02_shard_08.ipynb) | 8.0 |
| [main_wave_02_shard_09](wave_02/main_wave_02_shard_09.ipynb) | 8.0 |
| [main_wave_02_shard_10](wave_02/main_wave_02_shard_10.ipynb) | 8.0 |
| [main_wave_02_shard_11](wave_02/main_wave_02_shard_11.ipynb) | 8.0 |
