# Final paper results

Place the downloaded `didbcf_paper_*.zip` files here. They contain evaluation summaries and compact provenance, not posterior draw arrays. This folder is intentionally tracked and is not covered by the local intermediate archive ignore rule.

From `Paper_Rerun_20261011/`, collect with:

```text
python helpers/collect.py Results/didbcf_paper_*.zip --out Results/collected --require-main
```

The collector adds the 264 verified earlier fits automatically. Missing fits, identical duplicate ZIPs and conflicting results are checked explicitly. `--require-main` requires 5,100 complete main fits, while still writing missing-job status and partial tables for recovery.
