# Final repaired pretrend, sensitivity and application campaign

Run the loose notebooks described in [the Colab upload guide](Colab/UPLOAD_GUIDE.md). There are **132 experiment notebooks**, in waves of **48, 48 and 36**, plus one collector. Each targets about eight hours with two CPU workers. The 9.5-hour session deadline includes setup and export reserve. Runtime varies across Colab machines; a saved metrics ZIP resumes verified complete fits from the same assignment.

You can also use [the direct Colab launch links](Colab/OPEN_IN_COLAB.md) to open each published notebook without uploading it manually.

The Colab notebooks cover 3,999 remaining pretrend diagnostic fits, 4,000 pretrend-impact fits and 1,198 simulation prior-sensitivity fits. They are mixed into balanced assignments rather than isolating a study phase into an unequal runtime. Application and application prior-sensitivity runs are assigned locally: 22 new fits. The package includes eight completed, checksum-verified earlier fits, including five application fits, so no already completed anchor is repeated. Completing all assignments yields all 9,227 original non-main jobs.

All jobs retain their original DGPs, data, priors, specifications and seeds. The budget is 50 GFR sweeps, no extra burn-in, 500 retained draws per chain, thinning five and three chains. The main standardized noise variance prior is IG(2,1); the prespecified sensitivity rates are 0.1 and 10. The exact repaired engine/source fingerprint is inherited from the completed main rerun. The earlier budget is used under the user-authorized paper-performance policy, with diagnostics retained and no diagnostic or performance filtering. This policy does not certify finite-MCMC convergence or nominal frequentist interval coverage.

Each experiment notebook downloads **one compact metrics ZIP** with per-fit posterior summaries, paper evaluation metrics, scalar diagnostics and JSON provenance. All pretrend decision rows and application group, county-year and contrast summaries are retained. Posterior effect arrays, sweep traces, native binaries and copied source archives stay on the runtime; successful raw sampling scratch is removed after exact pooled summaries are published. The paper collector does not require those arrays.

Use a CPU Colab runtime with Python 3.12 to 3.14. Each notebook clones immutable helper commit `30b165b27610dc4b2c406a9d64281d77e660257a`, installs the pinned dependencies, and builds an isolated repaired engine with one compilation worker. It does not install a global stochtree package. Sampling uses at most two worker processes with one numeric/OpenMP thread each and reduces concurrency when RAM or CPU quota requires it.

Enable `USE_DRIVE` in each notebook to keep a complete-fit metrics checkpoint every 20 minutes. To resume a downloaded ZIP, upload it in the Colab Files pane and set `RESUME_ZIP` before the preparation cell. An interrupted fit restarts its unfinished chain or atomic pretrend job from the original seed; its results are never appended to an unrelated fit. Every failure or deferred fit remains explicitly listed for recovery.

Place returned `didbcf_nonmain_*.zip` files in [Results](Results/README.md). Collect from the package root:

```sh
python helpers/collect.py Results/didbcf_nonmain_*.zip --out Results/collected --require-complete
```

For the separately completed local application branch, use `--require-application` instead. The collector automatically includes the eight registered reused fits, preserves all seed-specific application outputs, separates study phases and prior rates, and reports every missing job. Repeated identical checkpoints are counted once; conflicting outputs are rejected. Hexadecimal job IDs are explicitly read as strings.

The local application runs use six workers on the currently available eight-CPU, 23.5 GiB environment, with a 2.5 GiB admission reservation per worker plus a 3 GiB memory floor. Each sampling subprocess is single-threaded and capped at 6 GiB virtual memory. Native and fixed eligible-feature controls passed before execution. Primary application reporting is prespecified as replication seed bundle 0; seed bundles 1 and 2 assess computational stability. Their interval endpoints are retained separately and are not averaged into an invented pooled posterior.

The immutable manifest, package seal, assignments and measurement-based scheduling are in `manifest.json`, `package.json`, `assignments/`, `preparation.json` and `scheduling.csv`. The source snapshot and reuse evidence make the helper checkout self-contained. Validation records cover the exact partition, all eight reuse records, notebook formats and code, mocked Python setup, collector fixtures, and isolated checkout collection. This package does not repeat the completed 5,100-fit main rerun. Final experiment outputs remain visible to Git.
