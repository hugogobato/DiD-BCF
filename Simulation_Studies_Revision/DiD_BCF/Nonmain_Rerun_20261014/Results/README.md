# Returned non-main metrics

Place the 132 Colab experiment ZIPs here. The local application branch writes `didbcf_nonmain_local_application.zip` here and keeps its complete-fit summaries in `local_application/`.

Run the collector from the package root with `--require-complete` after all Colab returns. It automatically includes eight registered reused fits. While the simulation waves are pending, `--require-application` validates completion of all nine base-application and 18 application-sensitivity jobs.

These are final experimental results and are not ignored. Keep the original returned ZIPs alongside collected metrics for validation; posterior arrays are not needed.

The local branch is complete: `local_application/` has 22 new complete-fit records and `didbcf_nonmain_local_application.zip` is their verified 0.63 MiB return. `application_collected/` combines those records with the eight registered reused fits. All 27 application jobs are present (9 base-prior and 18 sensitivity jobs); the remaining 9,197 simulation jobs are still missing until the Colab waves return. See [the application report](../analysis/application_report.md). These counts do not imply that the entire non-main campaign is complete.
