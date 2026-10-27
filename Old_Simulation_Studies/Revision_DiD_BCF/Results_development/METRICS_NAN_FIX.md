# Non-finite estimates corrupted the metrics table two ways at once

Found 2026-08-03 while verifying the `D_ramp_nt05` R benchmark run. It changes
numbers that are already in `revision_simulation_report.tex`, so it needs to be
carried into the revision rather than quietly fixed.

## The defect

Callaway--Sant'Anna (`did_dr`) sometimes returns a row with a NaN estimate: the
replication ran, the estimator produced output, but a given `(g,t)` cell could
not be estimated. `did_bcf_revision.metrics._metrics_for_group` handled those
rows inconsistently, because two families of statistic were computed two
different ways from the same frame:

| column | computed as | effect of one NaN row |
|---|---|---|
| `bias`, `rmse`, `mae`, `emp_sd` | `np.mean(...)` | NaN propagates: the **whole cell goes blank** |
| `cover90`, `cover95` | `(q025 <= true) & (true <= q975)` | `NaN <= true` is `False`, so the failure is **scored as a coverage miss** |
| `n_reps` | `len(g)` | counts the failed replication as if it had succeeded |

So a single bad replication simultaneously blanked the bias and pushed coverage
*down*. The two errors point in different directions, which is why neither was
obvious from reading the table.

## Blast radius

141 rows of `Results/aggregated/metrics_all.csv`, all of them `did_dr` (no other
estimator in the suite emits a non-finite estimate):

| setting | rows affected |
|---|---|
| `B1_selection_obs` | 8 |
| `D_contamination` | 14 |
| `D_staggered` | 14 |
| `D_ramp_nt40` / `nt30` / `nt25` / `nt20` / `nt10` / `nt05` | 105 |

`bias` was NaN in **141 of 141**. Mean `cover95` over the affected rows moves
**0.937 -> 0.965**. The correction runs *against* the paper's benchmark
comparison: Callaway--Sant'Anna's coverage was being reported lower than it
actually is. Example, `B1_selection_obs`, `linearity_degree=2`, GATT `g=4_t=4`:
`cover95` 0.81 -> 0.911.

Everything else is untouched: 2130 of 2130 metric rows reconcile with the
pre-fix table except those 141.

## The fix

Two parts, because there are two kinds of failure.

**Soft failures (a row exists, the estimate is NaN).** `_metrics_for_group` now
drops non-finite `post_mean`/`true` once, up front, and computes every statistic
on the surviving replications, so all columns describe the same sample. It
records `n_reps_attempted` and `retention = n_reps / n_reps_attempted`.

**Hard failures (no row at all).** `compute_metrics` cannot see a replication
that emitted nothing, so its `retention` still overstates. `aggregate_all.py`
recovers the intended count per design as the largest attempt count any method
managed there (a method that ran everywhere defines the target) and rescales.
At `D_ramp_nt05` this is the difference between 0.99 and the true **0.92**: 15
of `did_dr`'s 200 replications returned no rows whatsoever, and 1 more returned
all-NaN rows.

## Why the retention column has to stay visible

A row with `retention < 1` describes a **selected subsample** -- the replications
in which the estimator returned an estimate -- and is not comparable with a row
from an estimator that ran everywhere. This matters directly for Reviewer
3.1.2's question, because Callaway--Sant'Anna's estimability degrades along the
already-treated-weight axis:

| `w_already_treated` | retention | mean \|bias\| | `cover95` |
|---|---|---|---|
| 0.077 | 1.00 | 0.014 | 0.993 |
| 0.116 | 1.00 | 0.012 | 0.990 |
| 0.196 | 1.00 | 0.017 | 0.963 |
| 0.250 | **0.92** | 0.033 | 0.902 |

(at `linearity_degree` 2 and 3 the last point falls to 0.785)

Read without retention, the bias column looks flat and the estimator looks
robust. Part of that flatness is survivorship: the hardest replications are the
ones that dropped out. `make_ramp_analysis.py` therefore rings such points in
the figure and daggers them in the LaTeX table with a footnote.

## Reproducing

```bash
python Results/analysis/aggregate_all.py
python Results/analysis/make_ramp_analysis.py    # prints the selected-subsample list
```
