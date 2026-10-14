# Repaired-source sampling-budget pilot (frozen before results)

This pilot diagnoses computation. Its cases are not a Monte Carlo sample for
performance tables. All historical results remain separate. Source repair,
proper IG(2,1) noise prior, forest architecture, raw total-effect target,
standardization, DGPs and original job seeds are fixed in the new manifest.

The inherited twelve-case selection keeps the three named anchors and explicit
replication-zero strata for large N, serial correlation, observable selection,
the null, staggered adoption, degree-three effects, a unit-effects pretrend
diagnostic and both raw application specifications. Five prespecified additions
cover the application pretrend diagnostic, simulation prior rates 0.1 and 10
(N=50, degree one, replication zero), and unit-effects application rates 0.1
and 10 (replication zero). Historical partial chains are never pooled with this
repaired pilot. The emitted selection CSV identifies all seventeen cases.

B1 uses 50 GFR initialization sweeps, 2,500 ordinary burn-in sweeps, 1,000
retained draws per chain and thinning five. B2 increases ordinary burn-in to
5,000 while keeping retained length fixed. B3 instead increases retained length
to 2,000 while keeping ordinary burn-in at 2,500. Each uses four chains. GFR is
initialization; thinning controls storage. Neither establishes stationarity.
Structured chain seeds remain 1000 times replication plus chain index. The
atomic pretrend wrapper retains its original seed mapping and archives all
four chain segments. No truth or coverage result selects the sampling budget.

Every runtime builds the pinned repaired source in an isolated directory, checks
the exact original/repaired file hashes, imports that extension, and runs native
controls. Fixed eligible features have equal positive weights; exactly excluded
features have zero weights. The explicit support equivalence and additional
native fixtures cover the diagnostic wrapper's exclusions. Numeric inputs must
have finite representable spans; categorical inputs, adaptive weights and
feature subsampling are excluded. These controls concern supported transition
behavior, not full-forest mixing or causal consistency.

Each structured chain is a checkpoint unit. An interrupted chain starts again
from its recorded seed, with a new attempt directory. Diagnostic BCFModel jobs
are atomic across their four chains. There is no within-chain state continuation.
The controller uses one fresh process and one numeric thread at a time, refuses
units that do not fit the estimated remaining window, and terminates a running
unit at the hard sampling deadline. Completed chains survive that interruption.
Periodic checkpoint ZIPs can be copied to a user-enabled persistent Drive path.
A disconnected runtime cannot execute its final download.

Prespecified required screens are rank/folded split R-hat at most 1.01, bulk and
tail ESS at least 400, mean MCSE at most 2% of posterior SD, and both 90% and 95%
credible endpoint MCSEs at most 1% of the corresponding interval width. Required
states include ATT, every reported cohort/time and event-time average, nuisance
leaf scales, noise variance, applicable random-effect variance, and every scalar
pretrend/subgroup/contrast target. The effect variance is fixed. Forest size,
gauge components and fitted block means are also recorded for mechanism review;
their discrete or fixed traces are not added to the precision screens.

Mean stability flags compare B1/B2, B1/B3 and B2/B3 using the larger of twice
the combined MCSE and 5% of reference posterior SD. Endpoint flags use twice
combined endpoint MCSE or 2% of the reference interval width. Every required
scalar is compared. Passing a finite diagnostic screen does not prove mixing,
frequentist coverage or asymptotic consistency.

Release production only after a complete, integrity-verified pilot, all required
screens for the selected budget, resolution of its prespecified stability flags,
agreement of recorded traces with retained draws, and a written trace review.
The acceptance tool hashes the evidence and records the exact interpreter,
dependencies, source and engine specification. Failed cases remain in reports.
If no budget passes, record that result and prepare a versioned extended pilot;
do not drop cases or relax criteria after seeing favorable estimates.

The production universe has all 14,327 original statistical jobs: 5,100 repaired
replacement main jobs and the 9,227 previously absent jobs. Its final budget and
notebooks are generated from accepted evidence. Runtime and storage figures
before the pilot returns are proxies from historical timings, not repaired-run
measurements. Every job's diagnostics and failures remain visible during
production; a pilot budget does not guarantee that every replication will pass.
