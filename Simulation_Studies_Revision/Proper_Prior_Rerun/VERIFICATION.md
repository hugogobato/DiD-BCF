# Preparation verification, 2026-09-15

Final manifest SHA-256 (canonical manifest body):
`105ee8510f99817b132d69869167ec5a724596ed37d8094f7ae430c958ad1ccc`.

The combined sampler, proper-prior, existing pretrend-pipeline and campaign
regression suite passed: 39 tests, 17.01 seconds. After the final phase-balanced
sharding change, all five campaign tests were repeated and passed (5.87 seconds).
The old residual test's correlation threshold was replaced with tests of the
known conditional draw moments: correlation has no universal positive lower
bound when the conditional residual variance changes little across iterations.
The residual mean check and deliberately incorrect-residual negative control
remain in place.

Five final-manifest smoke jobs completed and were then rerun in resume mode;
all five were checksum-verified and skipped, without refitting. The tested paths
were the raw simulation estimator, simulation diagnostic, application without
random effects, application with unit effects, and application diagnostic.
Simulation smoke panels had 50 units and 400 rows. Application smoke panels
used the actual 500-county, 2,500-row data. Each smoke fit retained four draws
per chain in two chains, following three GFR iterations. These are code checks,
not convergence checks or usable posterior inference.

All 48 generated notebooks parse, and their code cells compile. An embedded
bundle was extracted into an independent temporary directory; the campaign
verified its source/data hashes and dependency versions successfully with
`run --max-jobs 0`. A fresh Colab dependency installation and execution on
Google's machines have not been tested here. The zip of notebooks is about
21 MB. The six campaign phases total 14,327 prepared fits, with each large
phase balanced across all 48 shards.

No production job was run. `Reruns/proper_ig_v1/` contains only explicitly
flagged smoke outputs. Initial smoke artifacts and their earlier manifest are
preserved separately under `Reruns/proper_ig_initial_smoke/` and
`manifest_initial_smoke.json`; they must not be mixed with the final campaign.
Historical experiment results were not rewritten or deleted.

Posterior propriety is established by the analytic bound in
`JBES_Submission/audit/PROPER_PRIOR_REPAIR.md`, with numerical checks for the
bound and a saturated design. Independent mathematical review remains pending
because the requested subagents exhausted their model allowance earlier in
the submission task. This verification is not an endorsement of unfinished
asymptotic claims or a submission-readiness certificate.
