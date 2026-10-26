# Repaired non-main results, 2026-10-26

This directory overlays the completed non-main metrics on the immutable
2026-10-14 repaired-main aggregate. It contains no posterior arrays and
does not run sampling.

Completed jobs: 9227 / 9227;
all phase counts match the frozen manifest.

The new pretrend diagnostic rows use the repaired raw `pretrend` fit.
Historical TWFE and other benchmark rows remain labelled historical.
The reference-fold benchmark comes from the archived completion metrics.
The `PT_violation_het20`, degree-3 historical comparator has n=150; paired
comparisons should use the matching replication intersection.

Prior sensitivity rates 0.1 and 10 are paired to the repaired-main rate-1
donor by seed/index and identical truth. No byte-level panel hashes are
available in these summary tables, so this pairing does not establish
binary dataset identity.
