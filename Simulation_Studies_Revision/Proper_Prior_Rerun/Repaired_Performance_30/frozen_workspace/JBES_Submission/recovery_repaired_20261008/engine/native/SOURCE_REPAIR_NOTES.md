# Proposed native source repair

This is an unapplied source proposal for the pinned stochtree 0.4.4 source at
commit `85becbd`. Apply `stochtree-0.4.4-native-mcmc-repair.patch` with
`patch -p1` at the clean source root. Its SHA-256 is
`b18a0a97e3302496b439bf8825ec0b5eeaa5bdba1a67287b5b92d6b84566c118`.
The patch changes only `include/stochtree/tree_sampler.h` and
`src/cutpoint_candidates.cpp`. The frozen source snapshot and installed
package were not modified by this work.

The source audit used `tree_sampler.h` lines 48--65 for `VarSplitRange`,
78--144 for the two node-constancy helpers, 863--968 for grow, 970--1046 for
prune, and 1048--1097 for the move dispatcher. `tree.h:835` routes a numeric
value left when `value <= threshold`. The GFR scan is
`cutpoint_candidates.cpp:260--317`, with the grid boundary check at the
original line 306 and the distinct-value check at 307. The prior convention
is the capped, feasible-tree depth law stated in `implemented_prior.tex`
around lines 222--230; the required MH ratio and transition scope are also
recorded in `NATIVE_KERNEL_SCOPE_DECISION.md`.

The old extrema initialization used the smallest positive `double` as a
maximum, and an `else if` prevented a value from updating both extrema.
This fails for negative-valued features. The proposal uses the lowest finite
`double` and independent minimum/maximum comparisons. The old constancy
helpers were also inadequate proxies for legal grow availability: they did
not account for min-leaf thresholds, max depth, or whether a positive-width
set of valid numeric cutpoints exists. The replacement availability check
examines every eligible leaf and positive-weight feature. For `m >= 1` and
`n >= 2m`, with sorted node values `x_(1) <= ... <= x_(n)` and minimum leaf
size `m`, a positive-measure set of valid thresholds exists exactly when the
node is below the depth cap and `x_(m) < x_(n-m+1)`. This includes `n = 2m`
when the two boundary values differ, consistent with the inclusive `<=`
routing and child checks using counts `>= m`. Max depth removes grow
proposals only. The raw depth split probability
`p_g(d) = alpha (1+d)^(-beta)` and terminal factor `1-p_g(d)` are retained
at the cap, as required by the intended prior.

For a grow from `T` to `T'`, let `L(T)` be the current leaf count and `P(T')`
the proposed leaf-parent count. The proposal terms are
\[
q(T'\mid T)=p_G(T)L(T)^{-1}q_{\rm rule},\qquad
q(T\mid T')=p_P(T')P(T')^{-1}.
\]
The patch computes `p_G` and `p_P` using whole-tree availability in the
corresponding state, and updates the leaf-parent count by `+1` except when
the selected leaf's parent was already a leaf-parent, in which case it is
unchanged. For prune, the reverse grow probability is evaluated in the
proposed pruned tree and uses its leaf count `L(T') = L(T)-1`; the forward
prune term uses the current state's leaf-parent count. A prune with no
positive-probability reverse grow is rejected. When neither move exists,
the dispatcher leaves the state unchanged rather than raising an error.

The grow target topology factor is
`p_g(d) (1-p_g(d+1))^2 / (1-p_g(d))`, and the prune factor is its inverse.
The proposal also includes the feature and raw continuous-threshold density.
In the configured all-numeric design with equal positive feature weights,
the target split-rule density and proposal rule density match and cancel.
The implementation writes this cancellation explicitly. It is scoped to
that configured prior/proposal pairing; arbitrary nonuniform weights or a
different split-rule prior require a separately matched target density and
are outside this proposal's audited scope.

The GFR change replaces the old grid-crossing condition at
`cutpoint_candidates.cpp:306`, which can remain false after a tied-value
stride has grown past the target bin boundary, with a `stride_length >=
step_size` gate. The existing distinct-value check remains in force, so a
cutpoint is emitted only between distinct values. Tied observations still
route together under the numeric `<=` rule. This is a narrow equality/tie
gate correction, not a proof that the GFR grid is otherwise an exact or
uniform split-rule sampler.

The source patch is intentionally scoped to the production numeric,
equal-feature-weight configuration. `MCMCSampleTreeOneIter` does not receive
feature-type metadata, and its categorical-tree check cannot identify a
categorical-typed input while the tree is still at its root. The integration
must therefore reject categorical feature types and unsupported weight
configurations before invoking this branch. The patch also does not certify
categorical proposals, alternative split priors, nonfinite or extreme
floating-point inputs, or all GFR/grid behaviors.

This artifact is a source-level proposal, not an applied repair or sampler
certification. It does not change priors, likelihoods, residual bookkeeping,
historical outputs, or root initialization. The GFR candidate grid is changed
as described above. A clean-source build, closed
tiny-state transition checks across the supported branches, independent
review, integration-side scope gate, and a fresh audit of the built binary
remain necessary before any claim that the repaired sampler targets the
intended posterior. No installation, rebuild, or fit was run by this source
audit.

Coordinator validation addendum, 2026-09-30: isolated original and repaired
builds and bounded native controls have since been completed; the author's
original source-review scope above is preserved. Their results and disposition
are recorded in `../local_results/` and `../reviews/ADJUDICATION.md`. The supported
application domain additionally requires finite representable numeric spans.
The native patch itself does not reject every extreme finite-input overflow;
the paired runner supplies the pre-sampling range guard. No general forest or
asymptotic certificate follows from these checks.
