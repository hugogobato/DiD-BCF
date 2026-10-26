"""Bounded independent controls for the stochtree native repair.

The deterministic controls below enumerate the intended posterior and the
frozen v0.4.4 transition equations independently of stochtree.  Optional
``--native`` probes call the installed Python API and record behavior from the
loaded binary.  Source-equation checks never certify that binary.

Run with::

    rtk python3 controls.py
    rtk python3 controls.py --native --variant repaired

The module uses only the Python standard library for its exact controls.
NumPy and stochtree are imported only by optional native probes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import platform
import sys
from typing import Iterable, Optional


# These limits are fixed before any native probe is run.  The exact matrix
# controls use 1e-12 for floating-point row-sum and stationarity checks.
EXACT_TOL = 1e-12
STOCHTREE_NUMERIC_EPSILON = 1e-15
NATIVE_MCMC_BURN = 100
NATIVE_MCMC_DRAWS = 20_000
NATIVE_MCMC_ABS_TOL = 0.015
NATIVE_GFR_REPEATS = 20
NATIVE_GFR_MIN_BOUNDARY_SPLITS = 19
NATIVE_EXTREMA_REPEATS = 64
NATIVE_EXTREMA_MIN_SPLITS = NATIVE_EXTREMA_REPEATS


@dataclass(frozen=True)
class Tree:
    """A one-feature ordered tree; ``cut`` is the first row on the right."""

    cut: int
    left: Optional["Tree"] = None
    right: Optional["Tree"] = None


TreeState = Optional[Tree]


def tree_key(tree: TreeState) -> str:
    if tree is None:
        return "L"
    return f"({tree.cut}:{tree_key(tree.left)}:{tree_key(tree.right)})"


def enumerate_trees(n: int, min_leaf: int, max_depth: int) -> list[TreeState]:
    """Enumerate every legal binary partition tree on ordered rows 0,...,n-1."""

    memo: dict[tuple[int, int, int], tuple[TreeState, ...]] = {}

    def build(lo: int, hi: int, depth: int) -> tuple[TreeState, ...]:
        key = (lo, hi, depth)
        if key in memo:
            return memo[key]
        result: list[TreeState] = [None]
        if depth < max_depth:
            for cut in range(lo + min_leaf, hi - min_leaf + 1):
                for left in build(lo, cut, depth + 1):
                    for right in build(cut, hi, depth + 1):
                        result.append(Tree(cut, left, right))
        memo[key] = tuple(result)
        return memo[key]

    return list(build(0, n, 0))


def replace_subtree(tree: TreeState, path: tuple[int, ...], replacement: TreeState) -> TreeState:
    if not path:
        return replacement
    if tree is None:
        raise ValueError("path descends through a leaf")
    head, tail = path[0], path[1:]
    if head == 0:
        return Tree(tree.cut, replace_subtree(tree.left, tail, replacement), tree.right)
    return Tree(tree.cut, tree.left, replace_subtree(tree.right, tail, replacement))


def leaves(tree: TreeState, n: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []

    def visit(
        node: TreeState,
        lo: int,
        hi: int,
        depth: int,
        path: tuple[int, ...],
        parent: TreeState,
    ) -> None:
        if node is None:
            out.append({
                "path": path,
                "lo": lo,
                "hi": hi,
                "depth": depth,
                "parent_was_leaf_parent": (
                    parent is not None and parent.left is None and parent.right is None
                ),
            })
            return
        visit(node.left, lo, node.cut, depth + 1, path + (0,), node)
        visit(node.right, node.cut, hi, depth + 1, path + (1,), node)

    visit(tree, 0, n, 0, (), None)
    return out


def internal_nodes(tree: TreeState, n: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []

    def visit(node: TreeState, lo: int, hi: int, depth: int, path: tuple[int, ...]) -> None:
        if node is None:
            return
        out.append({"node": node, "path": path, "lo": lo, "hi": hi, "depth": depth})
        visit(node.left, lo, node.cut, depth + 1, path + (0,))
        visit(node.right, node.cut, hi, depth + 1, path + (1,))

    visit(tree, 0, n, 0, ())
    return out


def leaf_parent_count(tree: TreeState, n: int) -> int:
    return sum(
        1
        for item in internal_nodes(tree, n)
        if item["node"].left is None and item["node"].right is None  # type: ignore[union-attr]
    )


def legal_cuts(
    x: list[float], lo: int, hi: int, min_leaf: int, depth: int, max_depth: int
) -> list[int]:
    if depth >= max_depth:
        return []
    return [
        cut
        for cut in range(lo + min_leaf, hi - min_leaf + 1)
        if x[cut - 1] < x[cut]
    ]


def grow_edges(
    tree: TreeState, x: list[float], min_leaf: int, max_depth: int
) -> list[tuple[TreeState, dict[str, object]]]:
    out: list[tuple[TreeState, dict[str, object]]] = []
    for leaf in leaves(tree, len(x)):
        lo, hi = int(leaf["lo"]), int(leaf["hi"])
        depth = int(leaf["depth"])
        for cut in legal_cuts(x, lo, hi, min_leaf, depth, max_depth):
            new_tree = replace_subtree(tree, leaf["path"], Tree(cut))  # type: ignore[arg-type]
            out.append((new_tree, {
                "path": leaf["path"],
                "lo": lo,
                "hi": hi,
                "depth": depth,
                "cut": cut,
                "gap": x[cut] - x[cut - 1],
                "range": x[hi - 1] - x[lo],
                "parent_was_leaf_parent": bool(leaf["parent_was_leaf_parent"]),
            }))
    return out


def prune_edges(tree: TreeState, n: int) -> list[tuple[TreeState, dict[str, object]]]:
    out: list[tuple[TreeState, dict[str, object]]] = []
    for item in internal_nodes(tree, n):
        node = item["node"]
        if node.left is None and node.right is None:  # type: ignore[union-attr]
            out.append((
                replace_subtree(tree, item["path"], None),
                {**item, "cut": node.cut},
            ))  # type: ignore[arg-type]
    return out


def split_probability(alpha: float, beta: float, depth: int) -> float:
    return alpha / ((1 + depth) ** beta)


def leaf_log_marginal(y: list[float], lo: int, hi: int, leaf_var: float, noise_var: float) -> float:
    values = y[lo:hi]
    n = len(values)
    sum_y = math.fsum(values)
    sum_y2 = math.fsum(value * value for value in values)
    logdet = (n - 1) * math.log(noise_var) + math.log(noise_var + n * leaf_var)
    quadratic = (
        sum_y2 / noise_var
        - leaf_var * sum_y * sum_y / (noise_var * (noise_var + n * leaf_var))
    )
    return -0.5 * (n * math.log(2 * math.pi) + logdet + quadratic)


def tree_log_marginal(
    tree: TreeState, n: int, y: list[float], leaf_var: float, noise_var: float
) -> float:
    return math.fsum(
        leaf_log_marginal(y, int(leaf["lo"]), int(leaf["hi"]), leaf_var, noise_var)
        for leaf in leaves(tree, n)
    )


def tree_log_prior(
    tree: TreeState,
    x: list[float],
    min_leaf: int,
    max_depth: int,
    alpha: float,
    beta: float,
) -> float:
    """Raw depth-law prior mass on induced partitions, integrated over cutpoints.

    The admissible capped/minleaf tree set is normalized globally. Every
    terminal node retains its raw ``1-p_g(depth)`` factor, including leaves at
    the depth cap; the cap only removes grow proposals from that node.
    """

    n = len(x)

    def visit(node: TreeState, lo: int, hi: int, depth: int) -> float:
        p = split_probability(alpha, beta, depth)
        if node is None:
            return math.log1p(-p)
        cut = node.cut
        if depth >= max_depth:
            return -math.inf
        gap = x[cut] - x[cut - 1]
        node_range = x[hi - 1] - x[lo]
        if gap <= 0 or node_range <= 0:
            return -math.inf
        value = math.log(p) + math.log(gap / node_range)
        return value + visit(node.left, lo, cut, depth + 1) + visit(node.right, cut, hi, depth + 1)

    return visit(tree, 0, n, 0)


def normalized_weights(log_weights: list[float]) -> list[float]:
    top = max(log_weights)
    raw = [math.exp(value - top) for value in log_weights]
    total = math.fsum(raw)
    return [value / total for value in raw]


def move_probabilities(
    tree: TreeState,
    x: list[float],
    min_leaf: int,
    max_depth: int,
    availability: str,
) -> tuple[float, float, bool, bool]:
    if availability == "exact":
        grow = bool(grow_edges(tree, x, min_leaf, max_depth))
    elif availability == "frozen":
        grow = any(int(leaf["hi"]) - int(leaf["lo"]) > 2 * min_leaf for leaf in leaves(tree, len(x)))
    else:
        raise ValueError(availability)
    prune = bool(prune_edges(tree, len(x)))
    if grow and prune:
        return 0.5, 0.5, grow, prune
    if grow:
        return 1.0, 0.0, grow, prune
    if prune:
        return 0.0, 1.0, grow, prune
    return 0.0, 0.0, grow, prune


def frozen_node_nonconstant(values: list[float]) -> bool:
    """Reproduce the frozen helper's sentinel and ``if/else if`` reduction."""

    current_max = sys.float_info.min
    current_min = sys.float_info.max
    for value in values:
        if current_max < value:
            current_max = value
        elif current_min > value:
            current_min = value
    return current_max > current_min


def source_prune_probability_after_grow(
    x: list[float], edge: dict[str, object], min_leaf: int
) -> float:
    cut = int(edge["cut"])
    lo, hi = int(edge["lo"]), int(edge["hi"])
    left_n, right_n = cut - lo, hi - cut
    # The frozen helper scans the tracker node rows in their retained input
    # order. This fixture's input order is increasing in x.
    left_nonconstant = frozen_node_nonconstant(x[lo:cut])
    right_nonconstant = frozen_node_nonconstant(x[cut:hi])
    if left_nonconstant and right_nonconstant and (left_n >= 2 * min_leaf or right_n >= 2 * min_leaf):
        return 0.5
    return 1.0


def edge_log_ratio(
    parent: TreeState,
    child: TreeState,
    edge: dict[str, object],
    *,
    x: list[float],
    y: list[float],
    min_leaf: int,
    max_depth: int,
    alpha: float,
    beta: float,
    leaf_var: float,
    noise_var: float,
    variant: str,
) -> tuple[float, float]:
    """Return grow and prune log acceptance ratios for one adjacent edge."""

    depth = int(edge["depth"])
    p_parent = split_probability(alpha, beta, depth)
    p_child = split_probability(alpha, beta, depth + 1)
    if variant not in {"original", "repaired"}:
        raise ValueError(variant)
    # The intended target retains the raw depth-law terminal factors at the
    # cap for both variants; max_depth restricts proposal support only.
    child_no_split_log = 2.0 * math.log1p(-p_child)
    structural = math.log(p_parent) - math.log1p(-p_parent) + child_no_split_log
    log_bf = (
        tree_log_marginal(child, len(x), y, leaf_var, noise_var)
        - tree_log_marginal(parent, len(x), y, leaf_var, noise_var)
    )

    parent_grow, _, _, _ = move_probabilities(parent, x, min_leaf, max_depth,
                                               "exact" if variant == "repaired" else "frozen")
    child_grow, child_prune, _, _ = move_probabilities(child, x, min_leaf, max_depth,
                                                       "exact" if variant == "repaired" else "frozen")
    parent_leaves = len(leaves(parent, len(x)))
    child_leaves = len(leaves(child, len(x)))
    parent_leaf_parents = leaf_parent_count(parent, len(x))
    child_leaf_parents = leaf_parent_count(child, len(x))

    if variant == "repaired":
        # Exact q(T'|T)=pG(T)/L(T)*gap/range and q(T|T')=pP(T')/P(T').
        log_grow = -math.inf if parent_grow <= 0 else (
            structural + log_bf + math.log(child_prune) - math.log(child_leaf_parents)
            - math.log(parent_grow) + math.log(parent_leaves)
        )
        log_prune = -log_grow
        return log_grow, log_prune

    # Frozen grow equation: heuristic reverse prune probability and P(T)+1.
    reverse_prune_heuristic = source_prune_probability_after_grow(x, edge, min_leaf)
    guessed_child_leaf_parents = parent_leaf_parents + 1
    log_grow = -math.inf if parent_grow <= 0 else (
        structural + log_bf + math.log(reverse_prune_heuristic)
        - math.log(guessed_child_leaf_parents) - math.log(parent_grow)
        + math.log(parent_leaves)
    )

    # Frozen prune equation uses the reciprocal tree odds but reverses the
    # move-probability terms and uses pre-prune NumNodes()>1 as pG_new=1/2.
    node = next(
        item["node"]
        for item in internal_nodes(child, len(x))
        if item["path"] == edge["path"]
    )
    lo, hi, cut = int(edge["lo"]), int(edge["hi"]), int(edge["cut"])
    left_nonconstant = frozen_node_nonconstant(x[lo:cut])
    right_nonconstant = frozen_node_nonconstant(x[cut:hi])
    prune_old_heuristic = 0.5 if left_nonconstant and right_nonconstant else 1.0
    log_prune = (
        -structural - log_bf + math.log(prune_old_heuristic) - math.log(0.5)
        + math.log(child_leaf_parents) - math.log(parent_leaves)
    )
    return log_grow, log_prune


def transition_matrix(
    trees: list[TreeState],
    *,
    x: list[float],
    y: list[float],
    min_leaf: int,
    max_depth: int,
    alpha: float,
    beta: float,
    leaf_var: float,
    noise_var: float,
    variant: str,
) -> tuple[list[list[float]], list[float], list[str]]:
    keys = [tree_key(tree) for tree in trees]
    index = {key: i for i, key in enumerate(keys)}
    log_targets = [
        tree_log_prior(tree, x, min_leaf, max_depth, alpha, beta)
        + tree_log_marginal(tree, len(x), y, leaf_var, noise_var)
        for tree in trees
    ]
    target = normalized_weights(log_targets)
    matrix = [[0.0 for _ in trees] for _ in trees]
    for i, tree in enumerate(trees):
        p_grow, _, _, _ = move_probabilities(
            tree, x, min_leaf, max_depth, "exact" if variant == "repaired" else "frozen"
        )
        n_leaves = len(leaves(tree, len(x)))
        for neighbor, edge in grow_edges(tree, x, min_leaf, max_depth):
            j = index[tree_key(neighbor)]
            q_forward = p_grow / n_leaves * float(edge["gap"]) / float(edge["range"])
            if q_forward <= 0:
                continue
            log_grow, _ = edge_log_ratio(
                tree, neighbor, edge, x=x, y=y, min_leaf=min_leaf,
                max_depth=max_depth, alpha=alpha, beta=beta,
                leaf_var=leaf_var, noise_var=noise_var, variant=variant,
            )
            acceptance = 1.0 if log_grow >= 0 else math.exp(log_grow)
            matrix[i][j] += q_forward * acceptance

        for neighbor, node_info in prune_edges(tree, len(x)):
            j = index[tree_key(neighbor)]
            _, p_prune, _, _ = move_probabilities(
                tree, x, min_leaf, max_depth, "exact" if variant == "repaired" else "frozen"
            )
            q_forward = p_prune / leaf_parent_count(tree, len(x))
            if q_forward <= 0:
                continue
            _, log_prune = edge_log_ratio(
                neighbor, tree, node_info, x=x, y=y, min_leaf=min_leaf,
                max_depth=max_depth, alpha=alpha, beta=beta,
                leaf_var=leaf_var, noise_var=noise_var, variant=variant,
            )
            acceptance = 1.0 if log_prune >= 0 else math.exp(log_prune)
            matrix[i][j] += q_forward * acceptance

    for i in range(len(trees)):
        off_diagonal = math.fsum(matrix[i][j] for j in range(len(trees)) if j != i)
        if off_diagonal > 1 + EXACT_TOL:
            raise ArithmeticError(f"transition row {i} exceeds one: {off_diagonal}")
        matrix[i][i] = max(0.0, 1.0 - off_diagonal)
    return matrix, target, keys


def matrix_diagnostics(matrix: list[list[float]], target: list[float]) -> dict[str, float]:
    n = len(matrix)
    row_error = max(abs(math.fsum(row) - 1.0) for row in matrix)
    stationarity_error = max(
        abs(math.fsum(target[i] * matrix[i][j] for i in range(n)) - target[j])
        for j in range(n)
    )
    detailed_balance_error = max(
        abs(target[i] * matrix[i][j] - target[j] * matrix[j][i])
        for i in range(n)
        for j in range(n)
    )
    return {
        "row_sum_error": row_error,
        "stationarity_error": stationarity_error,
        "detailed_balance_error": detailed_balance_error,
    }


def exact_transition_controls() -> dict[str, object]:
    x = [float(i) for i in range(8)]
    y = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    alpha, beta = 0.25, 3.0
    min_leaf, max_depth = 2, 2
    leaf_var, noise_var = 0.05, 0.5
    states = enumerate_trees(len(x), min_leaf, max_depth)
    original_matrix, original_target, state_keys = transition_matrix(
        states, x=x, y=y, min_leaf=min_leaf, max_depth=max_depth,
        alpha=alpha, beta=beta, leaf_var=leaf_var, noise_var=noise_var,
        variant="original",
    )
    repaired_matrix, repaired_target, repaired_keys = transition_matrix(
        states, x=x, y=y, min_leaf=min_leaf, max_depth=max_depth,
        alpha=alpha, beta=beta, leaf_var=leaf_var, noise_var=noise_var,
        variant="repaired",
    )
    if state_keys != repaired_keys:
        raise AssertionError("state enumeration differs between variants")
    original_diag = matrix_diagnostics(original_matrix, original_target)
    repaired_diag = matrix_diagnostics(repaired_matrix, repaired_target)

    def classify(diag: dict[str, float]) -> str:
        return "PASS" if max(diag.values()) <= EXACT_TOL else "EXPECTED_FAILURE"

    # A separate capped fixture isolates max-depth move availability.
    cap_trees = enumerate_trees(8, 2, 1)
    cap_original, cap_target, cap_keys = transition_matrix(
        cap_trees, x=x, y=y, min_leaf=2, max_depth=1,
        alpha=alpha, beta=beta, leaf_var=leaf_var, noise_var=noise_var,
        variant="original",
    )
    cap_repaired, cap_target_repaired, _ = transition_matrix(
        cap_trees, x=x, y=y, min_leaf=2, max_depth=1,
        alpha=alpha, beta=beta, leaf_var=leaf_var, noise_var=noise_var,
        variant="repaired",
    )
    cap_original_diag = matrix_diagnostics(cap_original, cap_target)
    cap_repaired_diag = matrix_diagnostics(cap_repaired, cap_target_repaired)

    # Preserve the previously recorded 20-row root-vs-split witness as an
    # independently reconstructed exact two-state matrix.
    two_x = [1.0] * 10 + [2.0] * 10
    two_y = [-1.0] * 10 + [1.0] * 10
    two_alpha, two_beta = 0.25, 3.0
    two_min_leaf, two_max_depth = 5, 5
    two_leaf_var, two_noise_var = 0.005, 0.5
    all_two_trees = enumerate_trees(len(two_x), two_min_leaf, two_max_depth)
    two_trees = [
        tree for tree in all_two_trees
        if math.isfinite(tree_log_prior(tree, two_x, two_min_leaf, two_max_depth,
                                        two_alpha, two_beta))
    ]
    two_old_matrix, two_target, two_keys = transition_matrix(
        two_trees, x=two_x, y=two_y, min_leaf=two_min_leaf,
        max_depth=two_max_depth, alpha=two_alpha, beta=two_beta,
        leaf_var=two_leaf_var, noise_var=two_noise_var, variant="original",
    )
    two_new_matrix, two_new_target, _ = transition_matrix(
        two_trees, x=two_x, y=two_y, min_leaf=two_min_leaf,
        max_depth=two_max_depth, alpha=two_alpha, beta=two_beta,
        leaf_var=two_leaf_var, noise_var=two_noise_var, variant="repaired",
    )
    two_old_diag = matrix_diagnostics(two_old_matrix, two_target)
    two_new_diag = matrix_diagnostics(two_new_matrix, two_new_target)
    two_log_bf = (
        0.5 * math.log1p(20 * two_leaf_var / two_noise_var)
        - math.log1p(10 * two_leaf_var / two_noise_var)
        + two_leaf_var * 10**2
        / (two_noise_var * (two_noise_var + 10 * two_leaf_var))
    )
    two_prior_odds = (
        two_alpha * (1 - split_probability(two_alpha, two_beta, 1)) ** 2
        / (1 - two_alpha)
    )
    two_posterior_odds = two_prior_odds * math.exp(two_log_bf)

    # Move availability and reverse-state counts are exposed explicitly.
    root: TreeState = None
    root_to_mid = next((tree for tree, edge in grow_edges(root, x, 2, 2) if int(edge["cut"]) == 4), None)
    if root_to_mid is None:
        raise AssertionError("closed topology fixture lost root cut 4")
    interior_edge = next(
        (tree, edge)
        for tree, edge in grow_edges(root_to_mid, x, 2, 2)
        if int(edge["cut"]) == 2 and edge["path"] == (0,)
    )
    interior_child, interior_meta = interior_edge
    root_with_large_children = next(
        tree for tree in states if tree is not None and tree.cut == 2
    )
    leafparent_growth = next(
        (tree, edge)
        for tree, edge in grow_edges(root_with_large_children, x, 2, 2)
        if edge["path"] == (1,) and int(edge["cut"]) == 4
    )
    leafparent_growth_child, leafparent_growth_meta = leafparent_growth
    root_metrics = {
        "state": tree_key(root), "leaves": len(leaves(root, 8)),
        "leaf_parents": leaf_parent_count(root, 8),
        "exact_grow_probability": move_probabilities(root, x, 2, 2, "exact")[0],
        "frozen_grow_probability": move_probabilities(root, x, 2, 2, "frozen")[0],
    }
    interior_metrics = {
        "from_state": tree_key(root_to_mid), "to_state": tree_key(interior_child),
        "split_leaf_depth": interior_meta["depth"],
        "split_leaf_rows": int(interior_meta["hi"]) - int(interior_meta["lo"]),
        "at_min_leaf_equality": int(interior_meta["hi"]) - int(interior_meta["lo"]) == 2 * min_leaf,
        "leaves_before": len(leaves(root_to_mid, 8)),
        "leaves_after": len(leaves(interior_child, 8)),
        "leaf_parents_before": leaf_parent_count(root_to_mid, 8),
        "leaf_parents_after": leaf_parent_count(interior_child, 8),
        "frozen_grow_leaf_parent_guess_after": leaf_parent_count(root_to_mid, 8) + 1,
        "exact_move_probabilities_before": list(move_probabilities(root_to_mid, x, 2, 2, "exact")[:2]),
        "frozen_move_probabilities_before": list(move_probabilities(root_to_mid, x, 2, 2, "frozen")[:2]),
        "exact_move_probabilities_after": list(move_probabilities(interior_child, x, 2, 2, "exact")[:2]),
        "frozen_move_probabilities_after": list(move_probabilities(interior_child, x, 2, 2, "frozen")[:2]),
        "frozen_nonconstant_helper_for_split_children": [
            frozen_node_nonconstant(x[0:2]), frozen_node_nonconstant(x[2:4])
        ],
    }
    leafparent_metrics = {
        "from_state": tree_key(root_with_large_children),
        "to_state": tree_key(leafparent_growth_child),
        "grow_leaf_path": leafparent_growth_meta["path"],
        "grow_leaf_rows": int(leafparent_growth_meta["hi"]) - int(leafparent_growth_meta["lo"]),
        "legacy_move_can_select_grow": move_probabilities(
            root_with_large_children, x, 2, 2, "frozen"
        )[0] > 0,
        "leaves_before": len(leaves(root_with_large_children, 8)),
        "leaves_after": len(leaves(leafparent_growth_child, 8)),
        "leaf_parents_before": leaf_parent_count(root_with_large_children, 8),
        "leaf_parents_after": leaf_parent_count(leafparent_growth_child, 8),
        "legacy_proposed_leaf_parent_guess": leaf_parent_count(root_with_large_children, 8) + 1,
        "correct_reverse_prune_probability": move_probabilities(
            leafparent_growth_child, x, 2, 2, "exact"
        )[1],
    }

    maxdepth_tree = next(tree for tree in cap_trees if tree is not None and tree.cut == 2)
    maxdepth_metrics = {
        "state": tree_key(maxdepth_tree),
        "max_depth": 1,
        "leaf_depths": [leaf["depth"] for leaf in leaves(maxdepth_tree, 8)],
        "exact_move_probabilities": list(move_probabilities(maxdepth_tree, x, 2, 1, "exact")[:2]),
        "frozen_move_probabilities": list(move_probabilities(maxdepth_tree, x, 2, 1, "frozen")[:2]),
    }

    eq_x = [0.0, 1.0, 2.0, 3.0]
    eq_root: TreeState = None
    eq_growable = bool(grow_edges(eq_root, eq_x, 2, 2))
    _, _, frozen_eq_grow, frozen_eq_prune = move_probabilities(eq_root, eq_x, 2, 2, "frozen")
    minleaf_metrics = {
        "n": 4,
        "min_leaf": 2,
        "valid_cut_at_equality": eq_growable,
        "exact_move_probabilities": list(move_probabilities(eq_root, eq_x, 2, 2, "exact")[:2]),
        "frozen_move_flags": {"grow_possible": frozen_eq_grow, "prune_possible": frozen_eq_prune},
        "frozen_kernel_has_any_move": frozen_eq_grow or frozen_eq_prune,
    }

    return {
        "scope": "exact one-dimensional induced-partition chain through depth 2",
        "posterior_reference": "independent product tree prior and integrated Gaussian leaf likelihood",
        "fixture": {
            "x": x, "y": y, "alpha": alpha, "beta": beta,
            "min_leaf": min_leaf, "max_depth": max_depth,
            "leaf_variance": leaf_var, "noise_variance": noise_var,
            "state_count": len(states),
        },
        "tolerances": {"exact_matrix_abs": EXACT_TOL},
        "original": {
            "transition_matrix": original_matrix,
            "stationary_target": original_target,
            "diagnostics": original_diag,
            "classification": classify(original_diag),
        },
        "repaired_equations": {
            "transition_matrix": repaired_matrix,
            "stationary_target": repaired_target,
            "diagnostics": repaired_diag,
            "classification": classify(repaired_diag),
        },
        "max_depth_move_control": {
            "state_count": len(cap_trees),
            "original_diagnostics": cap_original_diag,
            "original_classification": classify(cap_original_diag),
            "repaired_equation_diagnostics": cap_repaired_diag,
            "repaired_equation_classification": classify(cap_repaired_diag),
            "states": cap_keys,
        },
        "known_exact_two_state_root_control": {
            "scope": "one numeric split gap, 20 rows, production effect-tree hyperparameters",
            "state_keys": two_keys,
            "log_bayes_factor": two_log_bf,
            "prior_split_odds": two_prior_odds,
            "intended_posterior_split_odds": two_posterior_odds,
            "intended_split_probability": two_posterior_odds / (1 + two_posterior_odds),
            "original_transition_matrix": two_old_matrix,
            "original_diagnostics": two_old_diag,
            "original_classification": classify(two_old_diag),
            "repaired_equation_transition_matrix": two_new_matrix,
            "repaired_equation_diagnostics": two_new_diag,
            "repaired_equation_classification": classify(two_new_diag),
            "original_coded_prune_acceptance": two_old_matrix[1][0],
            "correct_prune_acceptance": two_new_matrix[1][0],
        },
        "availability_and_reverse_state_controls": {
            "root": root_metrics,
            "interior_minleaf_equality_and_leaf_parent_count": interior_metrics,
            "growable_leaf_parent_reverse_count": leafparent_metrics,
            "max_depth": maxdepth_metrics,
            "root_minleaf_equality": minleaf_metrics,
        },
        "state_keys": state_keys,
    }


def numeric_grid(values: Iterable[float], grid_size: int, variant: str) -> dict[str, object]:
    """Pure-Python reproduction of numeric GFR stride enumeration."""

    x = sorted(float(value) for value in values)
    n = len(x)
    if n == 0:
        raise ValueError("values must be nonempty")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if n <= grid_size:
        bins: list[tuple[int, int]] = []
        begin = 0
        length = 0
        for i, value in enumerate(x):
            length += 1
            if i == n - 1 or abs(x[i + 1] - value) > STOCHTREE_NUMERIC_EPSILON:
                bins.append((begin, length))
                begin += length
                length = 0
    elif max(x) - min(x) < STOCHTREE_NUMERIC_EPSILON:
        bins = [(0, n)]
    else:
        step_size = n // grid_size
        if step_size < 1:
            raise ValueError("large-node grid requires step_size >= 1")
        bins = []
        stride_begin = 0
        stride_length = 0
        for i, value in enumerate(x):
            stride_length += 1
            last = i == n - 1
            if last:
                bins.append((stride_begin, stride_length))
                continue
            distinct_boundary = abs(x[i + 1] - value) > STOCHTREE_NUMERIC_EPSILON
            if variant == "original":
                bin_complete = stride_length <= step_size and stride_length + 1 > step_size
                stride_complete = bin_complete and distinct_boundary
            elif variant == "repaired":
                stride_complete = stride_length >= step_size and distinct_boundary
            else:
                raise ValueError(variant)
            if stride_complete:
                bins.append((stride_begin, stride_length))
                stride_begin += stride_length
                stride_length = 0
    if variant not in {"original", "repaired"}:
        raise ValueError(variant)
    return {
        "sorted_values": x,
        "grid_size": grid_size,
        "step_size": n // grid_size if n > grid_size else None,
        "bin_starts": [start for start, _ in bins],
        "bin_lengths": [length for _, length in bins],
        "representative_cutpoints": [x[start + length - 1] for start, length in bins],
        "valid_gfr_cutpoint_count": max(0, len(bins) - 1),
    }


def gfr_stride_controls() -> dict[str, object]:
    cases = {
        "untied_positive_control": [float(i) for i in range(10)],
        "boundary_tied_regression": [0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "all_tied_negative_control": [3.0] * 10,
    }
    results: dict[str, object] = {}
    for name, values in cases.items():
        old = numeric_grid(values, grid_size=5, variant="original")
        repaired = numeric_grid(values, grid_size=5, variant="repaired")
        if name == "untied_positive_control":
            expected = old["valid_gfr_cutpoint_count"] == repaired["valid_gfr_cutpoint_count"] > 0
            original_classification = "PASS" if expected else "FAIL"
            repaired_classification = "PASS" if expected else "FAIL"
        elif name == "boundary_tied_regression":
            expected = old["valid_gfr_cutpoint_count"] == 0 and repaired["valid_gfr_cutpoint_count"] > 0
            original_classification = "EXPECTED_FAILURE" if expected else "UNEXPECTED_PASS"
            repaired_classification = "PASS" if expected else "FAIL"
        else:
            expected = old["valid_gfr_cutpoint_count"] == repaired["valid_gfr_cutpoint_count"] == 0
            original_classification = "PASS" if expected else "FAIL"
            repaired_classification = "PASS" if expected else "FAIL"
        results[name] = {
            "input_values": values,
            "original": old,
            "proposed_repaired_stride": repaired,
            "expected_relation_holds": expected,
            "original_classification": original_classification,
            "proposed_repaired_classification": repaired_classification,
        }
    return {
        "scope": "numeric GFR stride generation; exact integer-grid fixture",
        "grid_size": 5,
        "tie_semantics": "a repeated-value group stays together; distinct boundary emits a bin",
        "cases": results,
    }


def var_split_range(values: Iterable[float], variant: str) -> tuple[float, float]:
    x = [float(value) for value in values]
    if variant == "original":
        lo = sys.float_info.max
        hi = sys.float_info.min  # mirrors numeric_limits<double>::min(), not lowest()
        for value in x:
            if value < lo:
                lo = value
            elif value > hi:
                hi = value
        return lo, hi
    if variant == "repaired":
        return min(x), max(x)
    raise ValueError(variant)


def continuous_extrema_controls() -> dict[str, object]:
    cases = {
        "all_negative": [-9.0, -3.0, -5.0, -1.0],
        "positive_descending": [4.0, 3.0, 2.0, 1.0],
        "nonmonotonic_missed_initial_max": [10.0, 5.0, 7.0, 3.0],
    }
    output: dict[str, object] = {}
    for name, values in cases.items():
        expected = (min(values), max(values))
        old = var_split_range(values, "original")
        repaired = var_split_range(values, "repaired")
        output[name] = {
            "input_order": values,
            "expected_extrema": list(expected),
            "original_extrema": list(old),
            "proposed_repaired_extrema": list(repaired),
            "original_is_expected_failure": old != expected,
            "repaired_matches": repaired == expected,
            "original_classification": "EXPECTED_FAILURE" if old != expected else "UNEXPECTED_PASS",
            "proposed_repaired_classification": "PASS" if repaired == expected else "FAIL",
            "classification": "PASS" if repaired == expected else "FAIL",
        }
    return {
        "scope": "VarSplitRange scalar reduction under arbitrary row order",
        "cases": output,
    }


def _native_mcmc_probe(variant: str) -> dict[str, object]:
    """Run the actual installed one-tree native MCMC API on the exact two-state fixture."""

    import importlib.metadata
    import numpy as np
    from stochtree import (
        Dataset, Forest, ForestContainer, ForestModelConfig, ForestSampler,
        GlobalModelConfig, Residual, RNG,
    )

    alpha, beta, leaf_variance, noise_variance = 0.25, 3.0, 0.005, 0.5
    child_n = 10
    x = np.repeat([1.0, 2.0], child_n).reshape(-1, 1)
    y = np.repeat([-1.0, 1.0], child_n)
    dataset = Dataset()
    dataset.add_covariates(x)
    dataset.add_basis(np.ones(y.size))
    global_config = GlobalModelConfig(global_error_variance=noise_variance)
    config = ForestModelConfig(
        num_trees=1, num_features=1, num_observations=y.size,
        feature_types=np.zeros(1, dtype=int), variable_weights=np.ones(1),
        leaf_dimension=1, alpha=alpha, beta=beta, min_samples_leaf=5,
        max_depth=5, leaf_model_type=1,
        leaf_model_scale=np.array([[leaf_variance]]), cutpoint_grid_size=100,
    )
    sampler = ForestSampler(dataset, global_config, config)
    forest = Forest(1, 1, False, False)
    container = ForestContainer(1, 1, False, False)
    residual = Residual(y.copy())
    sampler.prepare_for_sampler(dataset, residual, forest, 1, np.array([0.0]))
    rng = RNG(20260929 if variant == "original" else 20260930)
    transitions = [[0, 0], [0, 0]]
    old = 0
    split_visits = 0
    max_residual_error = 0.0
    for k in range(NATIVE_MCMC_BURN + NATIVE_MCMC_DRAWS):
        sampler.sample_one_iteration(
            container, forest, dataset, residual, rng,
            global_config, config, False, False,
        )
        new = int(forest.num_nodes(0) > 1)
        if k >= NATIVE_MCMC_BURN:
            transitions[old][new] += 1
            split_visits += new
        predicted = np.asarray(forest.predict(dataset)).reshape(-1)
        actual_residual = np.asarray(residual.get_residual()).reshape(-1)
        max_residual_error = max(
            max_residual_error,
            float(np.max(np.abs(actual_residual - (y - predicted)))),
        )
        old = new

    log_bf = (
        0.5 * math.log1p(y.size * leaf_variance / noise_variance)
        - math.log1p(child_n * leaf_variance / noise_variance)
        + leaf_variance * child_n**2
        / (noise_variance * (noise_variance + child_n * leaf_variance))
    )
    prior_odds = alpha * (1 - alpha / (2**beta))**2 / (1 - alpha)
    posterior_odds = prior_odds * math.exp(log_bf)
    target_split = posterior_odds / (1 + posterior_odds)
    observed_split = split_visits / NATIVE_MCMC_DRAWS
    passed = abs(observed_split - target_split) <= NATIVE_MCMC_ABS_TOL and max_residual_error <= 1e-10
    if max_residual_error > 1e-10:
        expected_class = "FAIL"
    elif passed:
        expected_class = "UNEXPECTED_PASS" if variant == "original" else "PASS"
    else:
        expected_class = "EXPECTED_FAILURE" if variant == "original" else "FAIL"
    return {
        "backend": "actual installed stochtree Python/C++ API",
        "loaded_stochtree_version": importlib.metadata.version("stochtree"),
        "variant_label": variant,
        "seed": 20260929 if variant == "original" else 20260930,
        "burnin": NATIVE_MCMC_BURN,
        "retained_draws": NATIVE_MCMC_DRAWS,
        "transition_counts": transitions,
        "target_split_probability": target_split,
        "observed_split_fraction": observed_split,
        "absolute_tolerance": NATIVE_MCMC_ABS_TOL,
        "max_residual_consistency_error": max_residual_error,
        "pass_condition_met": passed,
        "classification": expected_class,
        "note": "classification is based on sampled native behavior, not source equations",
    }


def _native_gfr_stride_probe(variant: str) -> dict[str, object]:
    """Probe actual GFR cutpoint-grid behavior with untied, boundary-tied, all-tied x."""

    import numpy as np
    from stochtree import (
        Dataset, Forest, ForestContainer, ForestModelConfig, ForestSampler,
        GlobalModelConfig, Residual, RNG,
    )

    cases = {
        "untied_positive_control": np.arange(10.0),
        "boundary_tied_regression": np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
        "all_tied_negative_control": np.repeat(3.0, 10),
    }
    observed: dict[str, object] = {}
    for case_name, x_values in cases.items():
        split_count = 0
        root_thresholds: list[Optional[float]] = []
        root_left_row_counts: list[Optional[int]] = []
        threshold_tie_routed_left: list[bool] = []
        for repeat in range(NATIVE_GFR_REPEATS):
            y = np.where(x_values <= 1.0, -10.0, 10.0)
            data = Dataset()
            data.add_covariates(x_values.reshape(-1, 1))
            global_config = GlobalModelConfig(global_error_variance=0.5)
            config = ForestModelConfig(
                num_trees=1, num_features=1, num_observations=len(y),
                feature_types=np.zeros(1, dtype=int), variable_weights=np.ones(1),
                leaf_dimension=1, alpha=0.25, beta=3.0, min_samples_leaf=2,
                max_depth=1, leaf_model_type=0,
                leaf_model_scale=np.array([[0.05]]), cutpoint_grid_size=5,
            )
            sampler = ForestSampler(data, global_config, config)
            forest = Forest(1, 1, True, False)
            container = ForestContainer(1, 1, True, False)
            residual = Residual(y.copy())
            sampler.prepare_for_sampler(data, residual, forest, 0, np.array([0.0]))
            seed_offset = 10_000 if variant == "original" else 20_000
            rng = RNG(seed_offset + repeat)
            sampler.sample_one_iteration(
                container, forest, data, residual, rng,
                global_config, config, False, True,
            )
            has_split = forest.num_nodes(0) > 1
            split_count += int(has_split)
            if has_split:
                threshold = float(forest.node_split_threshold(0, 0))
                left_rows = int(np.sum(x_values <= threshold))
                root_thresholds.append(threshold)
                root_left_row_counts.append(left_rows)
                if case_name == "boundary_tied_regression" and abs(threshold - 1.0) <= 1e-12:
                    predictions = np.asarray(forest.predict(data)).reshape(-1)
                    threshold_tie_routed_left.append(
                        bool(np.max(np.abs(predictions[:3] - predictions[0])) <= 1e-12)
                    )
            else:
                root_thresholds.append(None)
                root_left_row_counts.append(None)
        if case_name == "boundary_tied_regression":
            threshold_one_count = sum(
                value is not None and abs(value - 1.0) <= 1e-12
                for value in root_thresholds
            )
            pass_condition = (
                split_count >= NATIVE_GFR_MIN_BOUNDARY_SPLITS
                and threshold_one_count >= NATIVE_GFR_MIN_BOUNDARY_SPLITS
                and len(threshold_tie_routed_left) == threshold_one_count
                and all(threshold_tie_routed_left)
                and all(
                    count == 3
                    for value, count in zip(root_thresholds, root_left_row_counts)
                    if value is not None and abs(value - 1.0) <= 1e-12
                )
            )
        elif case_name == "all_tied_negative_control":
            pass_condition = split_count == 0
        else:
            pass_condition = split_count >= NATIVE_GFR_MIN_BOUNDARY_SPLITS
        if case_name == "boundary_tied_regression" and variant == "original":
            classification = "EXPECTED_FAILURE" if not pass_condition else "UNEXPECTED_PASS"
        else:
            classification = "PASS" if pass_condition else (
                "EXPECTED_FAILURE" if variant == "original" else "FAIL"
            )
        observed[case_name] = {
            "repeats": NATIVE_GFR_REPEATS,
            "observed_root_split_count": split_count,
            "root_split_thresholds": root_thresholds,
            "root_left_row_counts": root_left_row_counts,
            "threshold_value_ties_routed_left": threshold_tie_routed_left,
            "threshold_one_root_split_count": sum(
                value is not None and abs(value - 1.0) <= 1e-12
                for value in root_thresholds
            ),
            "minimum_split_count_for_positive_case": NATIVE_GFR_MIN_BOUNDARY_SPLITS,
            "pass_condition_met": pass_condition,
            "classification": classification,
        }
    return {
        "backend": "actual installed stochtree Python/C++ API with gfr=True",
        "variant_label": variant,
        "results": observed,
        "note": "sampling probe; pure grid-equivalence controls are also reported separately",
    }


def _native_extrema_probe(variant: str) -> dict[str, object]:
    """Probe MCMC's actual VarSplitRange effect on root grow proposals."""

    import numpy as np
    from stochtree import (
        Dataset, Forest, ForestContainer, ForestModelConfig, ForestSampler,
        GlobalModelConfig, Residual, RNG,
    )

    x_values = np.repeat([-9.0, -1.0], 10)
    y = np.repeat([-1.0, 1.0], 10)
    splits = 0
    for repeat in range(NATIVE_EXTREMA_REPEATS):
        data = Dataset()
        data.add_covariates(x_values.reshape(-1, 1))
        data.add_basis(np.ones(y.size))
        global_config = GlobalModelConfig(global_error_variance=0.5)
        config = ForestModelConfig(
            num_trees=1, num_features=1, num_observations=y.size,
            feature_types=np.zeros(1, dtype=int), variable_weights=np.ones(1),
            leaf_dimension=1, alpha=0.25, beta=3.0, min_samples_leaf=5,
            max_depth=5, leaf_model_type=1,
            leaf_model_scale=np.array([[0.005]]), cutpoint_grid_size=100,
        )
        sampler = ForestSampler(data, global_config, config)
        forest = Forest(1, 1, False, False)
        container = ForestContainer(1, 1, False, False)
        residual = Residual(y.copy())
        sampler.prepare_for_sampler(data, residual, forest, 1, np.array([0.0]))
        seed_offset = 30_000 if variant == "original" else 40_000
        rng = RNG(seed_offset + repeat)
        sampler.sample_one_iteration(
            container, forest, data, residual, rng,
            global_config, config, False, False,
        )
        splits += int(forest.num_nodes(0) > 1)
    passed = splits >= NATIVE_EXTREMA_MIN_SPLITS
    if passed and variant == "original":
        classification = "UNEXPECTED_PASS"
    elif passed:
        classification = "PASS"
    else:
        classification = "EXPECTED_FAILURE" if variant == "original" else "FAIL"
    return {
        "backend": "actual installed stochtree Python/C++ API with gfr=False",
        "variant_label": variant,
        "x_values": [-9.0, -1.0],
        "repeats": NATIVE_EXTREMA_REPEATS,
        "observed_root_split_count": splits,
        "required_split_count": NATIVE_EXTREMA_MIN_SPLITS,
        "pass_condition_met": passed,
        "classification": classification,
        "note": "the negative-x control exercises the compiled VarSplitRange through MCMC",
    }


def run_native_probes(variant: str) -> dict[str, object]:
    if variant not in {"original", "repaired"}:
        raise ValueError("native variant must be original or repaired")
    import importlib.metadata
    import stochtree_cpp

    binary_path = stochtree_cpp.__file__
    with open(binary_path, "rb") as binary_file:
        binary_sha256 = hashlib.sha256(binary_file.read()).hexdigest()
    mcmc = _native_mcmc_probe(variant)
    gfr = _native_gfr_stride_probe(variant)
    extrema = _native_extrema_probe(variant)
    if variant == "original":
        native_gate = (
            "ORIGINAL_EXPECTED_FAILURES_REPRODUCED"
            if mcmc["classification"] == "EXPECTED_FAILURE"
            and gfr["results"]["untied_positive_control"]["classification"] == "PASS"
            and gfr["results"]["boundary_tied_regression"]["classification"] == "EXPECTED_FAILURE"
            and gfr["results"]["all_tied_negative_control"]["classification"] == "PASS"
            and extrema["classification"] == "EXPECTED_FAILURE"
            else "FAIL"
        )
    else:
        native_gate = (
            "PASS"
            if mcmc["classification"] == "PASS"
            and all(item["classification"] == "PASS" for item in gfr["results"].values())
            and extrema["classification"] == "PASS"
            else "FAIL"
        )
    return {
        "variant_label": variant,
        "loaded_native_binary": {
            "module_path": binary_path,
            "sha256": binary_sha256,
            "stochtree_distribution_version": importlib.metadata.version("stochtree"),
        },
        "mcmc_exact_two_state": mcmc,
        "gfr_stride_ties": gfr,
        "continuous_extrema": extrema,
        "native_scope_gate": native_gate,
        "certification_rule": "Only these loaded-binary probes report native behavior; equation/grid emulations are not binary certification.",
    }


def run_controls(native: bool = False, variant: str = "repaired") -> dict[str, object]:
    exact = exact_transition_controls()
    gfr_grid = gfr_stride_controls()
    extrema = continuous_extrema_controls()
    exact_gates = (
        exact["original"]["classification"] == "EXPECTED_FAILURE"
        and exact["repaired_equations"]["classification"] == "PASS"
        and exact["known_exact_two_state_root_control"]["original_classification"] == "EXPECTED_FAILURE"
        and exact["known_exact_two_state_root_control"]["repaired_equation_classification"] == "PASS"
        and exact["max_depth_move_control"]["original_classification"] == "EXPECTED_FAILURE"
        and exact["max_depth_move_control"]["repaired_equation_classification"] == "PASS"
    )
    gfr_gates = all(
        item["expected_relation_holds"]
        and item["proposed_repaired_classification"] == "PASS"
        for item in gfr_grid["cases"].values()
    ) and gfr_grid["cases"]["boundary_tied_regression"]["original_classification"] == "EXPECTED_FAILURE"
    extrema_gates = all(
        item["original_classification"] == "EXPECTED_FAILURE"
        and item["proposed_repaired_classification"] == "PASS"
        for item in extrema["cases"].values()
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "suite": "bounded_native_repair_controls",
        "python": platform.python_version(),
        "deterministic_scope_gate": "PASS" if exact_gates and gfr_gates and extrema_gates else "FAIL",
        "deterministic_gate_components": {
            "exact_transition_equations": "PASS" if exact_gates else "FAIL",
            "gfr_stride_ties": "PASS" if gfr_gates else "FAIL",
            "continuous_extrema": "PASS" if extrema_gates else "FAIL",
        },
        "exact_transition_controls": exact,
        "gfr_stride_controls": gfr_grid,
        "continuous_extrema_controls": extrema,
        "native_binary_probes": {"status": "NOT_RUN"},
    }
    if native:
        result["native_binary_probes"] = run_native_probes(variant)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="run installed stochtree binary probes")
    parser.add_argument("--variant", choices=("original", "repaired"), default="repaired")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()
    result = run_controls(native=args.native, variant=args.variant)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(payload)
        native_result = result["native_binary_probes"]
        print(json.dumps({
            "output": args.output,
            "deterministic_scope_gate": result["deterministic_scope_gate"],
            "native_scope_gate": native_result.get("native_scope_gate", native_result.get("status")),
        }, indent=2))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
