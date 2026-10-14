"""Seed normalization at the stochtree C++ boundary.

``stochtree_cpp.RngCpp`` is bound to a signed 32-bit integer even though the
Python-level wrappers annotate seeds as plain ``int``.  Experiment manifests
use a wide deterministic seed space (``deterministic_seed`` returns values
modulo ``2**32 - 1``), so roughly half of all task seeds overflow the binding.
Every direct ``BCFModel``/``BARTModel`` call must wrap its seed through
:func:`stochtree_seed`; NumPy continues to receive the unmodified seed, so the
Python RNG stream is unchanged.
"""

from __future__ import annotations

__all__ = ["STOCHTREE_SEED_MODULUS", "stochtree_seed"]

STOCHTREE_SEED_MODULUS = 2 ** 31


def stochtree_seed(seed: int | None) -> int:
    """Map a Python seed into the signed 32-bit range accepted by stochtree.

    ``None`` and the stochtree sentinel ``-1`` keep their documented
    nondeterministic behavior.  Every other integer, including large manifest
    seeds and negative values, is wrapped into ``[0, 2**31 - 1]``.
    """
    if seed is None:
        return -1
    value = int(seed)
    if value == -1:
        return -1
    return value % STOCHTREE_SEED_MODULUS
