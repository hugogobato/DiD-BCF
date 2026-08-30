#!/usr/bin/env python3
"""Run the controlled exact-m0 Algorithm-1 mechanics diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from extra_theory_experiments.controlled_mechanics import run_controlled_mechanics


def main():
    parser = argparse.ArgumentParser(
        description="Controlled special-case mechanics audit; no BCF fits.")
    parser.add_argument("--out-dir", default=str(HERE / "results" / "controlled_mechanics"))
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--n-units", type=int, default=200)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--smoke", action="store_true",
                        help="use one replication, 20 BB draws, and 40 units")
    args = parser.parse_args()
    reps, draws, units = args.reps, args.draws, args.n_units
    if args.smoke:
        reps, draws, units = 1, 20, 40
    _, metrics, archive = run_controlled_mechanics(
        out_dir=args.out_dir, reps=reps, n_draws=draws, n_units=units,
        base_seed=args.seed, K=args.K)
    print("controlled mechanics rows:", len(metrics))
    print(metrics.to_string(index=False))
    print("archive:", archive)


if __name__ == "__main__":
    main()
