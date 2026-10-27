"""Fit a new panel using the frozen paper model and a verified native build."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--covariates", nargs="+", default=["X1", "X2", "X3", "X4", "X5"])
    parser.add_argument("--effect-modifiers", nargs="+", default=["X1", "X2"])
    parser.add_argument("--rfx", choices=["none", "unit", "group"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gfr", type=int, default=50)
    parser.add_argument("--num-burnin", type=int, default=0)
    parser.add_argument("--num-mcmc", type=int, default=500)
    parser.add_argument("--keep-every", type=int, default=5)
    parser.add_argument("--num-chains", type=int, default=3)
    args = parser.parse_args()
    if min(args.num_gfr, args.num_burnin) < 0 or min(args.num_mcmc, args.keep_every, args.num_chains) < 1:
        parser.error("Sampling counts must be positive, except GFR/burn-in may be zero")
    args.out.mkdir(parents=True, exist_ok=True)
    targets = [args.out / "estimands.csv", args.out / "provenance.json"]
    if any(path.exists() for path in targets):
        parser.error("Use a new output directory; existing output files will not be overwritten")

    sys.path.insert(0, str(PACKAGE / "helpers"))
    from common import bootstrap, read, sha, verify_package

    _, manifest = verify_package()
    versions = {name: importlib.metadata.version(name) for name in manifest["dependencies"]}
    if versions != manifest["dependencies"]:
        raise ValueError("Dependency versions differ from requirements-paper.txt / the frozen campaign")
    if read(args.setup).get("status") != "PASS":
        raise ValueError("Native setup has not passed its controls")
    bootstrap(args.setup.resolve())
    from recovery_engine import activate, fingerprint
    engine = fingerprint(activate())
    if engine["specification"] != manifest["engine_specification"]:
        raise ValueError("Loaded native engine differs from the paper engine")

    import pandas as pd
    from didbcf_structured import ForestPrior, GlobalPrior, StructuredDiDBCF, structured_estimands

    model_source = Path(sys.modules["didbcf_structured"].__file__).resolve()
    expected_source = PACKAGE / "source_workspace/DiD-BCF/didbcf_structured"
    if model_source.parent != expected_source:
        raise RuntimeError("Python model import escaped the frozen source directory")
    panel = pd.read_csv(args.panel)
    spec = manifest["model_spec"]
    model = StructuredDiDBCF(
        level_prior=ForestPrior(**spec["forest_priors"]["level"]),
        trend_prior=ForestPrior(**spec["forest_priors"]["trend"]),
        effect_prior=ForestPrior(**spec["forest_priors"]["effect"]),
        global_prior=GlobalPrior(**spec["global_prior"]), rfx=args.rfx,
    ).sample(
        panel, covariates=args.covariates, effect_modifiers=args.effect_modifiers,
        effect_by_cohort=spec["effect_by_cohort"], num_gfr=args.num_gfr,
        num_burnin=args.num_burnin, num_mcmc=args.num_mcmc,
        keep_every=args.keep_every, num_chains=args.num_chains, seed=args.seed,
    )
    summaries = structured_estimands(model, method="DiD-BCF")
    summaries.to_csv(targets[0], index=False)
    record = {
        "purpose": "new_panel_fit_not_an_original_paper_replication",
        "panel_path": str(args.panel.resolve()), "panel_sha256": sha(args.panel),
        "source_manifest_hash": manifest["manifest_hash"], "python_model": str(model_source),
        "engine": engine, "dependencies": versions, "prior": model.prior_metadata,
        "rfx": args.rfx, "seed": args.seed,
        "covariates": args.covariates, "effect_modifiers": args.effect_modifiers,
        "sampling": {name: getattr(args, name) for name in
                     ["num_gfr", "num_burnin", "num_mcmc", "keep_every", "num_chains"]},
        "estimands_sha256": hashlib.sha256(targets[0].read_bytes()).hexdigest(),
    }
    targets[1].write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(summaries.to_string(index=False))
    print("Saved:", targets[0], "and", targets[1])


if __name__ == "__main__":
    main()
