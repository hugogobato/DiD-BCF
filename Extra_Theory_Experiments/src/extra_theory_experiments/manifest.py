"""Task manifests, provenance, and deterministic replication/shard seeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "configs"


def deterministic_seed(*parts: object, base_seed: int = 20260829) -> int:
    """Return a stable 32-bit seed, independent of Python's hash randomisation."""
    payload = "|".join([str(base_seed), *(str(x) for x in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32 - 1)


@dataclass(frozen=True)
class ExperimentTask:
    family: str
    design: str
    degree: int
    N: int
    rep: int
    estimator: str
    shard: int = 0
    n_shards: int = 1
    oracle_available: bool = False
    config_hash: str = ""
    base_seed: int = 20260829

    @property
    def seed(self) -> int:
        # This is the replication/DGP seed, so every estimator sees identical
        # latent data and truth.  Estimator-specific posterior seeds are derived
        # separately by the runner/reference adapter.
        return deterministic_seed(self.family, self.design, self.degree,
                                  self.N, self.rep,
                                  base_seed=self.base_seed)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["seed"] = self.seed
        return out


def _load_config(family: str, config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = DEFAULT_CONFIG_DIR / (
            "correction_audit.json" if family == "correction_audit"
            else "information_ablation.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("family") != family:
        raise ValueError(f"Config family {config.get('family')!r} does not match {family!r}")
    return config


def _config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def oracle_is_supported(family: str, design: str, degree: int,
                        config: dict[str, Any]) -> bool:
    """Check a manifest declaration, never silently substitute a pilot.

    The shipped DGPs add logistic-normal assignment noise.  Consequently an
    exact propensity is not available from the observed panel alone.  A future
    DGP may set ``oracle_supported_degrees`` explicitly after exposing exact
    nuisance columns; this gate keeps unsupported requests visible.
    """
    spec = config.get("oracle_supported", {})
    values = spec.get(design, []) if isinstance(spec, dict) else []
    return int(degree) in [int(x) for x in values]


def build_manifest(
    family: str,
    *,
    config_path: str | Path | None = None,
    reps: int | None = None,
    base_seed: int = 20260829,
    shard_id: int = 0,
    n_shards: int = 1,
    include_unavailable: bool = True,
) -> list[ExperimentTask]:
    """Expand a JSON family specification into deterministic shard tasks.

    Sharding is performed after sorting bundles by design, degree, N, and
    replication. Every estimator in a bundle is assigned to the same shard, so
    paired methods can share cached fits and changing worker counts does not
    change which replication is run by a shard.
    """
    if n_shards < 1 or not 0 <= shard_id < n_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < n_shards")
    config = _load_config(family, config_path)
    config_hash = _config_hash(config)
    methods = list(config["estimators"])
    out: list[ExperimentTask] = []
    for design_spec in config["designs"]:
        design = str(design_spec["name"])
        degrees = [int(x) for x in design_spec["degrees"]]
        Ns = [int(x) for x in design_spec["N"]]
        n_rep = int(design_spec.get("reps", config.get("reps", 100)))
        if reps is not None:
            n_rep = int(reps)
        for degree in degrees:
            for N in Ns:
                for rep in range(n_rep):
                    for estimator in methods:
                        available = True
                        if estimator.startswith("oracle"):
                            available = oracle_is_supported(family, design, degree, config)
                        if not available and not include_unavailable:
                            continue
                        out.append(ExperimentTask(
                            family=family, design=design, degree=degree, N=N,
                            rep=rep, estimator=estimator, oracle_available=available,
                            config_hash=config_hash, base_seed=base_seed,
                        ))
    out.sort(key=lambda t: (t.design, t.degree, t.N, t.rep, t.estimator))
    # Keep every estimator for one generated panel in the same shard.  This is
    # required for paired comparisons and permits the runner to cache one
    # full-panel posterior per replication.
    bundle_order = sorted({(t.design, t.degree, t.N, t.rep) for t in out})
    bundle_shard = {key: i % n_shards for i, key in enumerate(bundle_order)}
    return [
        ExperimentTask(
            family=task.family, design=task.design, degree=task.degree,
            N=task.N, rep=task.rep, estimator=task.estimator,
            shard=bundle_shard[(task.design, task.degree, task.N, task.rep)],
            n_shards=n_shards,
            oracle_available=task.oracle_available,
            config_hash=task.config_hash, base_seed=task.base_seed,
        )
        for task in out
        if bundle_shard[(task.design, task.degree, task.N, task.rep)] == shard_id
    ]


def manifest_frame(tasks: Iterable[ExperimentTask]):
    import pandas as pd
    return pd.DataFrame([task.as_dict() for task in tasks])


def write_manifest(tasks: Iterable[ExperimentTask], path: str | Path) -> None:
    frame = manifest_frame(tasks)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
