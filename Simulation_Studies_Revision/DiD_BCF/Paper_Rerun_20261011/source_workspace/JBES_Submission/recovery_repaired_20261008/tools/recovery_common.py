"""Frozen constants, hashing, bundle and IO helpers for the recovery campaign.

This module is part of the frozen source bundle embedded in every recovery
notebook.  Editing it changes the bundle hash and therefore requires
regenerating the manifests and notebooks (see ``recovery_plan.py``).

Conventions fixed here and repeated in the new manifests:

* statistical job ID   = the original 20-hex job ID (DGP/data/prior/method);
* run ID               = hash of the statistical job, budget, chain, trace flag
                         and parent manifest, identifying source/environment,
                         budget, chain and attempt namespace;
* chain seed           = ``1000 * rep + chain_index`` (the original mapping in
                         ``didbcf_structured/model.py``);
* budget               = ``(num_gfr, num_burnin, num_mcmc, keep_every, num_chains)``
                         with per-chain sweeps
                         ``num_gfr + num_burnin + num_mcmc * keep_every``;
* workflow states      = prepared, smoke-tested, user-run, returned,
                         integrity-verified, diagnostically accepted,
                         manuscript-integrated (never conflate them).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
THIS = Path(__file__).resolve()


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "DiD-BCF" / "didbcf_structured").is_dir() and (
            parent / "JBES_Submission"
        ).is_dir():
            return parent
    raise RuntimeError(f"Could not locate the repository root above {start}")


REPO = _find_repo_root(THIS)
PROJECT = REPO / "DiD-BCF"                      # matches the old campaign layout
SIM = PROJECT / "Simulation_Studies_Revision"
CAMPAIGN = SIM / "Proper_Prior_Rerun"
RECOVERY = REPO / "JBES_Submission" / "recovery_repaired_20261008"
TOOLS = RECOVERY / "tools"
MANIFESTS = RECOVERY / "manifests"
ASSIGNMENTS = RECOVERY / "assignments"
NOTEBOOKS = RECOVERY / "notebooks"
INVENTORY = RECOVERY / "E0_inventory"
INCOMING = RECOVERY / "incoming"
PILOT = RECOVERY / "pilot"

ORIGINAL_MANIFEST = CAMPAIGN / "manifest.json"
ORIGINAL_MANIFEST_HASH = (
    "105ee8510f99817b132d69869167ec5a724596ed37d8094f7ae430c958ad1ccc"
)
ORIGINAL_SOURCE_MANIFEST_VERSION = "0.2.0-proper-ig"
ORIGINAL_SAMPLER = dict(num_gfr=50, num_burnin=0, num_mcmc=500,
                        keep_every=5, num_chains=3)
ORIGINAL_PINS = {
    "stochtree": "0.4.4",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
}
ORIGINAL_PYTHON = "3.12.3"
PINNED_PYTHON = {k: v for k,v in ORIGINAL_PINS.items() if k != "stochtree"}
PINNED_PYTHON.update(cmake="4.3.4", matplotlib="3.11.0")
DEPENDENCIES = tuple(PINNED_PYTHON)
from recovery_engine import bootstrap, specification
bootstrap()
PDTATA_PIN = "2283bea1221a152420f98dfa20f633c5d054ea51d881115c8cd702a97bcd3167"
MPDTA = PROJECT / "Empirical_Study" / "mpdta.csv"

for _path in (PROJECT, SIM, SIM / "scripts", TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

RECOVERY_VERSION = "0.2.0-repaired-20261008"
PARENT_MANIFEST = ORIGINAL_MANIFEST_HASH
BUNDLE_FORMAT = 1


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    """Installed versions of the pinned numerical dependencies."""
    return {name: importlib.metadata.version(name) for name in DEPENDENCIES}


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


# --------------------------------------------------------------------------- #
# Model specification freeze (E0.3)
# --------------------------------------------------------------------------- #
def _prior_asdict(prior) -> dict:
    from dataclasses import asdict
    return asdict(prior)


def model_spec() -> dict:
    """Every field that fixes the estimator, frozen into each new manifest."""
    manifest = __import__("json").loads((RECOVERY / "engine/build_manifest.json").read_text())
    for name, expected in manifest["project_sources"].items():
        if sha256_file(PROJECT / "didbcf_structured" / name) != expected:
            raise ValueError("Structured source differs from the validated repair model: " + name)
    if sha256_file(MPDTA) != PDTATA_PIN:
        raise ValueError("Application data differs from original pin")
    from didbcf_structured import (EFFECT_PRIOR, GLOBAL_PRIOR, LEVEL_PRIOR,
                                   TREND_PRIOR)
    from didbcf_structured.design import (DEFAULT_COVARIATES,
                                          DEFAULT_EFFECT_MODIFIERS)
    from didbcf_structured import __init__ as _pkg  # noqa: F401  (import check)
    return {
        "estimator": "didbcf_structured.StructuredDiDBCF",
        "diagnostic_estimator": "stochtree.BCFModel via fit_pretrend",
        "diagnostic_defaults": __import__("recovery_bcf_spec").defaults(),
        "effect_by_cohort": True,
        "outcome_standardization": "population_sd_ddof0",
        "standardization_source": "np.std(y, ddof=0) of the job panel",
        "global_prior": _prior_asdict(GLOBAL_PRIOR),
        "forest_priors": {
            "level": _prior_asdict(LEVEL_PRIOR),
            "trend": _prior_asdict(TREND_PRIOR),
            "effect": _prior_asdict(EFFECT_PRIOR),
        },
        "features": {
            "simulation_covariates": list(DEFAULT_COVARIATES),
            "simulation_effect_modifiers": list(DEFAULT_EFFECT_MODIFIERS),
            "application_covariates": ["lpop"],
            "application_effect_modifiers": ["lpop"],
            "diagnostic_prognostic_cols_simulation": None,
            "diagnostic_treatment_cols_simulation": None,
            "diagnostic_prognostic_cols_application": ["lpop", "time"],
            "diagnostic_treatment_cols_application": ["lpop", "k_diag"],
        },
        "rfx": {
            "simulation_effect": "none",
            "application_structured": "none",
            "application_structured_rfx_unit": "unit",
            "diagnostic": "unit",
        },
        "seed_schedule": {
            "chain_seed": "1000 * rep + chain_index",
            "chain_index_start": 0,
            "note": "unchanged from didbcf_structured/model.py; only the "
                    "accepted budget, not the seed map, may change",
        },
        "raw_estimator": True,
        "posterior_correction": "none",
    }


# --------------------------------------------------------------------------- #
# Sampling budgets (candidate ladder; the accepted production budget is frozen
# in a new manifest after the E2 pilot, never selected on truth or coverage).
# --------------------------------------------------------------------------- #
def budget(num_gfr: int, num_burnin: int, num_mcmc: int, keep_every: int,
           num_chains: int = 3, **extra) -> dict:
    return {"num_gfr": int(num_gfr), "num_burnin": int(num_burnin),
            "num_mcmc": int(num_mcmc), "keep_every": int(keep_every),
            "num_chains": int(num_chains), **extra}


BUDGETS = {
    "B0_historical": budget(50, 0, 500, 5, 3,
                            label="archived budget; analyse saved output only"),
    "B1_warm2500_ret1000": budget(50, 2500, 1000, 5, 4,
                                  label="existing pilot budget, four chains"),
    "B2_warm5000_ret1000": budget(50, 5000, 1000, 5, 4,
                                  label="independently longer warmup"),
    "B3_warm2500_ret2000": budget(50, 2500, 2000, 5, 4,
                                  label="independently more retained draws"),
    "B4_warm10000_ret2000": budget(50, 10000, 2000, 5, 4,
                                   label="gated candidate; run only after "
                                         "reviewing B1/B2/B3"),
    "B1_3chain_local_recovery": budget(50, 2500, 1000, 5, 3,
                                       label="three-chain form used by the "
                                             "local large-N pilot retry"),
}

PILOT_BUDGET_KEYS = ("B1_warm2500_ret1000", "B2_warm5000_ret1000",
                     "B3_warm2500_ret2000")
GATED_BUDGET_KEYS = ("B4_warm10000_ret2000",)

# Acceptance targets frozen in pilot_protocol.md (precision targets, not
# mathematical convergence certificates).
ACCEPTANCE = {
    "rhat_max": 1.01,
    "ess_min": 400.0,
    "mean_mcse_over_sd_max": 0.02,
    "endpoint_mcse_over_width_max": 0.01,
    "stability_mean_tolerance": 0.05,
    "stability_endpoint_tolerance": 0.02,
}


def sweeps_per_chain(bdg: dict) -> int:
    return int(bdg["num_gfr"]) + int(bdg["num_burnin"]) + int(
        bdg["num_mcmc"]) * int(bdg["keep_every"])


def budget_key(manifest: dict, key: str) -> dict:
    if "budgets" in manifest:
        return manifest["budgets"][key]
    return BUDGETS[key]


# --------------------------------------------------------------------------- #
# Frozen source bundle
# --------------------------------------------------------------------------- #
def bundle_paths() -> list:
    """Files embedded in every notebook bundle, repo-relative layout."""
    paths = []
    for folder in (PROJECT / "didbcf_structured", SIM / "did_bcf_revision",
                   TOOLS):
        paths.extend(sorted(p for p in folder.rglob("*.py")
                            if "__pycache__" not in p.parts))
    paths.extend([
        SIM / "scripts" / "run_pretrend_empirical.py",
        SIM / "scripts" / "run_empirical_structured.py",
        CAMPAIGN / "campaign.py",
        MPDTA, ORIGINAL_MANIFEST,
    ])
    engine = RECOVERY / "engine"
    paths.extend(p for p in engine.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.append(RECOVERY / "pilot" / "pilot_protocol.md")
    paths.append(INVENTORY / "job_seconds.csv")
    return sorted({p.resolve() for p in paths})


def bundle_relpaths() -> list:
    return [str(p.relative_to(REPO)) for p in bundle_paths()]


def bundle_hashes() -> dict:
    return {rel: sha256_file(REPO / rel) for rel in bundle_relpaths()}


def pin_dependencies() -> dict:
    return dict(PINNED_PYTHON)


# --------------------------------------------------------------------------- #
# Atomic / safe IO
# --------------------------------------------------------------------------- #
def write_bytes_atomic(path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json_atomic(path, obj) -> None:
    write_bytes_atomic(path, json.dumps(obj, indent=2, allow_nan=False).encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def zip_member_safe(name: str) -> bool:
    """Reject absolute paths, parent traversal and drives."""
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    parts = Path(name.replace("\\", "/")).parts
    if any(part == ".." for part in parts):
        return False
    if any(part.endswith(":") for part in parts):
        return False
    return True


def extract_zip_verified(zip_path, target, verified: bool = False) -> dict:
    """Extract a results zip safely.

    With ``verified=True`` existing files whose bytes match are skipped and
    existing files whose bytes differ raise; this is the resume path.  Without
    it, any pre-existing file raises.  Returns a small extraction summary.
    """
    import zipfile
    from pathlib import PurePosixPath

    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate paths inside the zip")
        for info in archive.infolist():
            if info.is_dir():
                continue
            if not zip_member_safe(info.filename):
                raise ValueError(f"Unsafe zip member: {info.filename}")
            member = PurePosixPath(info.filename)
            destination = (target / member).resolve()
            if not str(destination).startswith(str(target) + os.sep) and destination != target:
                raise ValueError(f"Zip member escapes the target: {member}")
            payload = archive.read(info)
            if destination.exists():
                if not verified:
                    raise FileExistsError(
                        f"Refusing to overwrite {destination} on a clean import")
                if sha256_file(destination) != sha256_bytes(payload):
                    raise ValueError(f"Conflicting existing file {destination}")
                skipped.append(info.filename)
                continue
            write_bytes_atomic(destination, payload)
            written.append(info.filename)
    return {"written": written, "skipped_matching": skipped}


def make_single_zip(stem, directory) -> str:
    """Create one archive for a whole output directory (browser download cap)."""
    return shutil.make_archive(str(stem), "zip", str(directory))


# --------------------------------------------------------------------------- #
# Resource inspection (runtime CPU/RAM/disk, never assume Colab's spec sheet)
# --------------------------------------------------------------------------- #
def resource_report() -> dict:
    report = {}
    try:
        report["cpu_affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        report["cpu_affinity"] = os.cpu_count()
    report["cpu_count"] = os.cpu_count()
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            meminfo[key.strip()] = value.strip()
        report["mem_total_kib"] = meminfo.get("MemTotal")
        report["mem_available_kib"] = meminfo.get("MemAvailable")
    except OSError:
        pass
    usage = shutil.disk_usage(str(Path.cwd()))
    report["disk_total_gib"] = round(usage.total / 2 ** 30, 2)
    report["disk_free_gib"] = round(usage.free / 2 ** 30, 2)
    report.update(runtime_versions())
    report["dependencies"] = environment()
    return report


def set_single_thread_environment() -> None:
    """Must run before numpy/stochtree import in the worker process."""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/didbcf-mpl")


# --------------------------------------------------------------------------- #
# Run-unit identity
# --------------------------------------------------------------------------- #
def run_id_for(stat_job_id: str, bdg: dict, chain_index, trace: bool,
               parent_hash: str = PARENT_MANIFEST) -> str:
    return digest({
        "stat_job_id": stat_job_id,
        "budget": {k: bdg[k] for k in
                   ("num_gfr", "num_burnin", "num_mcmc", "keep_every",
                    "num_chains")},
        "chain_index": chain_index,
        "trace": bool(trace),
        "parent_manifest": parent_hash,
        "recovery_version": RECOVERY_VERSION,
        "engine": specification(),
    })[:16]


def assignment_hash(assignment: dict) -> str:
    body = {k: v for k, v in assignment.items() if k != "assignment_hash"}
    return digest(body)


def verify_assignment(assignment: dict) -> dict:
    body = {k: v for k, v in assignment.items() if k != "assignment_hash"}
    if digest(body) != assignment.get("assignment_hash"):
        raise ValueError("Assignment file was altered")
    return assignment
