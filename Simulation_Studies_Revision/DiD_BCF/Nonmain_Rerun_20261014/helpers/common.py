"""Identity and storage rules for the metrics-only paper campaign."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE / "source_workspace"
TOOLS = WORKSPACE / "JBES_Submission/recovery_repaired_20261008/tools"
BUDGET_KEY = "earlier_budget_repaired"
_JOB_INDEX = {}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def check_identity(value, field):
    if digest({k:v for k,v in value.items() if k != field}) != value[field]:
        raise ValueError("Altered " + field)
    return value

def bootstrap(setup=None):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/didbcf-paper-mpl")
    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(WORKSPACE / "DiD-BCF"))
    sys.path.insert(0, str(WORKSPACE / "DiD-BCF/Simulation_Studies_Revision"))
    sys.path.insert(0, str(WORKSPACE / "DiD-BCF/Simulation_Studies_Revision/scripts"))
    if setup:
        record = read(setup)
        for variable, field in (("DIDBCF_REPAIRED_BUILD", "build"),
                                ("DIDBCF_REPAIRED_CONTROLS", "controls"),
                                ("DIDBCF_WEIGHT_CONTROLS", "weight_controls")):
            os.environ[variable] = record[field]

def verify_package():
    package = check_identity(read(PACKAGE / "package.json"), "package_hash")
    manifest = check_identity(read(PACKAGE / "manifest.json"), "manifest_hash")
    if package["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Package/statistical identity mismatch")
    for relative, expected in manifest["source_hashes"].items():
        if sha(WORKSPACE / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    for relative, expected in package["files"].items():
        if sha(PACKAGE / relative) != expected:
            raise ValueError("Helper, assignment or reused result changed: " + relative)
    return package, manifest

def assignment(path, manifest):
    value = check_identity(read(path), "assignment_hash")
    if value["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Wrong campaign assignment")
    frozen = {j["id"]: j for j in manifest["jobs"]}
    if len(value["job_ids"]) != len(set(value["job_ids"])):
        raise ValueError("Duplicate assigned fit")
    if any(j not in frozen for j in value["job_ids"]):
        raise ValueError("Unknown assigned fit")
    official = PACKAGE / "assignments" / (value["name"] + ".json")
    if not official.is_file() or read(official)["assignment_hash"] != value["assignment_hash"]:
        raise ValueError("Assignment is not part of the sealed campaign")
    return value

def verify_fit(folder, manifest):
    """Validate an atomic compact result, including every exported checksum."""
    folder = Path(folder)
    if not (folder / "complete.json").exists():
        return None
    record = check_identity(read(folder / "complete.json"), "record_hash")
    if record["manifest_hash"] != manifest["manifest_hash"] or record["budget_key"] != BUDGET_KEY:
        raise ValueError("Wrong campaign or budget in " + str(folder))
    key = manifest["manifest_hash"]
    if key not in _JOB_INDEX:
        _JOB_INDEX[key] = {j["id"]:j for j in manifest["jobs"]}
    job = _JOB_INDEX[key].get(record["job_id"])
    if job is None or digest(job) != record["job_hash"]:
        raise ValueError("Wrong job in " + str(folder))
    if record["origin"] not in ("new_repaired_fit", "reused_verified_earlier_fit"):
        raise ValueError("Unknown result origin")
    if record["origin"] == "reused_verified_earlier_fit":
        registry = manifest["reuse_registry"].get(record["job_id"])
        if registry is None or record.get("reused_from_record_hash") != registry["record_hash"]:
            raise ValueError("Unregistered reused fit")
        if any(record["outputs"].get(name) != checksum for name,checksum in registry["outputs"].items()):
            raise ValueError("Reused output bytes differ from verified parent fit")
    expected_seeds = ([None] if job["kind"] in ("pretrend", "application_pretrend")
                      else [1000 * job["rep"] + i for i in range(3)])
    if record["chain_seeds"] != expected_seeds:
        raise ValueError("Incomplete or wrong chain seeds")
    if record["engine_specification"] != manifest["engine_specification"]:
        raise ValueError("Wrong repaired engine specification")
    allowed = {"per_replication.csv", "diagnostics.csv", "chain_provenance.json",
               "application_gate.csv", "application_catt.csv", "application_contrasts.csv"}
    if not {"per_replication.csv", "diagnostics.csv", "chain_provenance.json"}.issubset(record["outputs"]):
        raise ValueError("Missing required compact outputs")
    if set(record["outputs"]) - allowed:
        raise ValueError("Unexpected output or posterior array in export")
    for name, expected in record["outputs"].items():
        if sha(folder / name) != expected:
            raise ValueError("Corrupt compact output: " + name)
    provenance = read(folder / "chain_provenance.json")
    chains = [entry.get("record", entry) for entry in provenance["chains"]]
    if len(chains) != len(expected_seeds):
        raise ValueError("Incomplete chain provenance")
    atomic_job = job["kind"] in ("pretrend", "application_pretrend")
    indexes = [None] if atomic_job else [0,1,2]
    budget = manifest["budgets"][BUDGET_KEY]
    expected_source_digest = (manifest["reuse_source_hashes_sha256"]
                              if record["origin"] == "reused_verified_earlier_fit"
                              else digest(manifest["source_hashes"]))
    for chain, index, seed in zip(chains,indexes,expected_seeds):
        if chain.get("smoke") or "smoke" in str(chain.get("purpose", "")):
            raise ValueError("Smoke artifact cannot enter paper evidence")
        for field, value in {"stat_job_id":job["id"],"chain_index":index,"chain_seed":seed,
                             "budget_key":BUDGET_KEY,"budget":budget,
                             "actual_sampler":{**budget,"sigma2_shape":job["sigma2_shape"],"sigma2_rate":job["sigma2_rate"]},
                             "parent_manifest":manifest["parent_manifest"],
                             "source_hashes_sha256":expected_source_digest,
                             "dependencies":manifest["dependencies"],
                             **{k:job[k] for k in ("phase","kind","setting","rep","spec")}}.items():
            if chain.get(field) != value:
                raise ValueError("Chain provenance mismatch: " + field)
        if chain["engine"]["specification"] != manifest["engine_specification"]:
            raise ValueError("Wrong chain engine")
        prior = chain["prior"].get("global_prior",chain["prior"])
        if any(prior.get(k) != job[k] for k in ("sigma2_shape","sigma2_rate")):
            raise ValueError("Wrong chain variance prior")
        expected_rfx = "unit" if atomic_job or job["spec"].endswith("unit") else "none"
        if chain["prior"].get("rfx") != expected_rfx:
            raise ValueError("Wrong random-effect specification")
        if not atomic_job:
            if chain["prior"].get("effect_by_cohort") is not True:
                raise ValueError("Wrong cohort specification")
            if any(chain["prior"].get(k+"_prior") != manifest["model_spec"]["forest_priors"][k]
                   for k in ("level","trend","effect")):
                raise ValueError("Wrong forest priors")
    if len({canonical(c["engine"]) for c in chains}) != 1 or len({c["panel_hash"] for c in chains}) != 1:
        raise ValueError("Chains disagree on panel or native engine")
    return record

def restore_metrics(zip_path, out, manifest, assigned):
    """Only complete fits from this assignment; no native state continuation."""
    out = Path(out).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        saved = read_zip_json(archive, "bundle/assignment.json")
        if saved["assignment_hash"] != assigned["assignment_hash"]:
            raise ValueError("Resume ZIP belongs to a different notebook")
        for entry in archive.infolist():
            name = entry.filename
            target = (out / name).resolve()
            if (not target.is_relative_to(out) or ".." in Path(name).parts
                    or name.startswith("/") or ((entry.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe archive member")
            if entry.is_dir():
                continue
            if not (name.startswith("fits/") or name.startswith("bundle/")):
                raise ValueError("Unexpected resume member")
            if target.suffix not in (".csv", ".json") or entry.file_size > 16*2**20:
                raise ValueError("Resume requires compact metrics, not posterior arrays")
            data = archive.read(entry)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Resume overwrites different bytes")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for folder in (out / "fits").glob("*"):
        record = verify_fit(folder, manifest)
        if record is None or record["job_id"] not in assigned["job_ids"]:
            raise ValueError("Unassigned/incomplete restored fit")

def read_zip_json(archive, name):
    return json.loads(archive.read(name))
