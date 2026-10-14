"""Verify and activate an isolated repaired source build, never a global wheel."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys

PACKAGE = Path(__file__).resolve().parents[1]
ENGINE = PACKAGE / "engine"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def specification():
    manifest = json.loads((ENGINE / "build_manifest.json").read_text())
    return {
        "variant": "repaired", "build_manifest_sha256": sha(ENGINE / "build_manifest.json"),
        "patch_sha256": manifest["patch"]["sha256"],
        "source_hashes": manifest["repaired_source_hashes"],
        "archives": {k: v["sha256"] for k, v in manifest["archives"].items()},
        "target_prior": manifest["target_prior"],
    }

def activate(require_controls=True, require_weights=True):
    home = os.environ.get("DIDBCF_REPAIRED_BUILD")
    if not home:
        raise RuntimeError("DIDBCF_REPAIRED_BUILD must name the isolated repaired build; global stochtree is refused")
    home = Path(home).resolve()
    record = json.loads((home / "build_complete.json").read_text())
    spec = specification()
    for key in ("variant", "build_manifest_sha256", "patch_sha256", "source_hashes", "archives"):
        if record.get(key) != spec[key]:
            raise ValueError("Engine provenance mismatch: " + key)
    if record["python"].split()[0] != platform.python_version():
        raise ValueError("Isolated build Python version differs; rebuild in this interpreter")
    site = home / "site"
    for rel, expected in record["site_files"].items():
        path = site / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Engine file mismatch: " + rel)
    for name in ("stochtree", "stochtree_cpp"):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(site):
            raise RuntimeError("A different engine is already imported; restart this Python process")
    sys.path.insert(0, str(site))
    for name in ("stochtree", "stochtree_cpp"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(site):
            raise RuntimeError("Engine import escaped the isolated site: " + name)
    bcf_expected = json.loads((ENGINE / "bcf_defaults.json").read_text())["bcf_python_sha256"]
    if sha(site / "stochtree/bcf.py") != bcf_expected:
        raise ValueError("Diagnostic Python API differs from frozen defaults")
    extension = Path(sys.modules["stochtree_cpp"].__file__).resolve()
    result = {"specification": spec, "build_identity": record["identity"],
              "extension_sha256": sha(extension), "site_files": record["site_files"],
              "python": platform.python_version(), "platform": platform.platform(),
              "build_record_sha256": sha(home / "build_complete.json")}
    if require_controls:
        controls = Path(os.environ.get("DIDBCF_REPAIRED_CONTROLS", str(home / "repaired_controls.json")))
        data = json.loads(controls.read_text())
        probes = data.get("native_binary_probes", {})
        loaded = probes.get("loaded_native_binary", {})
        if probes.get("native_scope_gate") != "PASS":
            raise ValueError("Repaired native controls did not pass")
        # The native runner publishes the exact extension digest, not merely version 0.4.4.
        recorded = loaded.get("extension_sha256", loaded.get("sha256"))
        if recorded != result["extension_sha256"]:
            raise ValueError("Control binary differs from the active repaired extension")
        result["controls_sha256"] = sha(controls)
        if require_weights:
            weights = Path(os.environ.get("DIDBCF_WEIGHT_CONTROLS", str(home / "weight_controls.json")))
            data = json.loads(weights.read_text())
            if data.get("status") != "PASS" or data.get("loaded_engine", {}).get("binary_sha256") != result["extension_sha256"]:
                raise ValueError("Eligible-feature controls did not pass for this binary")
            result["weight_controls_sha256"] = sha(weights)
    return result

def bootstrap():
    """Activate before model imports if execution, while allowing read-only planning."""
    if os.environ.get("DIDBCF_REPAIRED_BUILD"):
        return activate()
    return None


def fingerprint(identity):
    """Stable numerical identity; archive attestations retain paths and controls."""
    files_hash = hashlib.sha256(json.dumps(identity["site_files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {key: identity[key] for key in ("specification", "build_identity", "extension_sha256", "python")} | {"site_files_sha256": files_hash}
