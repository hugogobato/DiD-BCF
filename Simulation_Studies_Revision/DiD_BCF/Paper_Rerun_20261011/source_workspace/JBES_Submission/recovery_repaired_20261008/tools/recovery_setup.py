"""Build one isolated repaired engine and validate the actual loaded binary."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from recovery_engine import ENGINE, activate

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--existing-build", type=Path)
    args = p.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/didbcf-repaired-mpl")
    args.work.mkdir(parents=True, exist_ok=True)
    home = args.existing_build.resolve() if args.existing_build else args.work.resolve() / "builds" / "repaired"
    if not args.existing_build:
        subprocess.run([sys.executable, str(ENGINE / "tools/build_native.py"),
                        "--variant", "repaired", "--work", str(args.work / "builds"),
                        "--cache", str(args.cache), "--jobs", "1"], check=True)
    os.environ["DIDBCF_REPAIRED_BUILD"] = str(home)
    identity = activate(require_controls=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(home / "site") + os.pathsep + str(Path(__file__).parent)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    controls = args.work.resolve() / "repaired_controls.json"
    with (args.work / "native_controls.log").open("w") as log:
        subprocess.run([sys.executable, str(ENGINE / "controls.py"), "--native",
                        "--variant", "repaired", "--output", str(controls)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    os.environ["DIDBCF_REPAIRED_CONTROLS"] = str(controls)
    identity = activate(require_weights=False)
    weight_path = args.work.resolve() / "weight_controls.json"
    subprocess.run([sys.executable, str(Path(__file__).with_name("recovery_weight_controls.py")),
                    "--site", str(home / "site"), "--expected-binary-sha256", identity["extension_sha256"], "--out", str(weight_path)], env={**env, **{key: os.environ[key] for key in
                    ("DIDBCF_REPAIRED_BUILD", "DIDBCF_REPAIRED_CONTROLS")}}, timeout=300, check=True)
    weight = json.loads(weight_path.read_text())
    if weight.get("status") != "PASS":
        raise ValueError("Fixed eligible-feature controls failed")
    os.environ["DIDBCF_WEIGHT_CONTROLS"] = str(weight_path)
    identity = activate()
    record = {"status": "PASS", "build": str(home), "controls": str(controls),
              "weight_controls": str(weight_path), "engine": identity}
    (args.work / "setup_complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "extension_sha256": identity["extension_sha256"]}))

if __name__ == "__main__":
    main()
