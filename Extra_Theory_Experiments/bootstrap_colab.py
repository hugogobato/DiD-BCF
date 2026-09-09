"""Small bootstrap helper used by Colab notebooks and local smoke runs."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/hugogobato/DiD-BCF.git"
BRANCH = "experiments/theory-calibration-colab"


def clone_and_install(target: str = "DiD-BCF") -> Path:
    target_path = Path(target).resolve()
    if not (target_path / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        REPO_URL, str(target_path)], check=True)
    else:
        # The target is a disposable clone. Refresh tracked source files on
        # reruns while leaving untracked result checkpoints intact.
        subprocess.run(["git", "-C", str(target_path), "fetch", "--depth", "1",
                        "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(target_path), "reset", "--hard",
                        f"origin/{BRANCH}"], check=True)
    extra = target_path / "Extra_Theory_Experiments"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    str(extra / "requirements-colab.txt")], check=True)
    sys.path.insert(0, str(extra / "src"))
    sys.path.insert(0, str(target_path))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    return target_path


if __name__ == "__main__":
    print("Repository ready:", clone_and_install())
