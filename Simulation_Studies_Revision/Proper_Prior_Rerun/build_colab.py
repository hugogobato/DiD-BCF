#!/usr/bin/env python3
"""Generate independent Colab notebooks with embedded, hash-pinned sources."""
import base64
import io
import json
from pathlib import Path
import zipfile
from campaign import HERE, PROJECT, source_files, verify_manifest


def cell(kind, text):
    out = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def main():
    manifest_path = HERE / "manifest.json"
    manifest = verify_manifest(manifest_path)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files() + [manifest_path]:
            archive.write(path, path.relative_to(PROJECT))
    payload = base64.b64encode(data.getvalue()).decode()
    dependencies = " ".join(f"{key}=={value}" for key, value in manifest["dependencies"].items())
    out = HERE / "Colab"
    out.mkdir(exist_ok=False)
    for shard in range(manifest["shards"]):
        setup = f'''# One worker, 1 CPU thread. No GPU required.
PHASE = "main"  # main,diagnostic,pt_impact,application,sensitivity,application_sensitivity
SHARD = {shard}
HOURS = 8
RESUME_ZIP = None  # optional path to a previously downloaded results zip
import os, sys, subprocess, pathlib, base64, io, zipfile
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/didbcf-mpl")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *{dependencies!r}.split()])
ROOT = pathlib.Path("/content/didbcf_proper_ig")
ROOT.mkdir(parents=True, exist_ok=True)
payload = {payload!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
    archive.extractall(ROOT)  # trusted embedded source bundle
OUT = pathlib.Path("/content/didbcf_proper_results")
OUT.mkdir(exist_ok=True)
if RESUME_ZIP:
    with zipfile.ZipFile(RESUME_ZIP) as archive:
        for item in archive.infolist():
            target = (OUT / item.filename).resolve()
            if not target.is_relative_to(OUT.resolve()):
                raise ValueError("Unsafe zip path")
            if target.exists() and not item.is_dir():
                raise FileExistsError("Resume into an empty output directory")
        archive.extractall(OUT)
'''
        run = '''import shutil
SCRIPT = ROOT / "Simulation_Studies_Revision/Proper_Prior_Rerun/campaign.py"
try:
    subprocess.check_call([sys.executable, str(SCRIPT), "run", "--phase", PHASE,
                           "--shard", str(SHARD), "--hours", str(HOURS), "--out", str(OUT)])
finally:
    # Bundle all generated files: a single download avoids browser limits.
    output_file = shutil.make_archive(f"/content/didbcf_proper_shard_{SHARD:02d}", "zip", OUT)
    try:
        from google.colab import files
        files.download(output_file)
        print("Downloaded:", output_file)
    except Exception as e:
        print("(Not on Colab / download skipped):", e)
        print("Output archive:", output_file)
'''
        notebook = dict(nbformat=4, nbformat_minor=5,
                        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                        cells=[cell("markdown", f"# DiD-BCF proper-prior rerun: shard {shard}/47\n\n"
                                    "Production work, not historical results. Run setup before sampling. "
                                    "Choose one phase; retain downloaded checkpoints. Source/dependency mismatches fail closed. "
                                    "Inspect chain diagnostics before using estimates. A forced Colab disconnect may prevent download.\n"),
                               cell("code", setup), cell("code", run)])
        for index, c in enumerate(notebook["cells"]):
            c["id"] = f"shard-{shard}-cell-{index}"
        (out / f"proper_ig_shard_{shard:02d}.ipynb").write_text(json.dumps(notebook, indent=1))
    with zipfile.ZipFile(HERE / "Colab_notebooks.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.glob("*.ipynb")):
            archive.write(path, path.name)
    print(f"Created {manifest['shards']} self-contained notebooks and Colab_notebooks.zip")


if __name__ == "__main__":
    main()
