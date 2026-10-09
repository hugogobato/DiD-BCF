#!/usr/bin/env python3
"""Isolated, hash-pinned source builds for DiD-BCF sampler validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "build_manifest.json"

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def fetch(spec, cache):
    path = Path(cache) / spec["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != spec["sha256"]:
            raise ValueError("Existing archive has wrong hash: " + str(path))
        return path
    temporary = path.with_suffix(".download")
    errors = []
    for url in spec["urls"]:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "didbcf-repair-validation/1"})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if sha256(temporary) != spec["sha256"]:
                raise ValueError("Archive digest differs from the predeclared pin")
            temporary.replace(path)
            return path
        except Exception as error:
            errors.append(url + ": " + repr(error))
            temporary.unlink(missing_ok=True)
    raise RuntimeError("Pinned download failed; no revision substitution allowed.\n" + "\n".join(errors))

def extract_archive(archive, destination):
    """Strip one archive root; accept only regular files/directories."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        roots = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
        if len(roots) != 1:
            raise ValueError("Archive must have one root directory")
        for member in members:
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                raise ValueError("Unsafe archive path " + member.name)
            if len(parts) <= 1:
                continue
            path = destination.joinpath(*parts[1:])
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as inp, path.open("wb") as out:
                    shutil.copyfileobj(inp, out)
                path.chmod(member.mode & 0o777)
            else:
                raise ValueError("Unsupported archive link/device " + member.name)

def run_logged(command, log, cwd=None, env=None, timeout=1800):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        output.write(json.dumps(command) + "\n")
        output.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)

def source_hashes(source, names):
    return {name: sha256(source / name) for name in names}

def build_variant(variant, work, cache, jobs=1):
    if variant not in ("original", "repaired"):
        raise ValueError(variant)
    if jobs not in (1, 2):
        raise ValueError("Compile workers are capped at two")
    manifest = json.loads(MANIFEST.read_text())
    patch = ROOT / manifest["patch"]["path"]
    if sha256(patch) != manifest["patch"]["sha256"]:
        raise ValueError("Patch differs from build manifest")
    identity = hashlib.sha256((sha256(MANIFEST) + variant).encode()).hexdigest()
    home = Path(work) / variant
    completion = home / "build_complete.json"
    if completion.exists():
        existing = json.loads(completion.read_text())
        if existing["identity"] != identity:
            raise ValueError("Existing build belongs to a different manifest")
        for rel, digest in existing["site_files"].items():
            if sha256(home / "site" / rel) != digest:
                raise ValueError("Existing compiled artifact changed: " + rel)
        return existing
    # Failed attempts have their logs retained; start a fresh source directory.
    if (home / "src").exists():
        raise RuntimeError("Incomplete build exists at " + str(home) +
                           "; use a new WORK path to preserve failed-build evidence")
    home.mkdir(parents=True, exist_ok=True)
    archives = {key: fetch(spec, cache) for key, spec in manifest["archives"].items()}
    source = home / "src"
    extract_archive(archives["stochtree"], source)
    for key in manifest["dependencies"]:
        extract_archive(archives[key], source / "deps" / key)
    original = source_hashes(source, manifest["changed_sources"])
    if original != manifest["original_source_hashes"]:
        raise ValueError("Original source does not match frozen audited files")
    if variant == "repaired":
        run_logged(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch.resolve())],
                   home / "patch.log", cwd=source)
    patched = source_hashes(source, manifest["changed_sources"])
    expected = manifest["repaired_source_hashes"] if variant == "repaired" else manifest["original_source_hashes"]
    if patched != expected:
        raise ValueError("Applied source has unexpected digest")
    site = home / "site"
    site.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    configure = ["cmake", "-S", str(source.resolve()), "-B", str((home / "cmake").resolve()),
                 "-DBUILD_PYTHON=ON", "-DBUILD_TEST=OFF", "-DBUILD_DEBUG_TARGETS=OFF",
                 "-DUSE_OPENMP=ON", "-DUSE_DEBUG=OFF", "-DCMAKE_BUILD_TYPE=Release",
                 "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DEXAMPLE_VERSION_INFO=0.4.4",
                 "-DPYTHON_EXECUTABLE=" + sys.executable,
                 "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(site.resolve())]
    run_logged(configure, home / "configure.log", env=env)
    compile_command = ["cmake", "--build", str((home / "cmake").resolve()), "--parallel", str(jobs)]
    run_logged(compile_command, home / "compile.log", env=env)
    candidates = sorted(site.glob("stochtree_cpp*.so"))
    if not candidates:
        candidates = sorted((source / "build").glob("stochtree_cpp*.so"))
    if len(candidates) != 1:
        raise RuntimeError("Expected one compiled Python extension: " + repr(candidates))
    if candidates[0].parent != site:
        shutil.copy2(candidates[0], site / candidates[0].name)
    shutil.copytree(source / "stochtree", site / "stochtree", dirs_exist_ok=True)
    metadata = site / "stochtree-0.4.4.dist-info"
    metadata.mkdir(exist_ok=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: stochtree\nVersion: 0.4.4\n"
        "Summary: isolated source build for DiD-BCF validation\n")
    files = {str(p.relative_to(site)): sha256(p) for p in site.rglob("*") if p.is_file()}
    record = {"variant": variant, "identity": identity, "build_manifest_sha256": sha256(MANIFEST),
              "patch_sha256": sha256(patch) if variant == "repaired" else None,
              "source_hashes": patched, "site_files": files,
              "configure": configure, "compile": compile_command,
              "python": sys.version, "executable": sys.executable,
              "compile_jobs": jobs, "archives": {k: v["sha256"] for k,v in manifest["archives"].items()},
              "scope": "isolated source build; original installed/historical binary is not replaced"}
    atomic_json(completion, record)
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["original", "repaired"], required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    result = build_variant(args.variant, args.work, args.cache, args.jobs)
    print(json.dumps({"variant":result["variant"], "identity":result["identity"],
                      "compile_jobs":result["compile_jobs"]}, indent=2))
if __name__ == "__main__":
    main()
