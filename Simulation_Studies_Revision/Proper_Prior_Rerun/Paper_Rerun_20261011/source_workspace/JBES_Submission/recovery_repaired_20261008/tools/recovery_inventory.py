#!/usr/bin/env python3
"""E0 inventory: read-only verification of returned archives and pilot state.

Reproduces the old collector's identity/hash checks (``collect_proper_reruns.py``
conventions) without editing any returned file, and records every artifact's
disposition: valid, absent, partial, conflicting, duplicate or smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from recovery_common import (CAMPAIGN, INVENTORY, ORIGINAL_MANIFEST,
                             ORIGINAL_MANIFEST_HASH, ORIGINAL_PINS, PROJECT,
                             SIM, digest, sha256_file, write_json_atomic)

RERUNS = SIM / "Reruns" / "proper_ig_v1"
LOCAL_PILOT = RERUNS / "convergence_pilot_20260916"


def load_original_manifest():
    manifest = json.loads(ORIGINAL_MANIFEST.read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(body) != manifest["manifest_hash"]:
        raise ValueError("Original manifest hash mismatch")
    if manifest["manifest_hash"] != ORIGINAL_MANIFEST_HASH:
        raise ValueError("Original manifest identifier changed")
    return manifest


def scan_archive(path: Path, manifest: dict, expected: dict):
    """Verify one returned archive and return per-job records and audit notes."""
    archive = {"archive": path.name, "bytes": path.stat().st_size,
               "jobs": [], "partial": [], "smoke": [], "conflicts": [],
               "duplicates": [], "unsafe": [], "bad_zip": False}
    shard = int(path.stem.rsplit("_", 1)[1])
    try:
        with ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                archive["duplicates"].append("<zip paths>")
            for name in names:
                parts = Path(name.replace("\\", "/")).parts
                if name.startswith("/") or ".." in parts:
                    archive["unsafe"].append(name)
            completed = sorted(n for n in names if n.endswith("/complete.json"))
            started = {n.rsplit("/", 1)[0] for n in names
                       if n.endswith("/started.json")}
            completed_dirs = {n.rsplit("/", 1)[0] for n in completed}
            for orphan in sorted(started - completed_dirs):
                archive["partial"].append(orphan)
            for name in completed:
                record = json.loads(z.read(name))
                identity = record.get("identity", {})
                job = identity.get("job", {})
                jid = job.get("id")
                if jid not in expected:
                    archive["conflicts"].append(
                        dict(name=name, reason="unknown job id"))
                    continue
                if jid in {r["job_id"] for r in archive["jobs"]}:
                    archive["conflicts"].append(
                        dict(name=name, reason="duplicate job in archive"))
                    continue
                problems = []
                if job != expected[jid]:
                    problems.append("job definition mismatch")
                if job.get("shard") != shard:
                    problems.append("shard mismatch")
                if identity.get("manifest_hash") != manifest["manifest_hash"]:
                    problems.append("manifest hash mismatch")
                if identity.get("smoke") is not False:
                    problems.append("smoke or unknown identity")
                if record.get("dependencies") != manifest["dependencies"]:
                    problems.append("dependency mismatch")
                expected_sampler = {
                    **manifest["sampler"],
                    "sigma2_shape": job.get("sigma2_shape"),
                    "sigma2_rate": job.get("sigma2_rate")}
                if record.get("actual_sampler") != expected_sampler:
                    problems.append("sampler mismatch")
                prior = record.get("prior", {})
                global_prior = prior.get("global_prior", {})
                if (prior.get("variance_prior_scale") != "standardized_outcome"
                        or prior.get("effect_by_cohort") is not True
                        or global_prior.get("sigma2_shape")
                        != job.get("sigma2_shape")
                        or global_prior.get("sigma2_rate")
                        != job.get("sigma2_rate")):
                    problems.append("prior mismatch")
                if record.get("actual_units") != job.get("N"):
                    problems.append("unit count mismatch")
                prefix = name.rsplit("/", 1)[0] + "/"
                expected_outputs = {"started.json", "summaries.csv", "draws.npz"}
                if set(record.get("outputs", {})) != expected_outputs:
                    problems.append("output set mismatch")
                payloads = {}
                for member, checksum in record.get("outputs", {}).items():
                    payload = z.read(prefix + member)
                    payloads[member] = payload
                    if hashlib.sha256(payload).hexdigest() != checksum:
                        problems.append(f"checksum {member}")
                if not problems and "draws.npz" in payloads:
                    try:
                        with np.load(io.BytesIO(payloads["draws.npz"]),
                                    allow_pickle=False) as draws:
                            for key in draws.files:
                                x = draws[key]
                                if x.shape != (3, 500) or not np.isfinite(x).all():
                                    problems.append(f"draw shape {key}")
                                    break
                                if key == "sigma2" and not (x > 0).all():
                                    problems.append("nonpositive variance")
                                    break
                    except Exception as error:  # unreadable draws
                        problems.append(f"draws unreadable: {error!r}")
                if problems:
                    archive["conflicts"].append(
                        dict(name=name, job_id=jid, reasons=problems))
                    continue
                archive["jobs"].append(dict(
                    job_id=jid, phase=job["phase"], setting=job["setting"],
                    N=job["N"], linearity_degree=job["linearity_degree"],
                    rep=job["rep"], spec=job["spec"],
                    seconds=record.get("seconds"),
                    peak=None, panel_hash=record.get("panel_hash"),
                    python=record.get("python")))
    except BadZipFile:
        archive["bad_zip"] = True
    return archive


def _scan_smoke() -> list:
    found = []
    for folder in (RERUNS / "smoke", RERUNS / "proper_ig_initial_smoke"):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("complete.json")):
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                found.append(dict(path=str(path), status="unreadable"))
                continue
            found.append(dict(
                path=str(path), job_id=record.get("identity", {}).get("job", {}).get("id"),
                smoke=record.get("identity", {}).get("smoke"),
                manifest_hash=record.get("identity", {}).get("manifest_hash")))
    return found


def _scan_local_pilot() -> dict:
    out = {"pilot": None, "retry": None}
    if not LOCAL_PILOT.exists():
        return out
    for name in ("pilot_plan.json", "pilot_status.json"):
        path = LOCAL_PILOT / name
        if path.exists():
            out["pilot"] = out.get("pilot") or {}
            out["pilot"][name] = json.loads(path.read_text())
    retry = LOCAL_PILOT / "bd2541dd8b654930a2fe_checkpointed_retry"
    if retry.exists():
        chains = {}
        for folder in sorted(retry.glob("chain_*")):
            complete = folder / "complete.json"
            record = json.loads(complete.read_text()) if complete.exists() else None
            chains[folder.name] = {
                "has_complete": complete.exists(),
                "has_draws": (folder / "draws.npz").exists(),
                "chain": record.get("chain") if record else None,
                "seed": record.get("seed") if record else None,
                "seconds": record.get("seconds") if record else None}
        out["retry"] = {
            "folder": str(retry), "chains": chains,
            "has_chain_status": (retry / "chain_status.json").exists(),
            "has_combined_draws": (retry / "draws.npz").exists(),
            "has_complete": (retry / "complete.json").exists()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=INVENTORY)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = load_original_manifest()
    expected = {job["id"]: job for job in manifest["jobs"]}
    archives = []
    records = []
    seen = {}
    conflicts = []
    for path in sorted(CAMPAIGN.glob("didbcf_proper_shard_*.zip")):
        archive = scan_archive(path, manifest, expected)
        archives.append({k: v for k, v in archive.items() if k != "jobs"})
        for record in archive["jobs"]:
            jid = record["job_id"]
            if jid in seen:
                conflicts.append(dict(job_id=jid, reason="duplicate across archives",
                                      archives=[seen[jid], path.name]))
                continue
            seen[jid] = path.name
            records.append(record)
    jobs = pd.DataFrame(records)
    if not jobs.empty:
        jobs["shard"] = [
            int(seen[jid].rsplit("_", 1)[1][:-4]) for jid in jobs.job_id]
        jobs.to_csv(args.out / "job_seconds.csv", index=False)

    by_phase = Counter(j["phase"] for j in manifest["jobs"])
    verified_phase = Counter(jobs.phase) if not jobs.empty else Counter()
    missing = [j for j in manifest["jobs"] if j["id"] not in seen]
    missing_phase = Counter(j["phase"] for j in missing)
    absent_by_phase = dict(missing_phase)
    partial = [p for a in archives for p in a["partial"]]
    unsafe = [p for a in archives for p in a["unsafe"]]
    bad_zip = [a["archive"] for a in archives if a["bad_zip"]]
    archive_conflicts = [dict(archive=a["archive"], **c)
                         for a in archives for c in a["conflicts"]]
    duplicates = [dict(archive=a["archive"], item=d)
                  for a in archives for d in a["duplicates"]]
    smoke = _scan_smoke()
    local = _scan_local_pilot()

    source_drift = {}
    for rel, expected_hash in manifest["source_hashes"].items():
        path = PROJECT / rel
        if not path.exists():
            source_drift[rel] = "missing"
        elif sha256_file(path) != expected_hash:
            source_drift[rel] = "changed"
    if source_drift:
        raise SystemExit(f"Source drift detected: {source_drift}")

    seconds = {}
    if not jobs.empty and jobs.seconds.notna().any():
        valid = jobs[jobs.seconds.notna()]
        per_chain = valid.seconds / manifest["sampler"]["num_chains"]
        seconds = dict(n=len(valid), min=float(valid.seconds.min()),
                       median=float(valid.seconds.median()),
                       max=float(valid.seconds.max()),
                       per_chain_median=float(per_chain.median()))
    inventory = {
        "scope": "E0 read-only inventory, no fits",
        "original_manifest_hash": manifest["manifest_hash"],
        "original_manifest_file_sha256": sha256_file(ORIGINAL_MANIFEST),
        "original_pins": manifest["dependencies"],
        "original_python": manifest["python"],
        "verified_pins": ORIGINAL_PINS,
        "source_drift": source_drift,
        "phases": {"prepared": dict(by_phase),
                   "verified_main": dict(verified_phase),
                   "absent": absent_by_phase},
        "archives": archives,
        "valid_jobs": len(seen),
        "absent_jobs": len(missing),
        "partial_jobs": partial,
        "conflicting": archive_conflicts + conflicts,
        "duplicates": duplicates,
        "unsafe_paths": unsafe,
        "bad_zips": bad_zip,
        "smoke": smoke,
        "local_pilot": local,
        "measured_seconds": seconds,
        "disposition": {
            "valid": f"{len(seen)} main jobs verified against the manifest",
            "absent": f"{len(missing)} prepared jobs never returned "
                      f"({dict(missing_phase)})",
            "partial": f"{len(partial)} started-without-completion packages",
            "conflicting": f"{len(archive_conflicts) + len(conflicts)} findings",
            "duplicate": f"{len(duplicates)} archive-level duplicate findings",
            "smoke": f"{len(smoke)} smoke records kept separate",
        },
        "note": ("Completeness of returned main jobs is not convergence. "
                 "Returned main results remain provisional; the missing phases "
                 "are not scheduled under the questionable historical budget."),
    }
    write_json_atomic(args.out / "recovery_inventory.json", inventory)

    report = [
        "# E0 recovery inventory (read-only)",
        "",
        f"Original manifest `{manifest['manifest_hash']}` verified, source "
        "hashes unchanged, dependencies as pinned.",
        "",
        "## Dispositions",
        "",
        f"1. valid: {len(seen)} main jobs verified across {len(archives)} archives;",
        f"2. absent: {len(missing)} jobs ({dict(missing_phase)});",
        f"3. partial: {len(partial)} started-without-completion packages;",
        f"4. conflicting: {len(archive_conflicts) + len(conflicts)};",
        f"5. duplicate: {len(duplicates)};",
        f"6. smoke: {len(smoke)} records, kept separate.",
        "",
        "## Measured main-phase runtime (old budget)",
        "",
        f"`{json.dumps(seconds)}`",
        "",
        "## Local large-N retry state",
        "",
        f"```json\n{json.dumps(local.get('retry'), indent=2)}\n```",
        "",
        "The retry is not restartable in place; the recovery notebooks use a "
        "new attempt/driver path with integrated completed-chain verification.",
        "",
        "## Source hash check",
        "",
        "All original source/data hashes match the manifest. The application "
        f"data pin is `{manifest['source_hashes'].get('Empirical_Study/mpdta.csv')}`.",
    ]
    (args.out / "RECOVERY_INVENTORY_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({k: v for k, v in inventory.items()
                      if k not in ("archives", "smoke", "local_pilot")},
                     indent=2, default=str))
    print(f"Wrote {args.out/'recovery_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
