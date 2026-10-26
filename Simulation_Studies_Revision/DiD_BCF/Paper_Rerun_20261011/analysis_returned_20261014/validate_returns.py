"""Audit returned compact archives without sampling or changing frozen inputs."""
from pathlib import Path
import io
import json
import sys
import zipfile

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE / "helpers"))
from common import canonical, sha, verify_package


def main():
    import pandas as pd
    package, manifest = verify_package()
    expected = {p.stem: json.loads(p.read_text())
                for p in (PACKAGE / "assignments").glob("*.json")}
    rows, runs, ids, parsed_ids = [], [], set(), []
    seen = set()
    for path in sorted((PACKAGE / "Colab").glob("didbcf_paper_*.zip")):
        with zipfile.ZipFile(path) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            load = lambda name: json.loads(archive.read(name))
            assigned = load("bundle/assignment.json")
            assert canonical(assigned) == canonical(expected[assigned["name"]])
            assert assigned["name"] not in seen
            seen.add(assigned["name"])
            assert canonical(load("bundle/manifest.json")) == canonical(manifest)
            assert canonical(load("bundle/package.json")) == canonical(package)
            setup, status = load("bundle/setup_provenance.json"), load("bundle/status.json")
            assert setup["source_git_commit"] == "3078e2ca1155b932feab92c722d697803370b8c7"
            assert setup["native_controls"] == setup["eligible_feature_controls"] == "PASS"
            assert setup["dependencies"] == manifest["dependencies"]
            assert setup["engine"]["specification"] == manifest["engine_specification"]
            assert not status["failed"] and not status["missing_job_ids"]
            completed = {n.split("/")[1] for n in archive.namelist()
                         if n.startswith("fits/") and n.endswith("/complete.json")}
            assert completed == set(assigned["job_ids"])
            assert not ids.intersection(completed)
            ids.update(completed)
            assert status["completed"] == status["expected"] == len(completed)
            assert status["assignment_hash"] == assigned["assignment_hash"]
            assert status["manifest_hash"] == manifest["manifest_hash"]
            for job_id in completed:
                provenance = load(f"fits/{job_id}/chain_provenance.json")
                assert provenance["package_hash"] == package["package_hash"]
                assert provenance["assignment_hash"] == assigned["assignment_hash"]
                assert provenance["engine"] == setup["engine"]
                assert provenance["local_arrays_exported"] is False
                for chain in provenance["chains"]:
                    record = chain.get("record", chain)
                    assert record["engine"] == setup["engine"]
                    runs.append({"job_id": job_id, "archive": path.name,
                                 "chain_seed": record["chain_seed"],
                                 "kind": record["kind"], "phase": record["phase"],
                                 "seconds": record.get("seconds"),
                                 "python": record["runtime"]["python"]})
                if all(c in "0123456789e" for c in job_id):
                    frame = pd.read_csv(io.BytesIO(archive.read(f"fits/{job_id}/per_replication.csv")),
                                        dtype={"job_id": str}, float_precision="round_trip")
                    assert set(frame.job_id) == {job_id}
                    parsed_ids.append(job_id)
            rows.append({"archive": path.name, "sha256": sha(path),
                         "assignment": assigned["name"], "bytes": path.stat().st_size,
                         "fits": len(completed), "elapsed_hours": status["elapsed_hours"],
                         "workers": status["workers"], "python": setup["engine"]["python"],
                         "native_controls": setup["native_controls"],
                         "eligible_feature_controls": setup["eligible_feature_controls"]})
    assert seen == set(expected)
    reused = set(manifest["reused_main_job_ids"])
    assert not ids.intersection(reused)
    assert ids | reused == {j["id"] for j in manifest["jobs"]}
    pd.DataFrame(rows).to_csv(HERE / "archive_inventory.csv", index=False)
    pd.DataFrame(runs).to_csv(HERE / "returned_chain_inventory.csv", index=False)
    summary = {"status": "PASS", "archives": len(rows), "returned_fits": len(ids),
               "verified_reused_fits": len(reused), "total_fits": len(ids | reused),
               "returned_chain_or_atomic_records": len(runs),
               "all_assignments_complete": True, "all_native_and_feature_controls": "PASS",
               "source_commit": setup["source_git_commit"],
               "package_hash": package["package_hash"], "manifest_hash": manifest["manifest_hash"],
               "numeric_looking_ids_preserved_as_text": sorted(parsed_ids),
               "production_sampling_calls": 0}
    (HERE / "RETURN_VALIDATION.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
