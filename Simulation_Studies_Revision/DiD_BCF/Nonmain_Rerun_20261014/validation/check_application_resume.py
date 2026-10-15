"""Check real compact-fit checkpoint restoration without starting sampling."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "helpers"))
from common import assignment, bootstrap, restore_metrics, sha, verify_fit, verify_package
from runtime import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    _, manifest = verify_package()
    assignment_path = PACKAGE / "assignments/local_application.json"
    assigned = assignment(assignment_path, manifest)
    source = PACKAGE / "Results/local_application"
    complete = [job_id for job_id in assigned["job_ids"]
                if verify_fit(source / "fits" / job_id, manifest)]
    if not complete:
        parser.error("Wait for at least one real new application fit")
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    fixture = scratch / "checkpoint_source"
    shutil.copytree(source / "bundle", fixture / "bundle")
    for job_id in complete:
        shutil.copytree(source / "fits" / job_id, fixture / "fits" / job_id)
    archive = scratch / "metrics_checkpoint.zip"
    state = snapshot(fixture, assigned, manifest, archive,
                     {"validation_only": True, "sampling_calls": 0})
    restored = scratch / "restored"
    restore_metrics(archive, restored, manifest, assigned)
    # Repeating the restore must accept identical complete-fit records.
    restore_metrics(archive, restored, manifest, assigned)
    bootstrap(args.setup.resolve())
    # Frozen modules initialize the native library on first import. Complete
    # normal import initialization before testing the completed-fit skip path.
    import recovery_core, recovery_common, recovery_plan, recovery_runner, paper_metrics
    import recovery_engine
    import job_worker
    with patch.object(recovery_engine, "activate", side_effect=AssertionError("Engine activation during skip")), \
         patch.object(job_worker.subprocess, "run", side_effect=AssertionError("Sampling during skip")):
        for job_id in complete:
            assert verify_fit(restored / "fits" / job_id, manifest)
            job_worker.run(job_id, assignment_path, args.setup.resolve(),
                           scratch / "unused_sampling_work", restored)
    assert not (scratch / "unused_sampling_work").exists()
    record = {"status": "PASS", "manifest_hash": manifest["manifest_hash"],
              "real_new_fit_ids": complete, "restored_and_skipped": len(complete),
              "checkpoint_sha256": sha(archive), "sampling_calls": 0,
              "native_activation_calls_during_skip": 0,
              "native_library_initialized_before_skip_check": True,
              "identical_repeat_restore": "PASS",
              "snapshot_completed": state["completed"],
              "actual_running_output_modified": False}
    (PACKAGE / "validation/APPLICATION_RESUME_CHECKS.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
