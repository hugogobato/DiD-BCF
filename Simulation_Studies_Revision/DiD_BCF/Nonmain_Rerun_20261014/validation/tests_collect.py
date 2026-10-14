"""Test-only fixtures for the non-main result collector; no sampler is run."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parents[1]
HELPERS = PACKAGE_DIR / "helpers"
sys.path.insert(0, str(HELPERS))

import collect as collector  # noqa: E402
import common  # noqa: E402
from paper_metrics import _APPLICATION_COLUMNS  # noqa: E402


FIXTURE_PHASE_COUNTS = {
    "diagnostic": 1,
    "pt_impact": 1,
    "sensitivity": 1,
    "application": 1,
    "application_sensitivity": 1,
}
NUMERIC_LOOKING_HEX_ID = "1101199218e104432309"


def _jobs():
    specs = [
        (NUMERIC_LOOKING_HEX_ID, "diagnostic", "pretrend", 1),
        ("pt-impact-fixture", "pt_impact", "effect", 1),
        ("sensitivity-fixture", "sensitivity", "effect", 1),
        ("application-fixture", "application", "application", 1),
        ("application-sensitivity-fixture", "application_sensitivity", "application", 2),
    ]
    return [
        {
            "id": job_id,
            "phase": phase,
            "kind": kind,
            "dgp": "fixture_dgp",
            "setting": "fixture_setting",
            "N": 20,
            "linearity_degree": 1,
            "rep": rep,
            "spec": "unit",
            "sigma2_shape": 2.0,
            "sigma2_rate": 2.0,
        }
        for job_id, phase, kind, rep in specs
    ]


def _application_frame(kind, job):
    rows = {
        "gate": {
            "spec": job["spec"], "group": "all", "n_obs": 10,
            "att_log": 0.2, "post_sd": 0.1, "lo95_log": 0.0,
            "hi95_log": 0.4, "att_pct": 22.1, "lo95_pct": 0.0,
            "hi95_pct": 49.2, "p_bayes": 0.98,
        },
        "catt": {
            "unit_id": "county-001", "time": 2004, "lpop": 3.1,
            "lpop_group": "small", "spec": job["spec"],
            "catt_log": 0.2, "catt_pct": 22.1,
        },
        "contrasts": {
            "spec": job["spec"], "contrast": "small-large", "diff_log": 0.1,
            "post_sd": 0.05, "lo95_log": 0.0, "hi95_log": 0.2,
            "p_bayes": 0.97,
        },
    }[kind]
    return pd.DataFrame([{
        **rows,
        "job_id": job["id"],
        "budget_key": "earlier_budget_repaired",
        "rep": job["rep"],
    }], columns=[*_APPLICATION_COLUMNS[kind], "job_id", "budget_key", "rep"])


def _per_replication(job):
    ids = ("decision_gate", "decision_pretrend") if job["phase"] == "diagnostic" else ("fixture_estimand",)
    return pd.DataFrame([
        {
            "job_id": job["id"],
            "budget_key": "earlier_budget_repaired",
            "phase": job["phase"],
            "kind": job["kind"],
            "dgp": job["dgp"],
            "setting": job["setting"],
            "N": job["N"],
            "linearity_degree": job["linearity_degree"],
            "rep": job["rep"],
            "spec": job["spec"],
            "sigma2_shape": job["sigma2_shape"],
            "sigma2_rate": job["sigma2_rate"],
            "estimand_type": "decision" if job["phase"] == "diagnostic" else "ATT",
            "estimand_id": estimand_id,
            "method": "didbcf",
            "post_mean": 0.2,
            "true": float("nan") if job["kind"] == "application" else 0.1,
        }
        for estimand_id in ids
    ])


def _record(job, root, record_hash=None):
    folder = root / "fits" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    per = _per_replication(job)
    diag = pd.DataFrame([{
        "job_id": job["id"], "budget_key": "earlier_budget_repaired",
        "parameter": "fixture_parameter", "post_mean": 0.2,
    }])
    per.to_csv(folder / "per_replication.csv", index=False)
    diag.to_csv(folder / "diagnostics.csv", index=False)
    outputs = {"per_replication.csv": "fixture-checksum", "diagnostics.csv": "fixture-checksum"}
    app_counts = {}
    if job["phase"] in collector.APPLICATION_PHASES:
        for kind in ("gate", "catt", "contrasts"):
            _application_frame(kind, job).to_csv(folder / f"application_{kind}.csv", index=False)
            outputs[f"application_{kind}.csv"] = "fixture-checksum"
            app_counts[kind] = 1
    record = {
        "job_id": job["id"],
        "record_hash": record_hash or f"hash-{job['id']}",
        "budget_key": "earlier_budget_repaired",
        "outputs": outputs,
        "per_replication_rows": len(per),
        "diagnostic_rows": len(diag),
        "application_output_rows": app_counts,
    }
    (folder / "complete.json").write_text(json.dumps(record))
    return folder


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="collect-fixture-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        (self.package / "reuse" / "fits").mkdir(parents=True)
        (self.package / "assignments").mkdir()
        self.jobs = _jobs()
        self.manifest = {
            "manifest_hash": "fixture-manifest",
            "jobs": self.jobs,
            "reused_job_ids": [],
        }
        self.package_record = {"package_hash": "fixture-package"}
        self.patches = [
            patch.object(collector, "PACKAGE", self.package),
            patch.object(common, "PACKAGE", self.package),
            patch.object(collector, "PHASE_EXPECTED", dict(FIXTURE_PHASE_COUNTS)),
            patch.object(collector, "REUSED_EXPECTED", 0),
            patch.object(collector, "bootstrap", lambda: None),
            patch.object(collector, "verify_package", lambda: (self.package_record, self.manifest)),
            patch.object(collector, "verify_fit", lambda folder, manifest: json.loads((Path(folder) / "complete.json").read_text())),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def _source(self, name, job_ids, complete_jobs):
        source = self.root / name
        (source / "bundle").mkdir(parents=True)
        payload = {
            "name": name,
            "manifest_hash": self.manifest["manifest_hash"],
            "job_ids": list(job_ids),
        }
        payload["assignment_hash"] = common.digest(payload)
        assignment_path = self.package / "assignments" / f"{name}.json"
        assignment_path.write_text(json.dumps(payload))
        (source / "bundle" / "assignment.json").write_text(json.dumps(payload))
        for job in complete_jobs:
            _record(job, source)
        return source

    def test_partial_collection_keeps_ids_pretrend_rows_and_application_tables(self):
        diag_job, _, _, app_job, _ = self.jobs
        source = self._source("local-app", [job["id"] for job in self.jobs], [diag_job, app_job])
        out = self.root / "collected"
        observed_metric_groups = []

        def aggregate(group):
            observed_metric_groups.append((
                group["phase"].iloc[0], group["budget_key"].iloc[0],
                group["sigma2_shape"].iloc[0], group["sigma2_rate"].iloc[0],
            ))
            return pd.DataFrame(), pd.DataFrame()

        with patch("paper_metrics.aggregate_metrics", side_effect=aggregate):
            status = collector.collect([source], out)

        self.assertEqual(status["phase_expected"], FIXTURE_PHASE_COUNTS)
        self.assertEqual(status["phase_completed"], {
            "diagnostic": 1, "pt_impact": 0, "sensitivity": 0,
            "application": 1, "application_sensitivity": 0,
        })
        self.assertEqual(status["completed_total"], 2)
        self.assertEqual(status["expected_total"], 5)
        self.assertEqual(status["missing_total"], 3)
        self.assertEqual(len(status["missing_ids"]), 3)
        self.assertEqual(observed_metric_groups, [("diagnostic", "earlier_budget_repaired", 2.0, 2.0)])

        per = pd.read_csv(out / "per_replication.csv", dtype={"job_id": str})
        self.assertIn(NUMERIC_LOOKING_HEX_ID, set(per["job_id"]))
        decisions = per.loc[per["job_id"] == NUMERIC_LOOKING_HEX_ID]
        self.assertEqual(set(decisions["estimand_id"]), {"decision_gate", "decision_pretrend"})
        self.assertEqual(len(decisions), 2)
        application_rows = per.loc[per["phase"] == "application"]
        self.assertTrue(application_rows["true"].isna().all())

        diagnostics = pd.read_csv(out / "diagnostics.csv", dtype={"job_id": str})
        self.assertEqual(set(diagnostics["phase"]), {"diagnostic", "application"})
        self.assertIn(NUMERIC_LOOKING_HEX_ID, set(diagnostics["job_id"]))
        for kind in ("gate", "catt", "contrasts"):
            app = pd.read_csv(out / f"application_{kind}.csv", dtype={"job_id": str})
            self.assertEqual(len(app), 1)
            self.assertEqual(app.loc[0, "job_id"], app_job["id"])
            self.assertFalse({"true", "error", "absolute_error", "squared_error"} & set(app.columns))
        self.assertEqual(status["application_truth_or_error_metrics_created"], False)

    def test_identical_duplicate_is_deduplicated_and_conflict_is_rejected(self):
        job = self.jobs[0]
        first = self._source("first", [job["id"]], [job])
        second = self._source("second", [job["id"]], [])
        _record(job, second)
        out = self.root / "deduped"
        with patch("paper_metrics.aggregate_metrics", return_value=(pd.DataFrame(), pd.DataFrame())):
            status = collector.collect([first, second], out)
        self.assertEqual(status["completed_total"], 1)
        self.assertEqual(len(status["duplicates_identical"]), 1)

        third = self._source("third", [job["id"]], [])
        _record(job, third, record_hash="conflicting-record-hash")
        with self.assertRaisesRegex(ValueError, "Conflicting repeated fit"):
            collector.collect([first, third], self.root / "conflict")

    def test_application_gate_and_zip_path_validation(self):
        app_job = self.jobs[3]
        source = self._source("application-only", [job["id"] for job in self.jobs], [app_job])
        with self.assertRaisesRegex(RuntimeError, "Application phases incomplete"):
            collector.collect([source], self.root / "app-required", require_application=True)

        unsafe = self.root / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("../../outside.csv", "x")
        with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
            collector.collect([unsafe], self.root / "zip-check")


if __name__ == "__main__":
    unittest.main(verbosity=2)
