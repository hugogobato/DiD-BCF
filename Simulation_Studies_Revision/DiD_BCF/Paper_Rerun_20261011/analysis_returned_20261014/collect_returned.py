# Derived from sealed helpers/collect.py (SHA256 5c602e2065dacbae6fd0b921e900b92761c56b7e83ec396a855a5f3259b2b476).
# Reader-only fix: preserve hexadecimal job IDs as text in every CSV.
# Frozen helper, package, manifest, fit checksums and experimental values are unchanged.
"""Collect compact results; reject conflicts and report every missing fit."""
from __future__ import annotations
import argparse
from pathlib import Path
import tempfile
import zipfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
from common import PACKAGE, atomic, assignment, bootstrap, read, sha, verify_fit, verify_package


def collect(sources, out, require_main=False):
    bootstrap()
    import pandas as pd
    from paper_metrics import aggregate_metrics, _write_csv
    _, manifest = verify_package()
    out.mkdir(parents=True,exist_ok=True)
    summaries, diagnostics, metadata, app = [], [], {}, {k:[] for k in ("gate","catt","contrasts")}
    duplicates = []
    def accept(folder, source, allowed_ids=None):
        record = verify_fit(folder,manifest)
        if not record:
            raise ValueError("Incomplete compact fit: " + str(folder))
        job_id = record["job_id"]
        if allowed_ids is not None and job_id not in allowed_ids:
            raise ValueError("Fit outside source assignment")
        if job_id in metadata:
            if metadata[job_id]["record_hash"] != record["record_hash"]:
                raise ValueError("Conflicting repeated fit " + job_id)
            duplicates.append({"job_id":job_id,"source":source})
            return
        metadata[job_id] = {**record,"source":source}
        per = pd.read_csv(folder / "per_replication.csv",float_precision="round_trip",dtype={"job_id":str})
        if len(per) != record["per_replication_rows"] or set(per.job_id) != {job_id} or set(per.budget_key) != {record["budget_key"]}:
            raise ValueError("Per-fit CSV row identity mismatch")
        if per.duplicated(["job_id","budget_key","estimand_type","estimand_id","method"]).any():
            raise ValueError("Repeated per-fit metric")
        summaries.append(per)
        diag = pd.read_csv(folder / "diagnostics.csv",float_precision="round_trip",dtype={"job_id":str})
        if len(diag) != record["diagnostic_rows"]:
            raise ValueError("Diagnostic row count mismatch")
        diagnostics.append(diag)
        for kind in app:
            path = folder / f"application_{kind}.csv"
            if path.exists():
                app[kind].append(pd.read_csv(path,float_precision="round_trip",dtype={"job_id":str}))
    for folder in sorted((PACKAGE / "reuse/fits").iterdir()):
        accept(folder,"verified_reuse")
    for source in sources:
        source = Path(source)
        if source.is_dir():
            assigned = assignment(source / "bundle/assignment.json",manifest)
            for folder in sorted((source / "fits").glob("*")):
                if (folder / "complete.json").exists():
                    accept(folder,str(source),assigned["job_ids"])
            continue
        with tempfile.TemporaryDirectory(prefix="didbcf-paper-collect-") as temp:
            root = Path(temp).resolve()
            with zipfile.ZipFile(source) as archive:
                if len(archive.namelist()) != len(set(archive.namelist())):
                    raise ValueError("Duplicate ZIP entries")
                for entry in archive.infolist():
                    name = entry.filename
                    target = (root / name).resolve()
                    if (not target.is_relative_to(root) or ".." in Path(name).parts or name.startswith("/")
                            or ((entry.external_attr >> 16) & 0o170000) == 0o120000):
                        raise ValueError("Unsafe archive member")
                    if entry.is_dir():
                        continue
                    if not ((name.startswith("fits/") or name.startswith("bundle/")) and target.suffix in (".csv",".json")):
                        raise ValueError("Not a metrics-only paper archive: " + name)
                    if entry.file_size > 16*2**20:
                        raise ValueError("Oversized compact artifact")
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.read(entry))
            assigned = assignment(root / "bundle/assignment.json",manifest)
            for folder in sorted((root / "fits").glob("*")):
                accept(folder,str(source),assigned["job_ids"])
    frame = pd.concat(summaries,ignore_index=True)
    diagnostic = pd.concat(diagnostics,ignore_index=True)
    _write_csv(frame,out / "per_replication.csv")
    _write_csv(diagnostic,out / "diagnostics.csv")
    scalar_tables, surfaces = [], []
    # The frozen metric keys omit budgets, phases and prior rates. Split on
    # these first so a sensitivity/application row never pools with main.
    for key, group in frame.groupby(["phase","budget_key","sigma2_shape","sigma2_rate"],dropna=False):
        scalar, surface = aggregate_metrics(group)
        for table, target in ((scalar,scalar_tables),(surface,surfaces)):
            if len(table):
                for field,value in zip(("phase","budget_key","sigma2_shape","sigma2_rate"),key):
                    table[field] = value
                target.append(table)
    scalar = pd.concat(scalar_tables,ignore_index=True) if scalar_tables else pd.DataFrame()
    surface = pd.concat(surfaces,ignore_index=True) if surfaces else pd.DataFrame()
    _write_csv(scalar,out / "scalar_metrics.csv")
    _write_csv(surface,out / "surface_metrics.csv")
    for kind,tables in app.items():
        if tables:
            _write_csv(pd.concat(tables,ignore_index=True),out / f"application_{kind}.csv")
    by_id = {j["id"]:j for j in manifest["jobs"]}
    missing = [j for j in manifest["jobs"] if j["id"] not in metadata]
    missing_columns = ["id","phase","setting","N","linearity_degree","rep","spec"]
    _write_csv(pd.DataFrame([{k:j.get(k) for k in missing_columns}
                            for j in missing],columns=missing_columns),out / "missing_jobs.csv")
    main_ids = {j["id"] for j in manifest["jobs"] if j["phase"] == "main"}
    status = {"manifest_hash":manifest["manifest_hash"],"main_expected":5100,
              "main_completed":len(main_ids & set(metadata)),
              "bridge_expected":8,"bridge_completed":len(set(metadata)-main_ids),
              "missing_main_ids":sorted(main_ids-set(metadata)),"missing_total":len(missing),
              "duplicates_identical":duplicates,"diagnostic_filtering":False,
              "full_state_gate_passed":False,"posterior_arrays_needed_for_collection":False}
    atomic(out / "status.json",status)
    atomic(out / "provenance.json", {"package_hash":read(PACKAGE / "package.json")["package_hash"],
           "sources":[{"path":str(p),"sha256":sha(p) if Path(p).is_file() else None} for p in sources],
           "fit_record_hashes":{j:r["record_hash"] for j,r in metadata.items()}})
    bridge = frame.loc[frame.job_id.isin(manifest["bridge_job_ids"])].copy()
    if len(bridge):
        reference = pd.concat([pd.read_csv(p,float_precision="round_trip",dtype={"job_id":str}) for p in
                              (PACKAGE / "bridge_reference/fits").glob("*/per_replication.csv")],ignore_index=True)
        bridge = pd.concat([bridge,reference],ignore_index=True)
        _write_csv(bridge,out / "bridge_budget_summaries.csv")
    print({k:v for k,v in status.items() if k not in ("missing_main_ids","duplicates_identical")})
    if require_main and status["main_completed"] != 5100:
        raise RuntimeError("Main campaign incomplete; status and partial tables were saved")
    return status

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources",type=Path,nargs="*")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--require-main",action="store_true")
    args = parser.parse_args()
    collect(args.sources,args.out,args.require_main)
