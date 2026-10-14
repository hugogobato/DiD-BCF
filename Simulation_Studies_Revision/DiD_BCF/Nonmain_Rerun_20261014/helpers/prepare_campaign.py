"""Freeze remaining non-main jobs, verified reuse and measured-time shards."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

from common import PACKAGE, WORKSPACE, BUDGET_KEY, atomic, digest, read, sha

SIM_PHASES = ("diagnostic", "pt_impact", "sensitivity")
APP_PHASES = ("application", "application_sensitivity")


def prepare(parent):
    import json
    import pandas as pd
    spec=importlib.util.spec_from_file_location("verified_parent_common",parent/"helpers/common.py")
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    parent_package,parent_manifest=old.verify_package()
    original=read(WORKSPACE/"DiD-BCF/Simulation_Studies_Revision/Proper_Prior_Rerun/manifest.json")
    jobs=[j for j in original["jobs"] if j["phase"]!="main"]
    by_id={j["id"]:j for j in jobs}
    assert len(jobs)==9227
    reused=set(parent_manifest["bridge_job_ids"])
    assert len(reused)==8 and reused.issubset(by_id)
    measured=[]
    registry={}
    # Full-fit elapsed times include all chains and compact postprocessing.
    # These are worker wall times, not notebook wall times or CPU-seconds.
    old_jobs={j["id"]:j for j in parent_manifest["jobs"]}
    timing_inputs={}
    for archive in sorted((parent/"Colab").glob("didbcf_paper_*.zip")):
        timing_inputs[archive.name]=sha(archive)
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if not name.endswith("/chain_provenance.json"):
                    continue
                job_id=name.split("/")[1]
                provenance=json.loads(zipped.read(name))
                job=old_jobs[job_id]
                measured.append({"job_id":job_id, **{k:job.get(k) for k in
                    ("kind","phase","N","linearity_degree","spec","sigma2_rate")},
                    "seconds":provenance["elapsed_seconds"],
                    "peak_rss_mb":max(c.get("peak_rss_mb",0) for c in provenance["chains"]),
                    "archive":archive.name})
                if job_id not in reused:
                    continue
                record=json.loads(zipped.read(f"fits/{job_id}/complete.json"))
                with tempfile.TemporaryDirectory(prefix="nonmain-reuse-") as temp:
                    folder=Path(temp)
                    for filename in [*record["outputs"],"complete.json"]:
                        (folder/filename).write_bytes(zipped.read(f"fits/{job_id}/{filename}"))
                    old.verify_fit(folder,parent_manifest)
                    assert digest(by_id[job_id])==record["job_hash"]
                    target=PACKAGE/"reuse/fits"/job_id
                    target.mkdir(parents=True,exist_ok=True)
                    for filename in record["outputs"]:
                        shutil.copy2(folder/filename,target/filename)
                    registry[job_id]={"record_hash":record["record_hash"],"outputs":record["outputs"],
                                      "parent_manifest_hash":parent_manifest["manifest_hash"],
                                      "archive":archive.name,"archive_sha256":timing_inputs[archive.name]}
    assert set(registry)==reused
    measured=pd.DataFrame(measured)
    assert not measured.job_id.duplicated().any()
    measured.to_csv(PACKAGE/"validation/measured_fit_times.csv",index=False)
    manifest={
        "schema":"didbcf-nonmain-metrics-1", "purpose":"final_nonmain_paper_rerun",
        "parent_manifest":original["manifest_hash"],
        "parent_repaired_manifest":parent_manifest["manifest_hash"],
        **{k:parent_manifest[k] for k in ("recovery_version","method_version","model_spec",
                                          "engine_specification","source_hashes","dependencies","budgets","budget_keys","trace")},
        "jobs":jobs,"reused_job_ids":sorted(reused),"reuse_registry":registry,
        "reuse_source_hashes_sha256":digest(parent_manifest["source_hashes"]),
        "phase_expected":dict(Counter(j["phase"] for j in jobs)),
        "execution":{"colab_phases":list(SIM_PHASES),"local_phases":list(APP_PHASES)},
        "decision_policy":{
            "policy":"user_authorized_earlier_repaired_budget_for_remaining_studies",
            "evidence":"completed 5100-fit repaired main grid and eight earlier-versus-B3 nonmain anchors",
            "strict_full_state_gate_passed":False,"diagnostic_filtering":False,
            "note":"No claim of finite-MCMC convergence or nominal raw-interval calibration; retain every assigned fit."},
        "export":parent_manifest["export"],
    }
    manifest["manifest_hash"]=digest(manifest)
    atomic(PACKAGE/"manifest.json",manifest)
    for job_id in sorted(reused):
        original_record=registry[job_id]
        folder=PACKAGE/"reuse/fits"/job_id
        import pandas as pd
        per=pd.read_csv(folder/"per_replication.csv",dtype={"job_id":str})
        diag=pd.read_csv(folder/"diagnostics.csv",dtype={"job_id":str})
        job=by_id[job_id]
        rec={"manifest_hash":manifest["manifest_hash"],"job_id":job_id,"job_hash":digest(job),
             "budget_key":BUDGET_KEY,"chain_seeds":([None] if job["kind"] in ("pretrend","application_pretrend") else [1000*job["rep"]+i for i in range(3)]),
             "engine_specification":manifest["engine_specification"],"origin":"reused_verified_earlier_fit",
             "reused_from_record_hash":original_record["record_hash"],"outputs":original_record["outputs"],
             "per_replication_rows":len(per),"diagnostic_rows":len(diag),
             "application_output_rows":{kind:len(pd.read_csv(folder/f"application_{kind}.csv")) for kind in ("gate","catt","contrasts") if (folder/f"application_{kind}.csv").exists()}}
        rec["record_hash"]=digest(rec)
        atomic(folder/"complete.json",rec)
    effects=measured.loc[measured.kind.eq("effect") & measured.phase.eq("main")]
    q75=effects.groupby(["N","linearity_degree"]).seconds.quantile(.75)
    diagnostic=float(measured.loc[measured.kind.eq("pretrend"),"seconds"].iloc[0])
    app_pre=float(measured.loc[measured.kind.eq("application_pretrend"),"seconds"].iloc[0])
    def estimate(job):
        kind=job["kind"]
        if kind=="pretrend":
            # Only degree1 was benchmarked; transformed degree3 design gets
            # an additional scheduling margin, with no change to the model.
            return diagnostic*1.30*(1.4 if job["linearity_degree"]==3 else 1.)
        if kind=="application_pretrend":return app_pre*1.30
        if kind=="application":
            subset=measured.loc[measured.kind.eq("application") & measured.spec.eq(job["spec"])]
            return float(subset.seconds.max())*1.25
        n,d=job["N"],job["linearity_degree"]
        value=float(q75.loc[(n,d)])
        return value*1.20*(1.15 if job["phase"]=="sensitivity" else 1.10)
    pending=[j for j in jobs if j["id"] not in reused]
    simulation=[j for j in pending if j["phase"] in SIM_PHASES]
    application=[j for j in pending if j["phase"] in APP_PHASES]
    assert len(simulation)==9197 and len(application)==22
    target=15*3600.0 # 7.5 hours sampling with two workers, 0.5 setup reserve.
    count=math.ceil(sum(estimate(j) for j in simulation)/target)
    bins=[[] for _ in range(count)];loads=[0.]*count
    for job in sorted(simulation,key=lambda j:(-estimate(j),j["id"])):
        index=min(range(count),key=lambda i:(loads[i],len(bins[i]),i))
        bins[index].append(job["id"]);loads[index]+=estimate(job)
    scheduling=[]
    def write_assignment(name,ids,execution,load,wave):
        item={"name":name,"execution":execution,"phase":"nonmain_mixed" if execution=="colab" else "local_application",
              "manifest_hash":manifest["manifest_hash"],"job_ids":ids,
              "phase_counts":dict(Counter(by_id[i]["phase"] for i in ids)),
              "estimated_fit_seconds":{i:estimate(by_id[i]) for i in ids},"target_hours":8,
              "nominal_workers":2 if execution=="colab" else 6,
              "estimated_wall_hours":load/(7200 if execution=="colab" else 21600)+.5,
              "hard_runtime_hours":9.5 if execution=="colab" else 4,
              "setup_reserve_hours":.5,"wave":wave}
        item["assignment_hash"]=digest(item)
        atomic(PACKAGE/"assignments"/f"{name}.json",item)
        scheduling.append({"notebook":name,"execution":execution,"fits":len(ids),"wave":wave,
                           "estimated_hours_two_workers":item["estimated_wall_hours"],**item["phase_counts"]})
    assigned=[]
    for index,(ids,load) in enumerate(zip(bins,loads)):
        wave,shard=index//48+1,index%48
        name=f"nonmain_wave_{wave:02d}_shard_{shard:02d}"
        write_assignment(name,ids,"colab",load,wave);assigned+=ids
    write_assignment("local_application",[j["id"] for j in application],"local",sum(estimate(j) for j in application),0)
    assert len(assigned)==len(set(assigned))==9197
    assert set(assigned)|{j["id"] for j in application}|reused==set(by_id)
    pd.DataFrame(scheduling).to_csv(PACKAGE/"scheduling.csv",index=False)
    prep={"expected_total":len(jobs),"phase_expected":manifest["phase_expected"],"reused":8,
          "colab_new_fits":len(simulation),"local_new_fits":len(application),
          "colab_notebooks":count,"colab_waves":math.ceil(count/48),"max_concurrent_notebooks":48,
          "min_estimated_notebook_hours":min(loads)/7200+.5,"max_estimated_notebook_hours":max(loads)/7200+.5,
          "estimated_total_simulation_worker_hours":sum(loads)/3600,
          "measured_diagnostic_seconds":diagnostic,"measured_application_pretrend_seconds":app_pre,
          "scheduling_margins":{"diagnostic":1.30,"diagnostic_degree3_extra":1.4,"pt_impact":1.32,"sensitivity":1.38},
          "timing_inputs":timing_inputs,
          "note":"Planning estimates from returned complete-fit worker wall times. Two-worker Colab; hardware variation or memory constraints may require resume."}
    atomic(PACKAGE/"preparation.json",prep)
    print({k:v for k,v in prep.items() if k!="timing_inputs"},flush=True)


def seal():
    manifest=read(PACKAGE/"manifest.json")
    files={}
    for base in ("helpers","assignments","reuse"):
        for path in sorted((PACKAGE/base).rglob("*")):
            if path.is_file() and path.suffix in (".py",".json",".csv"):
                files[str(path.relative_to(PACKAGE))]=sha(path)
    for name in ("scheduling.csv","preparation.json"):files[name]=sha(PACKAGE/name)
    package={"schema":"didbcf-nonmain-package-1","manifest_hash":manifest["manifest_hash"],"files":files}
    package["package_hash"]=digest(package)
    atomic(PACKAGE/"package.json",package)
    print("Sealed",package["package_hash"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path)
    parser.add_argument("--seal",action="store_true")
    args=parser.parse_args()
    if args.parent:prepare(args.parent.resolve())
    if args.seal:seal()
