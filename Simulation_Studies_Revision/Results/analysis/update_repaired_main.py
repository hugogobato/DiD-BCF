"""Overlay verified repaired main metrics on immutable historical benchmarks.

Historical aggregated/ remains unchanged, including the raw comparator used
by the paired reports. The versioned overlay is the manuscript's new input.
No historical non-main fit is relabeled as a repaired result.
"""
from pathlib import Path
import hashlib
import json
import sys

RESULTS = Path(__file__).resolve().parents[1]
REVISION = RESULTS.parent
CAMPAIGN = REVISION / "DiD_BCF/Paper_Rerun_20261011"
if not CAMPAIGN.exists():
    CAMPAIGN = REVISION / "Proper_Prior_Rerun/Paper_Rerun_20261011"
SOURCE = CAMPAIGN / "Results/collected_20261014"
HISTORICAL = RESULTS / "aggregated"
OUT = RESULTS / "aggregated_repaired_20261014"
sys.path.insert(0, str(CAMPAIGN / "helpers"))
from common import bootstrap, sha, verify_package
bootstrap()
import numpy as np
import pandas as pd
from did_bcf_revision.metrics import sqrt_n_summary

CELL = ["dgp", "setting", "linearity_degree", "N"]


def replace(frame, repaired, keys):
    """Replace precisely the declared main cells; label retained old raw rows."""
    historical = frame.copy()
    main_cells = repaired[CELL].drop_duplicates()
    marked = historical.merge(main_cells.assign(_replaced=True), on=CELL, how="left", validate="many_to_one")
    take = marked.method.eq("plain") & marked._replaced.eq(True)
    retained = marked.loc[~take].drop(columns="_replaced").copy()
    retained["result_origin"] = "historical"
    retained.loc[retained.method.eq("plain"), "method"] = "plain_historical"
    repaired = repaired.copy()
    repaired["result_origin"] = "repaired_main_20261014"
    out = pd.concat([retained,repaired], ignore_index=True)
    assert not out.duplicated(keys).any(), out.loc[out.duplicated(keys,keep=False),keys].head()
    return out


def main():
    verify_package()
    OUT.mkdir(exist_ok=True)
    status = json.loads((SOURCE / "status.json").read_text())
    assert status["main_completed"] == 5100 and not status["missing_main_ids"]
    per = pd.read_csv(SOURCE / "per_replication.csv", dtype={"job_id":str},float_precision="round_trip")
    per = per.loc[per.phase.eq("main")].copy()
    assert per.job_id.nunique()==5100
    metric = pd.read_csv(SOURCE / "scalar_metrics.csv",float_precision="round_trip")
    metric = metric.loc[metric.phase.eq("main")].copy()
    surface = pd.read_csv(SOURCE / "surface_metrics.csv",float_precision="round_trip")
    surface = surface.loc[surface.phase.eq("main")].copy()
    history_names=["metrics_all.csv","surface_all.csv","sqrtn_all.csv","coverage_grid.csv","all_summaries.csv.gz"]
    before={n:sha(HISTORICAL/n) for n in history_names}
    keys=CELL+["estimand_type","estimand_id","method"]
    met=replace(pd.read_csv(HISTORICAL/"metrics_all.csv"),metric,keys)
    surf=replace(pd.read_csv(HISTORICAL/"surface_all.csv"),surface,CELL+["spec","method"])
    # Sqrt(N) diagnostics are regenerated from every repaired replication.
    sq=[]
    for et,eid in (("ATT","ATT"),("GATT","g=4_t=4"),("GATT","g=4_t=7")):
        sub=per.loc[per.estimand_type.eq(et)&per.estimand_id.eq(eid)]
        frame=sqrt_n_summary(sub,et,eid)
        frame["estimand_type"],frame["estimand_id"]=et,eid
        sq.append(frame)
    sq=pd.concat(sq,ignore_index=True)
    # Historical sqrt-N tables omit dgp; use their exact published schema.
    oldsq=pd.read_csv(HISTORICAL/"sqrtn_all.csv")
    skeys=[c for c in CELL+["estimand_type","estimand_id"] if c in oldsq]
    marked=oldsq.merge(sq[skeys].drop_duplicates().assign(_new=True),on=skeys,how="left",validate="many_to_one")
    oldsq=marked.loc[~(marked.method.eq("plain")&marked._new.eq(True))].drop(columns="_new")
    oldsq.loc[oldsq.method.eq("plain"),"method"]="plain_historical"
    sq=pd.concat([oldsq,sq],ignore_index=True)
    grid=per.groupby(CELL+["method"],dropna=False).rep.nunique().reset_index(name="n_reps")
    oldgrid=pd.read_csv(HISTORICAL/"coverage_grid.csv")
    gkeys=[c for c in CELL if c in oldgrid]
    marked=oldgrid.merge(grid[gkeys].drop_duplicates().assign(_new=True),on=gkeys,how="left",validate="many_to_one")
    oldgrid=marked.loc[~(marked.method.eq("plain")&marked._new.eq(True))].drop(columns="_new")
    oldgrid.loc[oldgrid.method.eq("plain"),"method"]="plain_historical"
    grid=pd.concat([oldgrid,grid],ignore_index=True)
    for name, frame in (("metrics_all.csv",met),("surface_all.csv",surf),("sqrtn_all.csv",sq),("coverage_grid.csv",grid),("main_per_replication.csv",per)):
        frame.to_csv(OUT/name,index=False,float_format="%.17g",na_rep="NaN")
    # An explicit main-only table is convenient for manuscript number checks.
    metric.to_csv(OUT/"main_scalar_metrics.csv",index=False,float_format="%.17g",na_rep="NaN")
    surface.to_csv(OUT/"main_surface_metrics.csv",index=False,float_format="%.17g",na_rep="NaN")
    assert before=={n:sha(HISTORICAL/n) for n in history_names}
    assert len(metric.loc[metric.estimand_type.eq("ATT")])==48
    note={"status":"PASS","main_fits":5100,"main_cells":48,
          "repaired_policy":"GFR50, burn-in0, retained500, thinning5, three chains, IG(2,1)",
          "historical_aggregated_unchanged":before,
          "source_manifest_hash":status["manifest_hash"],
          "source_hashes":{n:sha(SOURCE/n) for n in ("status.json","per_replication.csv","scalar_metrics.csv","surface_metrics.csv")},
          "outputs":{p.name:sha(p) for p in OUT.glob("*.csv")},
          "historical_nonmain_raw_label":"plain_historical","diagnostic_filtering":False,"new_fits":0}
    (OUT/"provenance.json").write_text(json.dumps(note,indent=2)+"\n")
    print("PASS: repaired main overlay; immutable historical comparator retained",flush=True)


if __name__=="__main__":
    main()
