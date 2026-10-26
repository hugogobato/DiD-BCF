"""Reproduce final main metrics and matched historical/budget comparisons.

No posterior arrays, fitted models, filtering or experimental sampling.
"""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = next(p for p in PACKAGE.parents if (p / "JBES_Submission").is_dir())
COLLECTION = PACKAGE / "Results/collected_20261014"
sys.path.insert(0, str(PACKAGE / "helpers"))
from common import bootstrap, sha
bootstrap()
import numpy as np
import pandas as pd
from paper_metrics import aggregate_metrics

CELL = ["dgp", "setting", "N", "linearity_degree"]
DATA = CELL + ["rep"]
TARGET = CELL + ["estimand_type", "estimand_id", "method"]
PAIR = DATA + ["estimand_type", "estimand_id", "method"]
BOOTSTRAPS = 5000


def read(path):
    return pd.read_csv(path, dtype={"job_id": str}, float_precision="round_trip", low_memory=False)


def table(frame, columns, digits=4):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
                                           for v in row) + " |")
    return "\n".join(lines)


def contrasts(raw, new):
    output = []
    for key, right in new.groupby(TARGET, dropna=False):
        identity = dict(zip(TARGET, key))
        left = raw
        for name, value in identity.items():
            left = left.loc[left[name].eq(value)]
        left, right = left.sort_values("rep"), right.sort_values("rep")
        assert left.rep.tolist() == right.rep.tolist()
        n = len(left)
        seed = int(hashlib.sha256(json.dumps(identity, sort_keys=True, default=float).encode()).hexdigest()[:16], 16)
        indexes = np.random.default_rng(seed).integers(n, size=(BOOTSTRAPS, n))
        row = {**identity, "n_pairs": n}
        if identity["estimand_type"] == "CATT":
            a, b = left.surf_rmse.to_numpy(), right.surf_rmse.to_numpy()
            boot = b[indexes].mean(axis=1) / a[indexes].mean(axis=1)
            row.update(rmse_ratio=b.mean()/a.mean(), original_rmse=a.mean(), repaired_rmse=b.mean(),
                       coverage95_original=left.surf_cover95.mean(), coverage95_repaired=right.surf_cover95.mean(),
                       width95_original=left.surf_len95.mean(), width95_repaired=right.surf_len95.mean())
        else:
            assert np.allclose(left.true, right.true, rtol=0, atol=1e-12)
            a = (left.post_mean-left.true).to_numpy()
            b = (right.post_mean-right.true).to_numpy()
            boot = np.sqrt(np.square(b[indexes]).mean(axis=1) / np.square(a[indexes]).mean(axis=1))
            row.update(rmse_ratio=np.sqrt(np.mean(b*b)/np.mean(a*a)),
                       original_rmse=np.sqrt(np.mean(a*a)), repaired_rmse=np.sqrt(np.mean(b*b)),
                       bias_original=a.mean(), bias_repaired=b.mean(),
                       bias_difference_mcse=(b-a).std(ddof=1)/np.sqrt(n),
                       coverage95_original=((left.q025<=left.true)&(left.true<=left.q975)).mean(),
                       coverage95_repaired=((right.q025<=right.true)&(right.true<=right.q975)).mean(),
                       width95_original=(left.q975-left.q025).mean(),
                       width95_repaired=(right.q975-right.q025).mean())
        row.update(rmse_ratio_lo95=np.quantile(boot,.025), rmse_ratio_hi95=np.quantile(boot,.975))
        row["rmse_change_percent"] = 100*(row["rmse_ratio"]-1)
        output.append(row)
    return pd.DataFrame(output)


def main():
    status = json.loads((COLLECTION / "status.json").read_text())
    assert status["main_completed"] == 5100 and status["bridge_completed"] == 8
    all_new = read(COLLECTION / "per_replication.csv")
    new = all_new.loc[all_new.phase.eq("main")].copy()
    assert new.job_id.nunique() == 5100 and not new.duplicated(PAIR).any()
    scalar = new.loc[new.estimand_type.ne("CATT")]
    assert np.isfinite(scalar[["post_mean", "true", "sd", "q025", "q05", "q95", "q975"]]).all().all()
    assert (scalar.q025 <= scalar.q05).all() and (scalar.q05 <= scalar.q95).all() and (scalar.q95 <= scalar.q975).all()
    surfaces = new.loc[new.estimand_type.eq("CATT")]
    assert np.isfinite(surfaces[["surf_rmse", "surf_mae", "surf_cover95", "surf_len95", "surf_n"]]).all().all()
    assert surfaces.surf_cover95.between(0,1).all() and surfaces.surf_n.gt(0).all()

    raw_path = ROOT / "DiD-BCF/Simulation_Studies_Revision/Results/aggregated/all_summaries.csv.gz"
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    raw = raw.loc[raw.method.eq("plain")].copy()
    paired = new[PAIR + ["job_id"]].merge(raw, on=PAIR, how="inner", validate="one_to_one")
    paired_new = paired[PAIR].merge(new, on=PAIR, how="left", validate="one_to_one")
    assert paired.job_id.nunique() == 4800
    assert set(paired.job_id) == set(paired_new.job_id)
    checks = paired[PAIR+["true", "surf_n"]].merge(paired_new[PAIR+["true", "surf_n"]], on=PAIR, suffixes=("_raw","_new"), validate="one_to_one")
    is_surface = checks.estimand_type.eq("CATT")
    assert np.allclose(checks.loc[~is_surface,"true_raw"],checks.loc[~is_surface,"true_new"],rtol=0,atol=1e-12)
    assert np.array_equal(checks.loc[is_surface,"surf_n_raw"],checks.loc[is_surface,"surf_n_new"])
    unpaired = new.loc[~new.job_id.isin(paired.job_id)].drop_duplicates("job_id")
    assert len(unpaired)==300 and unpaired.setting.eq("B1_null").all() and unpaired.rep.ge(100).all()
    unpaired[DATA+["job_id"]].to_csv(HERE/"additional_null_replications.csv",index=False)
    full_scalar = read(COLLECTION / "scalar_metrics.csv")
    full_scalar = full_scalar.loc[full_scalar.phase.eq("main")].copy()
    full_surface = read(COLLECTION / "surface_metrics.csv")
    full_surface = full_surface.loc[full_surface.phase.eq("main")].copy()
    full_scalar.to_csv(HERE/"main_scalar_metrics.csv",index=False)
    full_surface.to_csv(HERE/"main_surface_metrics.csv",index=False)
    matched_scalar, matched_surface = [], []
    for arm, frame in (("original_raw",paired),("repaired_earlier",paired_new)):
        a,b=aggregate_metrics(frame)
        a["arm"],b["arm"]=arm,arm
        matched_scalar.append(a);matched_surface.append(b)
    pd.concat(matched_scalar).to_csv(HERE/"matched_scalar_metrics.csv",index=False)
    pd.concat(matched_surface).to_csv(HERE/"matched_surface_metrics.csv",index=False)
    comparison=contrasts(paired,paired_new)
    comparison.to_csv(HERE/"paired_metric_contrasts.csv",index=False)
    changes=comparison.groupby("estimand_type").agg(target_cells=("rmse_ratio","size"),
        min_rmse_change_percent=("rmse_change_percent","min"),median_rmse_change_percent=("rmse_change_percent","median"),
        max_rmse_change_percent=("rmse_change_percent","max")).reset_index()
    changes.to_csv(HERE/"metric_change_summary.csv",index=False)
    att=full_scalar.loc[full_scalar.estimand_type.eq("ATT")].sort_values(CELL)
    att.to_csv(HERE/"main_att_metrics.csv",index=False)
    comparison.loc[comparison.estimand_type.eq("ATT")].to_csv(HERE/"matched_att_comparison.csv",index=False)

    bridge = read(COLLECTION/"bridge_budget_summaries.csv")
    bkeys=["job_id","estimand_type","estimand_id","method"]
    short=bridge.loc[bridge.budget_key.eq("earlier_budget_repaired")]
    long=bridge.loc[bridge.budget_key.eq("B3_warm2500_ret2000")]
    assert not short.duplicated(bkeys).any() and not long.duplicated(bkeys).any()
    bridge=short.merge(long,on=bkeys,suffixes=("_earlier","_B3"),validate="one_to_one")
    assert len(bridge)==len(short)==len(long)
    bridge["mean_difference"]=bridge.post_mean_earlier-bridge.post_mean_B3
    bridge["difference_over_B3_sd"]=bridge.mean_difference.abs()/bridge.sd_B3
    bridge["width95_ratio"]=(bridge.q975_earlier-bridge.q025_earlier)/(bridge.q975_B3-bridge.q025_B3)
    bridge["reject_earlier"]=bridge.p_bayes_earlier.lt(.025)
    bridge["reject_B3"]=bridge.p_bayes_B3.lt(.025)
    bridge["decision_changed"]=bridge.reject_earlier.ne(bridge.reject_B3)
    bridge.to_csv(HERE/"bridge_target_comparison.csv",index=False)
    bridge_att=bridge.loc[bridge.estimand_type.eq("ATT"),bkeys+["phase_earlier","spec_earlier","sigma2_rate_earlier","post_mean_earlier","post_mean_B3","q025_earlier","q975_earlier","q025_B3","q975_B3","difference_over_B3_sd","width95_ratio"]]
    bridge_att.to_csv(HERE/"bridge_att_comparison.csv",index=False)
    pretrend=bridge.loc[bridge.estimand_type.str.startswith("PRE")]
    pretrend.to_csv(HERE/"bridge_pretrend_comparison.csv",index=False)

    diagnostics=read(COLLECTION/"diagnostics.csv")
    meta=all_new.drop_duplicates("job_id")[["job_id","phase"]]
    if "phase" in diagnostics:
        diagnostics=diagnostics.rename(columns={"phase":"recorded_phase"})
    diagnostics=diagnostics.merge(meta,on="job_id",how="left",validate="many_to_one")
    assert diagnostics.phase.notna().all()
    if "recorded_phase" in diagnostics:
        assert (diagnostics.recorded_phase.isna() | diagnostics.recorded_phase.eq(diagnostics.phase)).all()
    diagnostics["family"]=diagnostics.parameter.str.replace(r"^(GATT|ES)_.*",r"\1",regex=True)
    diagnostic=diagnostics.groupby(["phase","family"]).agg(sequences=("rhat","size"),
        median_rhat=("rhat","median"),max_rhat=("rhat","max"),
        undefined_rhat=("rhat",lambda x:x.isna().sum()),
        fraction_rhat_over_105=("rhat",lambda x:x.dropna().gt(1.05).mean()),median_bulk_ess=("bulk_ess","median")).reset_index()
    diagnostic.to_csv(HERE/"supporting_diagnostic_summary.csv",index=False)
    inventory=read(HERE/"archive_inventory.csv")
    main_inventory=inventory.loc[inventory.assignment.str.startswith("main")]

    # Independent arithmetic on every ATT cell, separate from frozen metrics.
    for row in att.itertuples():
        g=scalar.loc[scalar.estimand_type.eq("ATT") & scalar.setting.eq(row.setting)&scalar.N.eq(row.N)&scalar.linearity_degree.eq(row.linearity_degree)]
        error=g.post_mean.to_numpy()-g.true.to_numpy()
        assert len(g)==row.n_reps
        assert np.isclose(row.bias,error.mean(),rtol=0,atol=1e-14)
        assert np.isclose(row.rmse,np.sqrt(np.mean(error**2)),rtol=0,atol=1e-14)
        assert np.isclose(row.mcse_bias,error.std(ddof=1)/np.sqrt(len(g)),rtol=0,atol=1e-14)
        assert row.cover95_count==((g.q025<=g.true)&(g.true<=g.q975)).sum()
    assert att.loc[att.setting.eq("B1_null"),"mape"].isna().all()
    report=["# Returned main paper rerun, 2026-10-14","",
        "All 60 main archives and four budget-check archives are complete. Checksums, job and assignment identities, source pins, variance/forest priors, seeds, engine specification and native/eligible-feature controls pass. There are 4,836 returned new main fits plus 264 verified reused main fits: 5,100/5,100 main fits, with eight additional non-main checks. No failed or missing assigned fits remain. Every fit enters evaluation, without diagnostic or performance filtering.","",
        "The repaired earlier budget retains similar point-estimation accuracy to historical raw in this grid: matched ATT RMSE changes range from -4.32% to +4.88% (median +0.50%), and treated CATT-surface changes from -0.51% to +3.57% (median +1.50%). No main replacement fits are needed for completeness. Interval undercoverage remains material: ATT 95% inclusion is 75/100, 76/100 and 79/100 at N=800 in the serial sweep for degrees 2, 3 and 1. The manuscript must report this limitation.","",
        f"The main grid has {len(att)} cells: 100 replications per cell except the three null cells, which have 200 each. Historical raw has matching summaries for 4,800 datasets; the 300 extra null replications (indices 100–199) enter full repaired metrics but not paired comparisons. Scalar truths and treated surface row counts match on every paired target. Historical panel bytes are unavailable in the scalar summaries; matching summaries is not a byte-level historical panel verification.","",
        "## Matched changes from historical raw","",table(changes,list(changes.columns)),"",
        "Changes are computed within each target/cell on exactly matched datasets. Surface RMSE is the mean per-dataset treated CATT-surface RMSE. Paired dataset bootstrap intervals (5,000 resamples) for every ratio are in paired_metric_contrasts.csv. Historical raw differs in kernel, prior, computation and environment; these comparisons do not isolate the native repair or prove empirical equivalence.","",
        "## Full repaired ATT metrics","",table(att,["setting","N","linearity_degree","n_reps","bias","mcse_bias","rmse","mae","cover95_count","cover95_n","cover95","len95"]),"",
        "All GATT and event-study targets, size/power, 90% intervals, Wilson intervals and conditional-effect surface metrics are saved in the accompanying CSVs. Undefined null-truth MAPE is retained as NaN. These surfaces concern treated observations, not transported population CATE.","",
        "## Non-main earlier versus B3 checks","",table(bridge_att,["phase_earlier","spec_earlier","sigma2_rate_earlier","post_mean_earlier","post_mean_B3","difference_over_B3_sd","width95_ratio"]),"",
        f"The two pretrend anchors contain {len(pretrend)} reported targets/decisions; {int(pretrend.decision_changed.sum())} change rejection at the stored one-tail p-value cutoff 0.025. These are single-dataset checks, not repeated-sample validation of power or coverage. Application truth is unavailable; no application error or coverage is invented. Detailed endpoints, pretrend decisions, county effects, group effects and contrasts are retained.","",
        "Across the six ATT anchors, the earlier/B3 posterior mean difference is at most 0.132 B3 posterior standard deviations. Earlier/B3 95% interval-width ratios range from 0.959 to 1.128. A subgroup pretrend interval has a larger width difference (ratio about 0.50), despite unchanged rejection; aggregate ATT stability does not imply every internal target is equally stable. These checks support recommending the earlier budget for the remaining performance-focused campaign, with all diagnostic flags retained. They do not establish a formal budget-equivalence or MCMC-convergence result.","",
        "## Supporting diagnostics and runtime","",
        table(diagnostic.loc[diagnostic.phase.eq("main") & diagnostic.family.isin(["ATT","sigma2","GATT","ES"])],list(diagnostic.columns)),"",
        "The earlier budget remains GFR50, burn-in0, retained500, thin5, three chains. The old full-state convergence gate remains not passed. Performance comparisons and exact-posterior asymptotics do not certify finite-MCMC convergence or nominal interval coverage.","",
        f"Main notebook elapsed hours: minimum {main_inventory.elapsed_hours.min():.2f}, median {main_inventory.elapsed_hours.median():.2f}, maximum {main_inventory.elapsed_hours.max():.2f}. All 64 returned archives total {inventory.bytes.sum()/2**20:.2f} MiB. Outputs contain metrics/diagnostics/provenance and no posterior arrays. The collector reader fix preserves numeric-looking hexadecimal IDs as strings; the sealed helpers and original CSV/ZIP bytes are unchanged.","",
        "## Remaining paper work","",
        "The main rerun needs no replacement fits. The full original non-main universe still requires a separately frozen budget/allocation decision: 4,000 pretrend diagnostic jobs, 4,000 pretrend-impact jobs, 1,200 simulation sensitivity jobs, nine application jobs and 18 application-sensitivity jobs. Reusing the eight compatible earlier-budget anchors would leave 9,219 jobs (3,999 diagnostic, 4,000 pretrend-impact, 1,198 simulation sensitivity, six application and 16 application-sensitivity). No full non-main campaign or new fits were launched by this analysis. Final manuscript empirical tables/figures/text and the submission checks remain pending. See NEXT_STEPS.md and the standalone main_comparison.pdf figure."]
    (HERE/"RESULTS_REPORT.md").write_text("\n".join(report)+"\n")
    inputs={"raw_historical":sha(raw_path),"collection_status":sha(COLLECTION/"status.json"),
            "per_replication":sha(COLLECTION/"per_replication.csv"),"diagnostics":sha(COLLECTION/"diagnostics.csv"),
            "return_validation":sha(HERE/"RETURN_VALIDATION.json")}
    result={"status":"PASS","main_completed":5100,"main_cells":len(att),"paired_datasets":4800,
            "additional_null_datasets":300,"bootstrap_resamples":BOOTSTRAPS,"diagnostic_filtering":False,
            "independent_all_ATT_arithmetic":"PASS","new_fits":0,"inputs":inputs,
            "scripts":{p.name:sha(p) for p in HERE.glob("*.py")},
            "outputs":{p.name:sha(p) for p in HERE.iterdir() if p.suffix in (".csv",".md",".png",".pdf")}}
    (HERE/"ANALYSIS_VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n")
    print(changes.to_string(index=False),flush=True)
    print("ATT 95% coverage min/median/max",att.cover95.min(),att.cover95.median(),att.cover95.max(),flush=True)
    print("Bridge ATT",bridge_att.to_string(index=False),flush=True)
    print("Pretrend rejection changes",int(pretrend.decision_changed.sum()),"/",len(pretrend),flush=True)


if __name__=="__main__":
    main()
