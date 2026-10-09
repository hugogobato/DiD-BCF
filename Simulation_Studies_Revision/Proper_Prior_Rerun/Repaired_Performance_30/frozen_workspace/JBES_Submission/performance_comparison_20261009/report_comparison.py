"""Complete, unscreened paired paper metrics with dataset-level uncertainty."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KEYS = ["job_id", "estimand_type", "estimand_id", "method"]
CELL = ["dgp", "setting", "N", "linearity_degree"]
ARMS = ("saved_proper_prior", "repaired_earlier", "repaired_B3")


def wilson(count, n, z=1.959963984540054):
    p = count / n
    denominator = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denominator
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0., centre-radius), min(1., centre+radius)


def paired_statistics(left, right, indices, surface=False):
    """Dataset indices are common to both arms, including ratio uncertainty."""
    import numpy as np
    output = {}
    if surface:
        names = ["surf_rmse", "surf_mae", "surf_mape", "surf_cover90", "surf_cover95", "surf_len90", "surf_len95"]
        for name in names:
            a, b = left[name].to_numpy(dtype=float), right[name].to_numpy(dtype=float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                output[name + "_available_pairs"] = int((np.isfinite(a) & np.isfinite(b)).sum())
                continue
            diff = b-a
            boot = diff[indices].mean(axis=1)
            output.update({name+"_difference": float(diff.mean()), name+"_difference_mcse": float(diff.std(ddof=1)/np.sqrt(len(diff))),
                           name+"_difference_lo95": float(np.quantile(boot,.025)), name+"_difference_hi95": float(np.quantile(boot,.975))})
        a, b = left.surf_rmse.to_numpy(dtype=float), right.surf_rmse.to_numpy(dtype=float)
        ratio = b[indices].mean(axis=1)/a[indices].mean(axis=1)
        output.update(surface_rmse_ratio=float(b.mean()/a.mean()), surface_rmse_ratio_lo95=float(np.quantile(ratio,.025)),
                      surface_rmse_ratio_hi95=float(np.quantile(ratio,.975)))
        return output
    a = left.post_mean.to_numpy(dtype=float)-left.true.to_numpy(dtype=float)
    b = right.post_mean.to_numpy(dtype=float)-right.true.to_numpy(dtype=float)
    if not np.allclose(left.true, right.true, rtol=0, atol=1e-10):
        raise ValueError("Paired scalar truths differ")
    outputs = {"bias_difference": (b-a, (b-a)[indices].mean(axis=1)),
               "mae_difference": (np.abs(b)-np.abs(a), (np.abs(b)-np.abs(a))[indices].mean(axis=1))}
    for level, low, high in ((90,"q05","q95"),(95,"q025","q975")):
        def included(frame):
            return ((frame[low] <= frame.true) & (frame.true <= frame[high])).to_numpy(dtype=float)
        diff = included(right)-included(left)
        outputs[f"coverage{level}_difference"] = (diff, diff[indices].mean(axis=1))
        diff = (right[high]-right[low]).to_numpy(dtype=float)-(left[high]-left[low]).to_numpy(dtype=float)
        outputs[f"width{level}_difference"] = (diff, diff[indices].mean(axis=1))
    for name, (diff, boot) in outputs.items():
        output[name] = float(diff.mean())
        output[name+"_mcse"] = float(diff.std(ddof=1)/np.sqrt(len(diff)))
        output[name+"_lo95"] = float(np.quantile(boot,.025))
        output[name+"_hi95"] = float(np.quantile(boot,.975))
    boot_a = np.sqrt((a[indices]**2).mean(axis=1))
    boot_b = np.sqrt((b[indices]**2).mean(axis=1))
    ratio = boot_b/boot_a
    output.update(rmse_ratio=float(np.sqrt(np.mean(b*b))/np.sqrt(np.mean(a*a))),
                  rmse_ratio_lo95=float(np.quantile(ratio,.025)), rmse_ratio_hi95=float(np.quantile(ratio,.975)),
                  rmse_difference=float(np.sqrt(np.mean(b*b))-np.sqrt(np.mean(a*a))),
                  rmse_difference_lo95=float(np.quantile(boot_b-boot_a,.025)),
                  rmse_difference_hi95=float(np.quantile(boot_b-boot_a,.975)))
    return output


def build_report(run):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(REPO / "JBES_Submission/recovery_repaired_20261008/tools"))
    from recovery_common import sha256_file, digest
    manifest = json.loads((HERE / "manifest.json").read_text())
    if digest({k:v for k,v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("Manifest changed")
    for name, checksum in {**manifest["performance_sources"], **manifest["comparison"]["reference_sources"],
                           **manifest["comparison"]["reference_subsets"]}.items():
        if sha256_file(REPO / name) != checksum:
            raise ValueError("Frozen reporting/reference source changed: " + name)
    import numpy as np
    import pandas as pd
    from did_bcf_revision.metrics import compute_metrics, surface_metrics
    status = json.loads((run / "collection/status.json").read_text())
    if status["complete_units"] != 1890 or status["complete_job_budget_pairs"] != 540 or status["missing_units"] or status["conflicting_units"] or status["duplicate_units"]:
        raise ValueError("Complete integrity-verified paired fits are required; no failure filtering")
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(HERE / "reference_per_replication.csv")
    original["arm"] = ARMS[0]
    new = pd.read_csv(run / "collection/per_replication.csv")
    new["arm"] = new.budget_key.map({"earlier_budget_repaired": ARMS[1], "B3_warm2500_ret2000": ARMS[2]})
    if new.arm.isna().any():
        raise ValueError("Unexpected comparison budget")
    combined = pd.concat([original, new], ignore_index=True)
    wanted = {j["id"] for j in manifest["jobs"]}
    reference_keys = set(map(tuple, original[KEYS].to_numpy()))
    for arm, frame in combined.groupby("arm"):
        if frame.duplicated(KEYS).any() or set(frame.job_id) != wanted or set(map(tuple, frame[KEYS].to_numpy())) != reference_keys:
            raise ValueError("Incomplete, conflicting or unpaired target identities: " + arm)
        scalar = frame[frame.estimand_type != "CATT"]
        if not np.isfinite(scalar[["post_mean","sd","true","q025","q05","q95","q975"]].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite scalar metrics are never silently dropped")
        surface = frame[frame.estimand_type == "CATT"]
        if not np.isfinite(surface[["surf_rmse","surf_mae","surf_cover90","surf_cover95","surf_len90","surf_len95"]].to_numpy(dtype=float)).all():
            raise ValueError("Missing mandatory surface metrics")
        if not scalar.groupby(CELL+["estimand_type","estimand_id"], dropna=False).size().eq(30).all() or not surface.groupby(CELL, dropna=False).size().eq(30).all():
            raise ValueError("Not exactly 30 replications per target and cell")
    combined.to_csv(out / "paired_per_replication.csv", index=False)
    scalar_frames, surface_frames = [], []
    for arm, frame in combined.groupby("arm", sort=False):
        metrics = compute_metrics(frame)
        metrics["arm"] = arm
        for i, row in metrics.iterrows():
            group = frame[(frame.estimand_type == row.estimand_type) & (frame.estimand_id == row.estimand_id)
                          & (frame.setting == row.setting) & frame.N.eq(row.N) & frame.linearity_degree.eq(row.linearity_degree)]
            error = group.post_mean.to_numpy(dtype=float)-group.true.to_numpy(dtype=float)
            metrics.loc[i,"legacy_mcse_bias_estimate_sd"] = row.mcse_bias
            metrics.loc[i,"mcse_bias"] = float(error.std(ddof=1)/np.sqrt(len(error)))
            metrics.loc[i,"error_sd"] = float(error.std(ddof=1))
            metrics.loc[i,"post_sd_to_error_sd"] = row.avg_post_sd/float(error.std(ddof=1))
            for level in (90,95):
                count = int(round(row[f"cover{level}"]*30))
                lo, hi = wilson(count,30)
                metrics.loc[i,f"coverage{level}_count"] = count
                metrics.loc[i,f"coverage{level}_wilson_lo95"] = lo
                metrics.loc[i,f"coverage{level}_wilson_hi95"] = hi
        scalar_frames.append(metrics)
        surf = surface_metrics(frame)
        surf["arm"] = arm
        surface_frames.append(surf)
    scalar_metrics = pd.concat(scalar_frames, ignore_index=True)
    surfaces = pd.concat(surface_frames, ignore_index=True)
    scalar_metrics.to_csv(out / "scalar_metrics.csv", index=False)
    surfaces.to_csv(out / "surface_metrics.csv", index=False)
    contrasts = []
    for grouping, frame in combined.groupby(CELL+["estimand_type","estimand_id"], dropna=False, sort=True):
        identity = dict(zip(CELL+["estimand_type","estimand_id"], grouping))
        groups = {arm:g.sort_values("rep").reset_index(drop=True) for arm,g in frame.groupby("arm")}
        for a,b in ((ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])):
            left, right = groups[a], groups[b]
            if list(left.job_id) != list(right.job_id) or list(left.rep) != list(right.rep):
                raise ValueError("Incorrect replication pairing")
            if identity["estimand_type"] == "CATT" and not np.array_equal(left.surf_n.to_numpy(),right.surf_n.to_numpy()):
                raise ValueError("Paired surface row counts disagree")
            seed = int(hashlib.sha256(json.dumps(identity,sort_keys=True,default=int).encode()).hexdigest()[:16],16) ^ manifest["comparison"]["reporting_seed"]
            indices = np.random.default_rng(seed).integers(0,30,size=(manifest["comparison"]["bootstrap_resamples"],30))
            contrasts.append({**identity,"reference_arm":a,"comparison_arm":b,"n_pairs":30,
                              **paired_statistics(left,right,indices,identity["estimand_type"] == "CATT")})
    pd.DataFrame(contrasts).to_csv(out / "paired_metric_contrasts.csv", index=False)
    diagnostics = pd.read_csv(run / "collection/chain_diagnostics.csv")
    diagnostics["arm"] = diagnostics.budget_key.map({"earlier_budget_repaired": ARMS[1], "B3_warm2500_ret2000": ARMS[2]})
    old_diag = pd.read_csv(HERE / "reference_chain_diagnostics.csv")
    old_diag["arm"] = ARMS[0]
    diagnostics = pd.concat([old_diag,diagnostics],ignore_index=True)
    diagnostics.to_csv(out / "supporting_chain_diagnostics.csv",index=False)
    diagnostic_summary=[]
    for (arm,parameter), frame in diagnostics.groupby(["arm","parameter"]):
        record=dict(arm=arm, parameter=parameter, sequences=len(frame), median_rhat=float(frame.rhat.median()),
                    maximum_rhat=float(frame.rhat.max()), rhat_flags=int(frame.rhat.gt(1.01).sum()))
        if "bulk_ess" in frame and frame.bulk_ess.notna().any():
            record.update(min_bulk_ess=float(frame.bulk_ess.min()),bulk_ess_flags=int(frame.bulk_ess.lt(400).sum()))
        diagnostic_summary.append(record)
    pd.DataFrame(diagnostic_summary).to_csv(out / "diagnostic_summary.csv",index=False)
    att = scalar_metrics[scalar_metrics.estimand_type == "ATT"]
    lines=["# Thirty-replication paired paper-metric comparison", "",
           "All 270 selected datasets, 540 repaired fits and 1,890 new chains completed and passed output-integrity verification. The saved proper-prior reference uses the same 270 datasets; preparation verified their panel hashes. Every requested replication enters the metrics. Diagnostic flags did not select replications or draws.", "",
           "The reference is the saved unpatched proper-prior campaign. The two repaired arms use the earlier short budget and B3, respectively. Priors, architecture, targets and datasets are fixed. The reference interpreter differs from the local interpreter, so its comparison also changes runtime environment. The two new arms compare budgets within the same local repaired engine.", "",
           "These nine cells and 30 replications per cell support preliminary assessment. They do not cover all manuscript settings or prove empirical equivalence. Coverage counts and Wilson intervals are in scalar_metrics.csv; paired contrasts have dataset-level bootstrap intervals. No equivalence margin or automatic production decision was specified.", "",
           "| Setting | N | Degree | Arm | Bias | ATT RMSE | 95% inclusion | Mean 95% width |", "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in att.sort_values(["setting","N","linearity_degree","arm"]).itertuples():
        lines.append(f"| {row.setting} | {row.N:g} | {row.linearity_degree:g} | {row.arm} | {row.bias:.4f} | {row.rmse:.4f} | {int(row.coverage95_count)}/30 | {row.len95:.4f} |")
    lines += ["", "| Setting | N | Degree | Arm | Mean surface RMSE | Mean pointwise 95% coverage | Mean pointwise 95% width |", "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in surfaces.sort_values(["setting","N","linearity_degree","arm"]).itertuples():
        lines.append(f"| {row.setting} | {row.N:g} | {row.linearity_degree:g} | {row.arm} | {row.surf_rmse_mean:.4f} | {row.surf_cover95_mean:.3f} | {row.surf_len95_mean:.4f} |")
    lines += ["", "All scalar cohort/time GATT and event-time metrics, bias/RMSE/MAE/MAPE, 90%/95% coverage and widths, and size/power are retained in scalar_metrics.csv. Surface metrics remain conditional on the treated rows in each dataset. They do not establish transported population CATE accuracy.", "",
              "Supporting chain diagnostics and their summaries are retained separately. Their flags measure sampling agreement and precision, and do not establish causal bias. A performance resemblance also does not establish intended-posterior approximation or finite-chain consistency. The previous frozen full-state gate remains recorded as not passed.", "",
              "Detailed paired changes, uncertainty intervals and RMSE ratios: paired_metric_contrasts.csv. Raw matched summaries: paired_per_replication.csv. No full production campaign is launched by this report."]
    (out / "PERFORMANCE_REPORT.md").write_text("\n".join(lines)+"\n")
    provenance=dict(manifest_hash=manifest["manifest_hash"],requested_replications=30,selected_cells=9,new_fits=540,
                    new_chain_units=1890,diagnostic_filtering=False,bootstrap_resamples=5000,
                    report_files={p.name:sha256_file(p) for p in out.iterdir() if p.is_file() and p.name != "report_provenance.json"},
                    source_inputs={"reference_subset":sha256_file(HERE / "reference_per_replication.csv"),
                                   "new_per_replication":sha256_file(run / "collection/per_replication.csv"),
                                   "collection_status":sha256_file(run / "collection/status.json")})
    (out / "report_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(dict(report=str(out / "PERFORMANCE_REPORT.md"),scalar_metric_rows=len(scalar_metrics),
                         surface_metric_rows=len(surfaces),paired_contrast_rows=len(contrasts),completed_datasets=270),indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=HERE / "local_run")
    build_report(parser.parse_args().run_dir.resolve())
