# Callaway--Sant'Anna PRE-TREND test for scenario "PT_violation_g20".
#
# The benchmark counterpart of did_bcf_revision/pretrend.py, emitting the SAME
# estimands in the SAME schema (method = "did_dr"), so the Bayesian diagnostic
# and this one are scored by did_bcf_revision.metrics.compute_metrics on
# identical definitions and identical seeded panels.
#
# What the test is
# ----------------
# `att_gt(..., base_period = "universal")` reports, for every pre-treatment
# period t < g-1,
#
#     ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | never treated]
#
# doubly-robustly *conditional on X*.  That is exactly Delta(k) of pretrend.py
# with k = t - g and reference k = -1, verified against the DGP's realised
# violation slope (delta = 0.4 gives -1.18 / -0.79 / -0.21 against a truth of
# -1.2 / -0.8 / -0.4).  So this is the strongest form of the comparator: a
# covariate-conditional placebo, not a marginal event study.
#
# Three inference objects are reported, each the natural one for its statistic:
#   * per-k rows use `did`'s own SE (its default multiplier bootstrap);
#   * `slope` and the subgroup contrasts use the analytical covariance rebuilt
#     from `out$inffunc` (Cov(att) = t(inf) %*% inf / n^2, verified to reproduce
#     `V_analytical` to machine precision) -- the pre-period coefficients share
#     the g-1 base period, so their covariances are positive and a diagonal-only
#     slope SE would be anti-conservative (the same defect fixed for TWFE in
#     pretrend.py);
#   * `joint` is `did`'s own pre-test `Wpval`, halved so that p < 0.025 is the
#     two-sided 5% test the metrics layer applies to every other row.
#
# PRE_SUBC (the subgroup contrast) is obtained by SAMPLE SPLITTING: att_gt is
# re-fitted within each subgroup and the slopes differenced.  The two arms are
# disjoint sets of units, so the variances add.  This is deliberately generous
# to the comparator -- the manuscript's claim is that the contrast is an object
# a *marginal* event study cannot form, and a practitioner with a binary,
# pre-specified moderator can always split the sample, so the honest comparison
# gives them that.  X_1 is dropped from xformla inside an X1 split (it is
# constant there, hence collinear with the intercept).
library(did)
library(progress)
sink("output_did_dr_pretrend.txt")
DGP <- "canonical"; SETTING <- "PT_violation_g20"; METHOD <- "did_dr"
REF_K <- -1L

# ---- DiD-BCF-schema summary emission (shared helpers) -----------------------
SCHEMA <- c("dgp","setting","linearity_degree","N","rep","estimand_type",
            "estimand_id","g","t","k","method","post_mean","sd","q025","q05",
            "q95","q975","p_bayes","surf_rmse","surf_mae","surf_n","surf_mape",
            "surf_cover95","surf_len95","surf_cover90","surf_len90","true")
Z95 <- 1.959964; Z90 <- 1.644854
wald_tail <- function(est, se) if (is.finite(est) && is.finite(se) && se > 0)
  pnorm(-abs(est / se)) else NA_real_

new_scalar <- function(estimand_type, estimand_id, g, t, k, est, se, truth) {
  data.frame(estimand_type = estimand_type, estimand_id = estimand_id,
             g = g, t = t, k = k, post_mean = est, sd = se,
             q025 = est - Z95 * se, q05 = est - Z90 * se,
             q95 = est + Z90 * se, q975 = est + Z95 * se,
             p_bayes = wald_tail(est, se),
             surf_rmse = NA_real_, surf_mae = NA_real_, surf_n = NA_integer_,
             surf_mape = NA_real_, surf_cover95 = NA_real_, surf_len95 = NA_real_,
             surf_cover90 = NA_real_, surf_len90 = NA_real_, true = truth,
             stringsAsFactors = FALSE)
}

# A per-replication DECISION rule (any-k, Bonferroni any-k, joint Wald).  It has
# no point estimate and no interval by construction; metrics.py routes these to
# the tail-only aggregator, which turns `true == 0` into a size and `true != 0`
# into a detection rate.
new_decision <- function(estimand_type, estimand_id, p, truth) {
  data.frame(estimand_type = estimand_type, estimand_id = estimand_id,
             g = NA_real_, t = NA_real_, k = NA_real_,
             post_mean = NA_real_, sd = NA_real_, q025 = NA_real_, q05 = NA_real_,
             q95 = NA_real_, q975 = NA_real_, p_bayes = p,
             surf_rmse = NA_real_, surf_mae = NA_real_, surf_n = NA_integer_,
             surf_mape = NA_real_, surf_cover95 = NA_real_, surf_len95 = NA_real_,
             surf_cover90 = NA_real_, surf_len90 = NA_real_, true = truth,
             stringsAsFactors = FALSE)
}

finalize <- function(rows, rep, N, dgp, setting, lin_degree, method) {
  if (!length(rows)) return(NULL)
  df <- do.call(rbind, rows)
  df$dgp <- dgp; df$setting <- setting; df$linearity_degree <- lin_degree
  df$N <- N; df$rep <- rep; df$method <- method
  df[, SCHEMA]
}

# ---- Truth ------------------------------------------------------------------
# Mirrors did_bcf_revision.pretrend.true_pretrend: the treated-minus-control
# difference in the per-unit violation slope `pt_slope` (a truth column carried
# by the exported panel, like CATE).  Delta(k) = diff * (k - REF_K); the `slope`
# estimand and every decision rule carry `diff` itself, which is what makes
# reject05 a size when the violation is zero and a detection rate otherwise.
# The X_2 subgroup boundaries are computed on the TREATED units and applied to
# the controls, so both arms are compared over the same region of X -- exactly
# pretrend._sel_on.
truth_of <- function(d) {
  u <- d[!duplicated(d$unit_id), ]
  ever <- u$eventually_treated == 1
  if (all(ever) || !any(ever)) return(NULL)
  ut <- u[ever, ]
  q1 <- as.numeric(quantile(ut$X_2, 1 / 3)); q2 <- as.numeric(quantile(ut$X_2, 2 / 3))
  sel <- list("X1=0" = u$X_1 <= 0.5, "X1=1" = u$X_1 > 0.5,
              "X2=low" = u$X_2 <= q1, "X2=high" = u$X_2 >= q2)
  sub <- vapply(sel, function(m) {
    if (sum(m & ever) < 5 || sum(m & !ever) < 5) return(NA_real_)
    mean(u$pt_slope[m & ever]) - mean(u$pt_slope[m & !ever])
  }, numeric(1))
  list(diff = mean(u$pt_slope[ever]) - mean(u$pt_slope[!ever]),
       contrast = c(X1 = sub[["X1=1"]] - sub[["X1=0"]],
                    X2 = sub[["X2=high"]] - sub[["X2=low"]]),
       cut = c(q1, q2))
}

# ---- The pre-trend block, shared by the aggregate test and the contrasts -----
# Emits per-k Delta(k), the implied differential slope, and the two any-k
# decision rules, in the estimand ids pretrend.py uses.
pre_rows <- function(kk, est, se, V, truth_slope, type = "PRE", prefix = "") {
  rows <- list()
  for (i in seq_along(kk)) {
    rows[[length(rows) + 1]] <- new_scalar(
      type, sprintf("%sk=%d", prefix, kk[i]), NA_real_, NA_real_,
      as.integer(kk[i]), est[i], se[i], truth_slope * (kk[i] - REF_K))
  }
  w <- kk - REF_K
  slope <- sum(w * est) / sum(w * w)
  slope_se <- if (!is.null(V))
    sqrt(max(as.numeric(t(w) %*% V %*% w), 0)) / sum(w * w)
  else sqrt(sum((w * se) ^ 2)) / sum(w * w)
  rows[[length(rows) + 1]] <- new_scalar(type, paste0(prefix, "slope"),
    NA_real_, NA_real_, NA_real_, slope, slope_se, truth_slope)
  tails <- mapply(wald_tail, est, se)
  p_min <- if (any(is.finite(tails))) min(tails, na.rm = TRUE) else NA_real_
  rows[[length(rows) + 1]] <- new_decision(type, paste0(prefix, "any"),
                                           p_min, truth_slope)
  rows[[length(rows) + 1]] <- new_decision(type, paste0(prefix, "any_bonf"),
    min(1, length(tails) * p_min), truth_slope)
  rows
}

# ---- One att_gt fit -> the pre-treatment cells and their covariance ----------
XF_ALL <- ~ X_1 + X_2 + X_3 + X_4 + X_5
XF_NO1 <- ~ X_2 + X_3 + X_4 + X_5            # X_1 is constant inside an X1 split

fit_pre <- function(d, xformla = XF_ALL) {
  out <- tryCatch(
    att_gt(yname = "Y", tname = "time", idname = "unit_id",
           gname = "first_treat_period", xformla = xformla, data = d,
           est_method = "dr", control_group = "nevertreated",
           base_period = "universal", print_details = FALSE, pl = FALSE,
           cores = 1),
    error = function(e) NULL)
  if (is.null(out)) return(NULL)
  kk <- out$t - out$group
  sel <- which(kk < REF_K & is.finite(out$att) & is.finite(out$se) & out$se > 0)
  if (length(sel) < 2) return(NULL)
  inf <- as.matrix(out$inffunc)
  V <- (t(inf) %*% inf) / (out$n ^ 2)        # Cov(att), == V_analytical / n
  list(kk = as.integer(kk[sel]), att = out$att[sel], se = out$se[sel],
       V = V[sel, sel, drop = FALSE], Wpval = out$Wpval)
}

# ---- Sample-split subgroup contrast -----------------------------------------
contrast_rows <- function(hi, lo, label, truth_c) {
  if (is.null(hi) || is.null(lo) || !is.finite(truth_c)) return(list())
  ks <- intersect(hi$kk, lo$kk)
  if (length(ks) < 2) return(list())
  ih <- match(ks, hi$kk); il <- match(ks, lo$kk)
  pre_rows(ks, hi$att[ih] - lo$att[il],
           sqrt(hi$se[ih] ^ 2 + lo$se[il] ^ 2),
           hi$V[ih, ih, drop = FALSE] + lo$V[il, il, drop = FALSE],
           truth_c, type = "PRE_SUBC", prefix = paste0(label, "_"))
}

run_rep <- function(d) {
  tr <- truth_of(d); if (is.null(tr)) return(NULL)
  agg <- fit_pre(d); if (is.null(agg)) return(NULL)
  rows <- pre_rows(agg$kk, agg$att, agg$se, agg$V, tr$diff)
  # `did`'s own pre-test, halved onto the one-sided scale metrics.py reads.
  if (is.finite(agg$Wpval))
    rows[[length(rows) + 1]] <- new_decision("PRE", "joint",
                                             0.5 * agg$Wpval, tr$diff)
  splits <- list(
    c("X1", "X1=1", "X1=0"),
    c("X2", "X2=high", "X2=low"))
  keep <- list("X1=1" = d$X_1 > 0.5, "X1=0" = d$X_1 <= 0.5,
               "X2=high" = d$X_2 >= tr$cut[2], "X2=low" = d$X_2 <= tr$cut[1])
  fits <- list()
  for (s in splits) {
    xf <- if (s[1] == "X1") XF_NO1 else XF_ALL
    for (lab in s[2:3]) {
      if (is.null(fits[[lab]])) {
        sd_ <- d[keep[[lab]], ]
        fits[[lab]] <- if (length(unique(sd_$unit_id)) >= 20)
          fit_pre(sd_, xf) else NULL
      }
    }
    rows <- c(rows, contrast_rows(fits[[s[2]]], fits[[s[3]]], s[1],
                                  tr$contrast[[s[1]]]))
  }
  rows
}

lin_folders <- c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")
options(warn = -1)
for (lin in lin_folders) {
  if (!dir.exists(lin)) { cat("no", lin, "- skip\n"); next }
  lin_degree <- as.integer(sub(".*=", "", lin))
  files <- list.files(lin, pattern = "^iteration_", full.names = TRUE)
  files <- files[grepl("csv$", files)]
  files <- files[order(as.integer(gsub("[^0-9]", "", basename(files))))]
  if (length(files) == 0) { cat("No files in", lin, "- skipping\n"); next }
  all_rows <- list()
  pb <- progress_bar$new(total = length(files), format = paste0(lin, " [:bar] :current/:total"))
  for (ii in seq_along(files)) {
    pb$tick(); rep <- ii - 1
    d <- read.csv(files[ii])
    d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
    N <- length(unique(d$unit_id))
    rows <- tryCatch(run_rep(d), error = function(e) NULL)
    if (is.null(rows) || !length(rows)) next
    fr <- finalize(rows, rep, N, DGP, SETTING, lin_degree, METHOD)
    if (!is.null(fr)) all_rows[[length(all_rows) + 1]] <- fr
  }
  res <- if (length(all_rows)) do.call(rbind, all_rows) else data.frame()
  fn <- sprintf("summaries_%s_%s_lin_%d.csv", METHOD, SETTING, lin_degree)
  write.csv(res, fn, row.names = FALSE, na = ""); cat("\nwrote", fn, "(", nrow(res), "rows )\n")
}
sink()
