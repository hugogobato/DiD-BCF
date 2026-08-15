# DoubleML doubly-robust PRE-TREND test (random-forest nuisances) for scenario
# "PT_violation_g40".
#
# The benchmark counterpart of did_bcf_revision/pretrend.py, emitting the SAME
# estimands in the SAME schema (method = "doubleml") on the SAME seeded panels.
# Identical to did_dr_pretrend.R except that Callaway--Sant'Anna's `att_gt` is
# plugged with DoubleML's ATTE estimator (Chang 2020) instead of the analytical
# doubly-robust one, so the nuisances are random forests rather than
# logit/OLS -- this is the ML-conditional pre-trend test, the closest
# frequentist analogue of the Bayesian diagnostic.
#
# `base_period = "universal"` makes ATT(g,t) at t < g-1 exactly Delta(k) of
# pretrend.py; see did_dr_pretrend.R for the estimand and the inference objects.
#
# **No subgroup contrast (PRE_SUBC).** The contrast needs four extra fits per
# replication, and this estimator costs ~20-90 s per fit against ~0.15 s for
# Callaway--Sant'Anna, which is why it lives on Colab at all. Callaway--Sant'Anna
# and grf-DiD carry the sample-split contrast for the frequentist side.
library(did)
library(progress)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(lgr)
lgr::get_logger("mlr3")$set_threshold("fatal")

# Plug DoubleML's ATTE estimator into att_gt as a custom est_method (Chang 2020).
# `att_gt` hands it (y1, y0) = the outcome pair for the (g,t) cell being
# estimated, which under base_period="universal" is (Y_t, Y_{g-1}) for the
# pre-treatment placebos too, so no change is needed for the pre-trend use.
doubleml_did_rf <- function(y1, y0, D, covariates,
                            ml_g = lrn("regr.ranger", num.trees = 500),
                            ml_m = lrn("classif.ranger", num.trees = 500),
                            n_folds = 5, n_rep = 1, ...) {
  delta_y <- y1 - y0
  dml_data <- DoubleML::double_ml_data_from_matrix(X = covariates, y = delta_y, d = D)
  dml_obj <- DoubleML::DoubleMLIRM$new(dml_data, ml_g = ml_g, ml_m = ml_m,
                                       score = "ATTE", n_folds = n_folds)
  dml_obj$fit()
  list(ATT = dml_obj$coef[1], att.inf.func = dml_obj$psi[, 1, 1])
}

sink("output_DoubleML_pretrend.txt")
DGP <- "canonical"; SETTING <- "PT_violation_g40"; METHOD <- "doubleml"
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

fit_pre <- function(d, xformla = XF_ALL) {
  out <- tryCatch(
    att_gt(yname = "Y", tname = "time", idname = "unit_id",
           gname = "first_treat_period", xformla = xformla, data = d,
           est_method = doubleml_did_rf, control_group = "nevertreated",
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

run_rep <- function(d) {
  tr <- truth_of(d); if (is.null(tr)) return(NULL)
  agg <- fit_pre(d); if (is.null(agg)) return(NULL)
  rows <- pre_rows(agg$kk, agg$att, agg$se, agg$V, tr$diff)
  # `did`'s own pre-test, halved onto the one-sided scale metrics.py reads.
  if (is.finite(agg$Wpval))
    rows[[length(rows) + 1]] <- new_decision("PRE", "joint",
                                             0.5 * agg$Wpval, tr$diff)
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
