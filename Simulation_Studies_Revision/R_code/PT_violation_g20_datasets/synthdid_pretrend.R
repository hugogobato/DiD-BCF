# Synthetic DiD PRE-TREND test for scenario "PT_violation_g20".
#
# The benchmark counterpart of did_bcf_revision/pretrend.py, emitting the SAME
# estimands in the SAME schema (method = "synthdid") on the SAME seeded panels.
#
# What the test is: the package's own documented pre-trend check
# -------------------------------------------------------------
# The synthdid vignette's section "Checking for pre-treatment parallel trends"
# says to run `plot(tau.hat, overlay = 1)` and look at how parallel the treated
# and synthetic-control trajectories are.  Reading `synthdid:::synthdid_plot`,
# what that draws is
#
#     obs.trajectory = omega.target %*% Y            (treated average)
#     syn.trajectory = omega.synth  %*% Y + offset   (omega-weighted controls)
#     offset         = overlay * ((omega.target - omega.synth) %*% Y %*% lambda)
#
# so the gap the reader is asked to eyeball is
#
#     gap(t) = [treated average](t) - [omega-weighted control average](t)
#
# up to a constant.  Referenced to k = -1 instead of to the lambda-weighted
# pre-treatment average, that is exactly Delta(k) of pretrend.py:
#
#     Delta_sdid(k) = gap(t_k) - gap(t_{-1})
#
# and the overlay constant cancels identically in the difference, so this is the
# package's own object with its intercept choice differenced out rather than a
# construction of ours.  Verified on PT_violation_g40 seed 0: -0.676 / -0.446 /
# -0.098 against a truth of -1.2 / -0.8 / -0.4.
#
# What the package does NOT give, and is constructed here
# ------------------------------------------------------
# Inference.  `plot()` is a picture: its `se.method` argument sizes the error bar
# on the single scalar effect estimate (`sqrt(vcov(est, method = se.method))`)
# and there is no standard error, statistic or p-value on the pre-period gaps.
# So the standard errors below are ours.  They follow the package's own
# convention exactly (`synthdid:::jackknife_se`): a delete-one-unit jackknife
# with omega and lambda held FIXED and omega renormalised after the drop.
# Applied to the whole Delta vector at once it also returns the cross-k
# covariance, so `slope` and `joint` are internally consistent rather than
# assuming the per-k estimates are independent.
#
# Note `se.method = 'placebo'` cannot be used on this design at all: these panels
# are ~54% treated and synthdid errors with "must have more controls than
# treated units to use the placebo se".  synthdid.R hit the same wall for the
# real estimate and uses the jackknife.
#
# Two properties worth reporting rather than hiding
# -------------------------------------------------
# 1. omega is chosen to match the treated group's pre-treatment path, so this
#    diagnostic hunts for a trend the estimator has actively fitted away.  It is
#    conservative by construction (the -0.676 above against a truth of -1.2).
# 2. The fitted time weights on these panels are lambda = 0, 0.128, 0.414, 0.458:
#    synthdid gives the EARLIEST pre-period zero weight, which is exactly where a
#    linear violation is largest.  That is a property of the picture the vignette
#    asks the reader to inspect.
#
# PRE_SUBC is a sample split, as for Callaway--Sant'Anna and did2s: the whole
# procedure re-run within each subgroup and differenced.  The arms are disjoint
# sets of units, so the covariances add.
library(synthdid)
library(progress)
sink("output_synthdid_pretrend.txt")
DGP <- "canonical"; SETTING <- "PT_violation_g20"; METHOD <- "synthdid"
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

# ---- Truth (identical to did_bcf_revision.pretrend.true_pretrend) -----------
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

# ---- The overlay-plot gap, with a jackknife covariance ----------------------
# Returns Delta(k) for every pre-period k != -1, and its delete-one-unit
# jackknife covariance.  omega/lambda are held fixed across jackknife replicates
# and omega is renormalised after a drop, which is what synthdid:::jackknife_se
# does for the scalar estimate.
fit_pre <- function(d) {
  panel <- data.frame(unit_id = d$unit_id, time = d$time, res_Y = d$res_Y,
                      treated = d$eventually_treated * d$post_treatment)
  setup <- tryCatch(panel.matrices(panel), error = function(e) NULL)
  if (is.null(setup)) return(NULL)
  Y <- setup$Y; N0 <- setup$N0; T0 <- setup$T0
  N1 <- nrow(Y) - N0
  if (N0 < 5 || N1 < 5 || T0 < 3) return(NULL)
  tau <- tryCatch(synthdid_estimate(Y, N0, T0), error = function(e) NULL)
  if (is.null(tau)) return(NULL)
  omega <- attr(tau, "weights")$omega
  if (is.null(omega) || sum(omega != 0) < 2) return(NULL)

  # Column T0 is event time -1; column T0 + (k + 1) is event time k, so column 1
  # is the earliest pre-period, k = -T0.
  ks <- (-T0):(-2)
  cols <- T0 + (ks + 1)
  ok <- cols >= 1
  ks <- ks[ok]; cols <- cols[ok]
  if (length(ks) < 2) return(NULL)

  delta_of <- function(trt_avg, syn) {
    gap <- trt_avg - syn
    gap[cols] - gap[T0]
  }
  trt_avg <- colMeans(Y[(N0 + 1):nrow(Y), , drop = FALSE])
  syn <- as.numeric(omega %*% Y[1:N0, , drop = FALSE])
  est <- delta_of(trt_avg, syn)

  # Delete-one-unit jackknife over BOTH arms, as synthdid does.
  reps <- list()
  for (j in 1:N0) {
    if (omega[j] >= 1 - 1e-10) next
    syn_j <- (syn - omega[j] * Y[j, ]) / (1 - omega[j])
    reps[[length(reps) + 1]] <- delta_of(trt_avg, syn_j)
  }
  if (N1 > 1) for (j in (N0 + 1):nrow(Y)) {
    trt_j <- (N1 * trt_avg - Y[j, ]) / (N1 - 1)
    reps[[length(reps) + 1]] <- delta_of(trt_j, syn)
  }
  n <- length(reps)
  if (n < 3) return(NULL)
  R <- do.call(rbind, reps)
  Rc <- sweep(R, 2, colMeans(R))
  V <- ((n - 1) / n) * (t(Rc) %*% Rc)
  se <- sqrt(pmax(diag(V), 0))
  if (any(!is.finite(se)) || all(se == 0)) return(NULL)
  list(kk = as.integer(ks), att = as.numeric(est), se = se, V = V)
}

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
  # Residualise on the covariates exactly as synthdid.R does for the real fit.
  ols <- lm(Y ~ X_1 + X_2 + X_3 + X_4 + X_5, data = d)
  d$res_Y <- d$Y - predict(ols, newdata = d)

  agg <- fit_pre(d); if (is.null(agg)) return(NULL)
  rows <- pre_rows(agg$kk, agg$att, agg$se, agg$V, tr$diff)
  stat <- tryCatch(as.numeric(t(agg$att) %*% solve(agg$V) %*% agg$att),
                   error = function(e) NA_real_)
  if (is.finite(stat))
    rows[[length(rows) + 1]] <- new_decision("PRE", "joint",
      0.5 * pchisq(stat, df = length(agg$att), lower.tail = FALSE), tr$diff)

  keep <- list("X1=1" = d$X_1 > 0.5, "X1=0" = d$X_1 <= 0.5,
               "X2=high" = d$X_2 >= tr$cut[2], "X2=low" = d$X_2 <= tr$cut[1])
  fits <- list()
  for (s in list(c("X1", "X1=1", "X1=0"), c("X2", "X2=high", "X2=low"))) {
    for (lab in s[2:3]) {
      if (is.null(fits[[lab]])) {
        sd_ <- d[keep[[lab]], ]
        fits[[lab]] <- if (length(unique(sd_$unit_id)) >= 20) fit_pre(sd_) else NULL
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
