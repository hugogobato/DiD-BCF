# Gardner two-stage (did2s) PRE-TREND test for scenario "PT_violation_het20".
#
# The benchmark counterpart of did_bcf_revision/pretrend.py, emitting the SAME
# estimands in the SAME schema (method = "did2s") on the SAME seeded panels.
#
# What the test is
# ----------------
# The pre-period coefficients of the documented did2s event study, which is what
# the package's own vignette plots and what a practitioner reads off it.  It is
# an **unconditional** test: X_1..X_5 are time-invariant and therefore absorbed
# by the unit fixed effect, so the first stage is FE-only (see did2s.R for why
# the `event_study()` covariate path is not usable here).
#
# One deviation from did2s.R, and it is necessary rather than cosmetic.  The
# second stage is `resid ~ 0 + i(rel, ...)`, whose coefficients are *level* means
# of the first-stage residual within each event time, not differences from the
# reference period.  did2s.R uses `ref = c(-1, Inf)`, which leaves k = -1
# unestimated and pools it with the never-treated -- fine for the post-treatment
# rows, where the level IS the effect, but wrong here: Delta(k) is defined
# against k = -1, and regressing un-recentred levels through the origin mixes the
# pre-trend with an arbitrary level.  So the reference is `ref = Inf` alone,
# k = -1 is estimated, and Delta(k) = beta_k - beta_{-1} is formed with the
# matching covariance A V A'.  Estimates are unchanged; only the recentring is.
#
# A caveat that belongs in the write-up rather than in a footnote
# ---------------------------------------------------------------
# did2s imputes Y(0) from the *untreated* observations, which include the treated
# units' own pre-treatment rows.  The periods being tested are therefore inside
# the first-stage estimation sample and their residuals are in-sample, so the
# pre-period contrasts are attenuated: measured on PT_violation_g40 seed 0,
# Delta(k) comes out -0.469 / -0.309 / -0.071 against a truth of
# -1.2 / -0.8 / -0.4, an implied slope of 0.150 against 0.400.
# Borusyak-Jaravel-Spiess prescribe re-fitting the first stage with the tested
# periods excluded.  That is **not implementable here**: with four pre-periods,
# excluding k = -4,-3,-2 leaves each treated unit a single first-stage row, and
# fixest drops those units as fixed-effect singletons -- did2s then fails with
# "not a single explanatory variable is different from 0" (verified).  So the
# attenuated test is the one this estimator actually affords on this design, and
# it is reported as such.
#
# PRE_SUBC (the subgroup contrast) is obtained by SAMPLE SPLITTING: the event
# study is re-fitted within each subgroup and the slopes differenced.  The arms
# are disjoint sets of units, so the variances add.
library(did2s)
library(broom)
library(progress)
sink("output_did2s_pretrend.txt")
DGP <- "canonical"; SETTING <- "PT_violation_het20"; METHOD <- "did2s"
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

# ---- One did2s event study -> Delta(k) and its covariance --------------------
# `ref = Inf` keeps k = -1 estimable so the pre-period levels can be recentred on
# it (see the header).  Delta = A beta with A = [I, -1], Cov(Delta) = A V A'.
fit_pre <- function(d) {
  d$rel <- ifelse(d$eventually_treated == 1, d$event_time, Inf)
  es <- tryCatch(
    did2s(d, yname = "Y", first_stage = ~ 0 | unit_id + time,
          second_stage = ~ i(rel, ref = c(Inf)), treatment = "D",
          cluster_var = "unit_id"),
    error = function(e) NULL)
  if (is.null(es)) return(NULL)
  V <- tryCatch(as.matrix(vcov(es)), error = function(e) NULL)
  b <- tryCatch(coef(es), error = function(e) NULL)
  if (is.null(V) || is.null(b)) return(NULL)
  kk <- suppressWarnings(as.integer(sub(".*rel::", "", names(b))))
  iref <- which(is.finite(kk) & kk == REF_K)
  ipre <- which(is.finite(kk) & kk < REF_K & is.finite(b))
  if (length(iref) != 1 || length(ipre) < 2) return(NULL)
  idx <- c(ipre, iref)
  A <- cbind(diag(length(ipre)), -1)
  Vd <- A %*% V[idx, idx, drop = FALSE] %*% t(A)
  se <- sqrt(pmax(diag(Vd), 0))
  ok <- is.finite(se) & se > 0
  if (sum(ok) < 2) return(NULL)
  list(kk = kk[ipre][ok], att = as.numeric(b[ipre] - b[iref])[ok],
       se = se[ok], V = Vd[ok, ok, drop = FALSE])
}

contrast_rows <- function(hi, lo, label, truth_c) {
  if (is.null(hi) || is.null(lo) || !is.finite(truth_c)) return(list())
  ks <- intersect(hi$kk, lo$kk)
  if (length(ks) < 2) return(list())
  ih <- match(ks, hi$kk); il <- match(ks, lo$kk)
  V <- if (!is.null(hi$V) && !is.null(lo$V))
    hi$V[ih, ih, drop = FALSE] + lo$V[il, il, drop = FALSE] else NULL
  pre_rows(ks, hi$att[ih] - lo$att[il],
           sqrt(hi$se[ih] ^ 2 + lo$se[il] ^ 2), V, truth_c,
           type = "PRE_SUBC", prefix = paste0(label, "_"))
}

run_rep <- function(d) {
  tr <- truth_of(d); if (is.null(tr)) return(NULL)
  agg <- fit_pre(d); if (is.null(agg)) return(NULL)
  rows <- pre_rows(agg$kk, agg$att, agg$se, agg$V, tr$diff)
  # Joint Wald pre-trends test, on the same half-tail scale as every other row.
  if (!is.null(agg$V)) {
    stat <- tryCatch(as.numeric(t(agg$att) %*% solve(agg$V) %*% agg$att),
                     error = function(e) NA_real_)
    if (is.finite(stat))
      rows[[length(rows) + 1]] <- new_decision("PRE", "joint",
        0.5 * pchisq(stat, df = length(agg$att), lower.tail = FALSE), tr$diff)
  }
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
