# grf-DiD (Wang 2022 stand-in) PRE-TREND test.
#
# The benchmark counterpart of did_bcf_revision/pretrend.py, emitting the SAME
# estimands in the SAME schema (method = "wang") on the SAME seeded panels, so
# both are scored by did_bcf_revision.metrics.compute_metrics on identical
# definitions.
#
# Is this a "native" pre-trend test?  Partly -- state it plainly in the write-up.
# grf has no DiD module and no pre-trend test; wang_grf.R is a *recipe* (the
# "readily available grf causal-forest DiD" of R3.1.1): difference the outcome
# against the last clean pre-period, then run grf::causal_forest on
# (X, dY, W) with never-treated controls.  The placebo below is that identical
# recipe with a PRE-treatment period substituted for the post period -- no
# estimator logic changes, only which two periods are differenced, exactly as
# Callaway--Sant'Anna's pre-test is its ATT(g,t) evaluated at t < g-1.  What IS
# constructed here is (a) the slope forest and (b) the subgroup contrast; both
# are flagged below.  Note the Bayesian diagnostic is no more native: it needs a
# different model (Z = ever-treated), where this needs only a different period.
#
# What it estimates
# -----------------
#     Delta(k) = E[ Y_k - Y_{-1} | X, treated ] - E[ Y_k - Y_{-1} | X, control ]
#
# averaged over the treated units -- exactly pretrend.py's Delta(k), conditional
# on X, with the conditioning done by a causal forest instead of a BART pair.
# This is the closest frequentist analogue of the Bayesian diagnostic in the
# suite, and the fair comparison for the ``PT_violation_het*`` scenarios.
#
# CONSTRUCTED (1): the slope forest.  The headline scalar is the differential
# pre-trend slope, the least-squares slope of Delta(k) on (k + 1) through the
# origin.  Rather than fit three forests and then need their cross-k covariance
# (which grf does not hand back), that linear combination is moved INSIDE the
# forest: the outcome is the per-unit statistic
#
#     S_i = sum_k (k+1) * (Y_{i,k} - Y_{i,-1}) / sum_k (k+1)^2
#
# so one causal forest returns the slope and its standard error directly.  This
# is a reparameterisation of the same estimand, not a different one.
#
# CONSTRUCTED (2): PRE_SUBC, the contrast between subgroup pre-trends, from
# grf's own `average_treatment_effect(..., subset=)` on the SAME forest -- the
# strongest form of the comparator, since the forest learns the heterogeneity on
# the full sample rather than on half of it (Callaway--Sant'Anna and did2s have
# to sample-split; grf does not).  The two subsets are disjoint, and their
# variances are added; the residual cross-subset covariance runs through the
# shared forest fit and is second order under grf's honesty, which is worth a
# sentence in the write-up rather than a silent assumption.
#
# No `joint` row: grf returns no covariance across separately fitted per-k
# forests, so the Wald pre-test the other two report has no counterpart here.
#
# Reads linearity_degree=*/ under the CWD and writes
#   summaries_wang_<SETTING>_lin_<d>.csv
if (!requireNamespace("grf", quietly = TRUE))
  stop("package 'grf' is not installed -- re-run the install cell", call. = FALSE)
suppressMessages(library(grf))
sink("output_wang_pretrend.txt")
DGP <- "canonical"
SETTING <- "PT_violation_g10"   # rewritten per folder by scaffold_suite.py (PT_R_FILES)
# The Colab notebook has no per-folder copy, so it passes them as arguments
# instead; run_r_benchmarks.py calls `Rscript wang_pretrend.R` with none.
ARGS <- commandArgs(TRUE)
if (length(ARGS) >= 1) DGP <- ARGS[1]
if (length(ARGS) >= 2) SETTING <- ARGS[2]
METHOD <- "wang"
N_TREES <- if (length(ARGS) >= 3) as.integer(ARGS[3]) else 2000L
REPS <- if (length(ARGS) >= 4) as.integer(ARGS[4]) else 200L
# grf grabs every core by default, which thrashes when the runner has one
# process per scenario in flight.  0 (the default) keeps grf's own behaviour.
GRF_THREADS <- suppressWarnings(as.integer(Sys.getenv("GRF_THREADS", "0")))
N_THREADS <- if (is.finite(GRF_THREADS) && GRF_THREADS > 0L) GRF_THREADS else NULL
REF_K <- -1L

SCHEMA <- c("dgp","setting","linearity_degree","N","rep","estimand_type",
            "estimand_id","g","t","k","method","post_mean","sd","q025","q05",
            "q95","q975","p_bayes","surf_rmse","surf_mae","surf_n","surf_mape",
            "surf_cover95","surf_len95","surf_cover90","surf_len90","true")
Z95 <- 1.959964; Z90 <- 1.644854
XN <- paste0("X_", 1:5)
wald_tail <- function(est, se) if (is.finite(est) && is.finite(se) && se > 0)
  pnorm(-abs(est / se)) else NA_real_

new_scalar <- function(etype, eid, g, t, k, est, se, truth) {
  data.frame(estimand_type = etype, estimand_id = eid, g = g, t = t, k = k,
             post_mean = est, sd = se, q025 = est - Z95 * se, q05 = est - Z90 * se,
             q95 = est + Z90 * se, q975 = est + Z95 * se, p_bayes = wald_tail(est, se),
             surf_rmse = NA_real_, surf_mae = NA_real_, surf_n = NA_integer_,
             surf_mape = NA_real_, surf_cover95 = NA_real_, surf_len95 = NA_real_,
             surf_cover90 = NA_real_, surf_len90 = NA_real_, true = truth,
             stringsAsFactors = FALSE)
}

new_decision <- function(etype, eid, p, truth) {
  data.frame(estimand_type = etype, estimand_id = eid,
             g = NA_real_, t = NA_real_, k = NA_real_,
             post_mean = NA_real_, sd = NA_real_, q025 = NA_real_, q05 = NA_real_,
             q95 = NA_real_, q975 = NA_real_, p_bayes = p,
             surf_rmse = NA_real_, surf_mae = NA_real_, surf_n = NA_integer_,
             surf_mape = NA_real_, surf_cover95 = NA_real_, surf_len95 = NA_real_,
             surf_cover90 = NA_real_, surf_len90 = NA_real_, true = truth,
             stringsAsFactors = FALSE)
}

finalize <- function(rows, rep, N, lin_degree) {
  if (!length(rows)) return(NULL)
  df <- do.call(rbind, rows)
  df$dgp <- DGP; df$setting <- SETTING; df$linearity_degree <- lin_degree
  df$N <- N; df$rep <- rep; df$method <- METHOD
  df[, SCHEMA]
}

# ---- Truth (identical to did_bcf_revision.pretrend.true_pretrend) -----------
# X_2 subgroup boundaries are computed on the TREATED units and applied to the
# controls, so both arms are compared over the same region of X (pretrend._sel_on).
truth_of <- function(u) {
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

# grf ATT over a subset of the treated units, as (estimate, std.err).
att_on <- function(cf, subset = NULL) {
  a <- tryCatch(average_treatment_effect(cf, target.sample = "treated",
                                         subset = subset),
                error = function(e) NULL)
  if (is.null(a)) return(c(NA_real_, NA_real_))
  c(as.numeric(a["estimate"]), as.numeric(a["std.err"]))
}

run_rep <- function(d) {
  d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
  # One row per unit: the covariates, the group, and the pre-period outcomes.
  u <- d[!duplicated(d$unit_id), c("unit_id", "eventually_treated", XN, "pt_slope")]
  u <- u[order(u$unit_id), ]
  tr <- truth_of(u); if (is.null(tr)) return(NULL)

  treated_pre <- d[d$eventually_treated == 1 & d$event_time < 0, ]
  if (!nrow(treated_pre)) return(NULL)
  ref_t <- max(treated_pre$time[treated_pre$event_time == REF_K])
  ks <- sort(unique(treated_pre$event_time[treated_pre$event_time != REF_K]))
  if (length(ks) < 2) return(NULL)
  # Event time is only defined for the treated; the never-treated contribute the
  # same CALENDAR periods, which is what the difference is taken over.
  t_of_k <- setNames(ref_t + (ks - REF_K), as.character(ks))

  wide <- function(tt) {
    z <- d[d$time == tt, c("unit_id", "Y")]
    z$Y[match(u$unit_id, z$unit_id)]
  }
  y_ref <- wide(ref_t)
  dY <- lapply(ks, function(k) wide(t_of_k[[as.character(k)]]) - y_ref)
  names(dY) <- as.character(ks)
  if (any(!is.finite(unlist(dY))) || any(!is.finite(y_ref))) return(NULL)

  X <- as.matrix(u[, XN])
  W <- as.numeric(u$eventually_treated == 1)
  if (sum(W) < 10 || sum(1 - W) < 10) return(NULL)
  subs <- list("X1=1" = u$X_1 > 0.5, "X1=0" = u$X_1 <= 0.5,
               "X2=high" = u$X_2 >= tr$cut[2], "X2=low" = u$X_2 <= tr$cut[1])

  forest <- function(y) tryCatch(
    causal_forest(X = X, Y = y, W = W, num.trees = N_TREES, seed = 1L,
                  num.threads = N_THREADS),
    error = function(e) NULL)

  rows <- list(); tails <- c(); ctails <- list(X1 = c(), X2 = c())
  for (k in ks) {
    cf <- forest(dY[[as.character(k)]]); if (is.null(cf)) next
    a <- att_on(cf)
    rows[[length(rows) + 1]] <- new_scalar("PRE", sprintf("k=%d", k),
      NA_real_, NA_real_, as.integer(k), a[1], a[2], tr$diff * (k - REF_K))
    tails <- c(tails, wald_tail(a[1], a[2]))
    for (s in list(c("X1", "X1=1", "X1=0"), c("X2", "X2=high", "X2=low"))) {
      hi <- att_on(cf, subs[[s[2]]]); lo <- att_on(cf, subs[[s[3]]])
      if (!all(is.finite(c(hi, lo))) || !is.finite(tr$contrast[[s[1]]])) next
      est <- hi[1] - lo[1]; se <- sqrt(hi[2] ^ 2 + lo[2] ^ 2)
      rows[[length(rows) + 1]] <- new_scalar("PRE_SUBC",
        sprintf("%s_k=%d", s[1], k), NA_real_, NA_real_, as.integer(k),
        est, se, tr$contrast[[s[1]]] * (k - REF_K))
      ctails[[s[1]]] <- c(ctails[[s[1]]], wald_tail(est, se))
    }
  }
  if (!length(rows)) return(NULL)

  # CONSTRUCTED (1): the slope, as one forest on the per-unit slope statistic.
  w <- ks - REF_K
  S <- Reduce(`+`, Map(function(k, wi) wi * dY[[as.character(k)]], ks, w)) / sum(w * w)
  cfs <- forest(S)
  if (!is.null(cfs)) {
    a <- att_on(cfs)
    rows[[length(rows) + 1]] <- new_scalar("PRE", "slope", NA_real_, NA_real_,
                                           NA_real_, a[1], a[2], tr$diff)
    for (s in list(c("X1", "X1=1", "X1=0"), c("X2", "X2=high", "X2=low"))) {
      hi <- att_on(cfs, subs[[s[2]]]); lo <- att_on(cfs, subs[[s[3]]])
      if (!all(is.finite(c(hi, lo))) || !is.finite(tr$contrast[[s[1]]])) next
      rows[[length(rows) + 1]] <- new_scalar("PRE_SUBC", paste0(s[1], "_slope"),
        NA_real_, NA_real_, NA_real_, hi[1] - lo[1],
        sqrt(hi[2] ^ 2 + lo[2] ^ 2), tr$contrast[[s[1]]])
    }
  }

  # Any-k decision rules, raw and Bonferroni-scaled by the number of pre-periods.
  add_any <- function(tl, etype, prefix, truth) {
    if (!length(tl) || !any(is.finite(tl))) return(invisible(NULL))
    p <- min(tl, na.rm = TRUE)
    rows[[length(rows) + 1]] <<- new_decision(etype, paste0(prefix, "any"),
                                              p, truth)
    rows[[length(rows) + 1]] <<- new_decision(etype, paste0(prefix, "any_bonf"),
                                              min(1, length(tl) * p), truth)
  }
  add_any(tails, "PRE", "", tr$diff)
  for (nm in c("X1", "X2"))
    add_any(ctails[[nm]], "PRE_SUBC", paste0(nm, "_"), tr$contrast[[nm]])
  rows
}

options(warn = -1)
# Progress and per-replication failures go to STDERR (`message`), which the sink
# above does NOT capture -- so they stay visible in a Colab cell even while the
# run log is being written to output_wang_pretrend.txt.
for (lin in c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")) {
  if (!dir.exists(lin)) { cat("no", lin, "- skip\n"); next }
  lin_degree <- as.integer(sub(".*=", "", lin))
  files <- list.files(lin, pattern = "^iteration_.*csv$", full.names = TRUE)
  files <- files[order(as.integer(gsub("[^0-9]", "", basename(files))))]
  if (length(files) > REPS) files <- files[1:REPS]
  if (!length(files)) { cat("no files in", lin, "- skip\n"); next }
  all_rows <- list(); n_fail <- 0L; first_err <- NULL
  message(sprintf("[%s] %d replications", lin, length(files)))
  for (ii in seq_along(files)) {
    rep <- ii - 1
    # Everything that touches the data is inside the guard: a single malformed
    # panel, a degenerate split or a grf failure costs one replication, never
    # the whole run.
    fr <- tryCatch({
      d <- read.csv(files[ii])
      rows <- run_rep(d)
      if (is.null(rows) || !length(rows)) NULL
      else finalize(rows, rep, length(unique(d$unit_id)), lin_degree)
    }, error = function(e) {
      n_fail <<- n_fail + 1L
      if (is.null(first_err)) first_err <<- conditionMessage(e)
      NULL
    })
    if (!is.null(fr)) all_rows[[length(all_rows) + 1]] <- fr
    if (ii %% 25L == 0L || ii == length(files))
      message(sprintf("  %s %d/%d (%d kept, %d failed)", lin, ii, length(files),
                      length(all_rows), n_fail))
  }
  if (n_fail > 0L) {
    msg <- sprintf("%s: %d/%d replications failed; first error: %s",
                   lin, n_fail, length(files), first_err)
    message("  !! ", msg); cat("  !!", msg, "\n")
  }
  res <- if (length(all_rows)) do.call(rbind, all_rows) else data.frame()
  fn <- sprintf("summaries_%s_%s_lin_%d.csv", METHOD, SETTING, lin_degree)
  write.csv(res, fn, row.names = FALSE, na = "")
  cat("\nwrote", fn, "(", nrow(res), "rows )\n")
  message(sprintf("  wrote %s (%d rows)", fn, nrow(res)))
}
sink()
