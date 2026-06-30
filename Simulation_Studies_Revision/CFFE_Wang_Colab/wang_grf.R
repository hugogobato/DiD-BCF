# grf-based difference-in-differences benchmark in the spirit of Wang (2022):
# first-difference the outcome against the last clean pre-period, then fit a
# standard grf::causal_forest per (cohort g, post period t) with never-treated
# controls.  This is the "readily available grf causal-forest DiD" R3.1.1 names.
#
# CATT-capable: grf returns a per-observation tau_hat(X_i) with a variance
# estimate, so we emit a *genuine* CATT surface (per-treated-obs tau_hat vs the
# true CATE) plus averaged GATT(g,t) rows (grf::average_treatment_effect) -- the
# same DiD-BCF schema the other benchmarks use, fed to
# did_bcf_revision.metrics.{compute_metrics, surface_metrics}.
#
# Reads linearity_degree=1/2/3 under the CWD and writes
#   summaries_wang_<SETTING>_lin_<d>.csv
suppressMessages({library(grf); library(progress)})
sink("output_wang.txt")
ARGS <- commandArgs(TRUE)
DGP <- if (length(ARGS) >= 1) ARGS[1] else "canonical"
SETTING <- if (length(ARGS) >= 2) ARGS[2] else "B1_baseline"
METHOD <- "wang"
N_TREES <- if (length(ARGS) >= 3) as.integer(ARGS[3]) else 2000L
REPS <- if (length(ARGS) >= 4) as.integer(ARGS[4]) else 100L

SCHEMA <- c("dgp","setting","linearity_degree","N","rep","estimand_type",
            "estimand_id","g","t","k","method","post_mean","sd","q025","q05",
            "q95","q975","p_bayes","surf_rmse","surf_mae","surf_n","surf_mape",
            "surf_cover95","surf_len95","surf_cover90","surf_len90","true")
Z95 <- 1.959964; Z90 <- 1.644854
XN <- paste0("X_", 1:5)
wald_tail <- function(est, se) if (is.finite(se) && se > 0) pnorm(-abs(est / se)) else NA_real_

new_scalar <- function(etype, eid, g, t, k, est, se, truth) {
  data.frame(estimand_type = etype, estimand_id = eid, g = g, t = t, k = k,
             post_mean = est, sd = se, q025 = est - Z95 * se, q05 = est - Z90 * se,
             q95 = est + Z90 * se, q975 = est + Z95 * se, p_bayes = wald_tail(est, se),
             surf_rmse = NA_real_, surf_mae = NA_real_, surf_n = NA_integer_,
             surf_mape = NA_real_, surf_cover95 = NA_real_, surf_len95 = NA_real_,
             surf_cover90 = NA_real_, surf_len90 = NA_real_, true = truth,
             stringsAsFactors = FALSE)
}

new_surface <- function(true_vec, est_vec, se_vec) {
  ok <- is.finite(true_vec) & is.finite(est_vec) & is.finite(se_vec)
  true_vec <- true_vec[ok]; est_vec <- est_vec[ok]; se_vec <- se_vec[ok]
  if (!length(true_vec)) return(NULL)
  err <- est_vec - true_vec
  lo95 <- est_vec - Z95 * se_vec; hi95 <- est_vec + Z95 * se_vec
  lo90 <- est_vec - Z90 * se_vec; hi90 <- est_vec + Z90 * se_vec
  nz <- abs(true_vec) > 1e-8
  data.frame(estimand_type = "CATT", estimand_id = "surface",
             g = NA_real_, t = NA_real_, k = NA_real_, post_mean = NA_real_,
             sd = NA_real_, q025 = NA_real_, q05 = NA_real_, q95 = NA_real_,
             q975 = NA_real_, p_bayes = NA_real_,
             surf_rmse = sqrt(mean(err ^ 2)), surf_mae = mean(abs(err)),
             surf_n = length(err),
             surf_mape = if (any(nz)) mean(abs(err[nz] / true_vec[nz])) else NA_real_,
             surf_cover95 = mean(lo95 <= true_vec & true_vec <= hi95),
             surf_len95 = mean(hi95 - lo95),
             surf_cover90 = mean(lo90 <= true_vec & true_vec <= hi90),
             surf_len90 = mean(hi90 - lo90), true = NA_real_,
             stringsAsFactors = FALSE)
}

finalize <- function(rows, rep, N, lin_degree) {
  if (!length(rows)) return(NULL)
  df <- do.call(rbind, rows)
  df$dgp <- DGP; df$setting <- SETTING; df$linearity_degree <- lin_degree
  df$N <- N; df$rep <- rep; df$method <- METHOD
  df[, SCHEMA]
}

# one replication -> list of schema rows
run_rep <- function(d) {
  d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
  N <- length(unique(d$unit_id))
  times <- sort(unique(d$time))
  cohorts <- sort(unique(d$first_treat_period[d$first_treat_period > 0]))
  never <- d$first_treat_period == 0
  if (!any(never) || !length(cohorts)) return(NULL)
  rows <- list(); surf_true <- c(); surf_est <- c(); surf_se <- c()
  for (g in cohorts) {
    base <- if ((g - 1) %in% times) g - 1 else min(times)        # clean pre-period
    yb <- d[d$time == base, c("unit_id", "Y")]; names(yb) <- c("unit_id", "Y_base")
    post_t <- times[times >= g]
    for (tt in post_t) {
      cur <- d[d$time == tt, ]
      cur <- merge(cur, yb, by = "unit_id")
      keep <- (cur$first_treat_period == g) | (cur$first_treat_period == 0)
      cur <- cur[keep, ]
      if (sum(cur$first_treat_period == g) < 5 || sum(cur$first_treat_period == 0) < 5) next
      dY <- cur$Y - cur$Y_base
      W <- as.numeric(cur$first_treat_period == g)
      cf <- tryCatch(causal_forest(X = as.matrix(cur[, XN]), Y = dY, W = W,
                                   num.trees = N_TREES, clusters = cur$unit_id,
                                   seed = 1L),
                     error = function(e) NULL)
      if (is.null(cf)) next
      pr <- predict(cf, estimate.variance = TRUE)
      treated_idx <- which(W == 1 & cur$D == 1)
      if (!length(treated_idx)) next
      tau_i <- pr$predictions[treated_idx]
      se_i <- sqrt(pmax(pr$variance.estimates[treated_idx], 0))
      cate_i <- cur$CATE[treated_idx]
      # GATT(g,t): grf doubly-robust average over the treated cohort at t
      ate <- tryCatch(average_treatment_effect(cf, subset = (W == 1),
                                               target.sample = "treated"),
                      error = function(e) c(estimate = mean(tau_i),
                                            std.err = sqrt(mean(se_i ^ 2) / length(se_i))))
      truth <- mean(cate_i, na.rm = TRUE)
      rows[[length(rows) + 1]] <- new_scalar("GATT", sprintf("g=%g_t=%d", g, tt),
        as.numeric(g), as.integer(tt), as.integer(tt - g),
        as.numeric(ate["estimate"]), as.numeric(ate["std.err"]), truth)
      surf_true <- c(surf_true, cate_i); surf_est <- c(surf_est, tau_i)
      surf_se <- c(surf_se, se_i)
    }
  }
  s <- new_surface(surf_true, surf_est, surf_se)
  if (!is.null(s)) rows[[length(rows) + 1]] <- s
  list(rows = rows, N = N)
}

options(warn = -1)
for (lin in c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")) {
  if (!dir.exists(lin)) { cat("no", lin, "- skip\n"); next }
  lin_degree <- as.integer(sub(".*=", "", lin))
  files <- list.files(lin, pattern = "^iteration_.*csv$", full.names = TRUE)
  files <- files[order(as.integer(gsub("[^0-9]", "", basename(files))))]
  if (length(files) > REPS) files <- files[1:REPS]
  if (!length(files)) { cat("no files in", lin, "- skip\n"); next }
  all_rows <- list()
  pb <- progress_bar$new(total = length(files), format = paste0(lin, " [:bar] :current/:total"))
  for (ii in seq_along(files)) {
    pb$tick(); rep <- ii - 1
    d <- read.csv(files[ii])
    rr <- tryCatch(run_rep(d), error = function(e) NULL)
    if (is.null(rr)) next
    fr <- finalize(rr$rows, rep, rr$N, lin_degree)
    if (!is.null(fr)) all_rows[[length(all_rows) + 1]] <- fr
  }
  res <- if (length(all_rows)) do.call(rbind, all_rows) else data.frame()
  fn <- sprintf("summaries_%s_%s_lin_%d.csv", METHOD, SETTING, lin_degree)
  write.csv(res, fn, row.names = FALSE, na = "")
  cat("\nwrote", fn, "(", nrow(res), "rows )\n")
}
sink()
