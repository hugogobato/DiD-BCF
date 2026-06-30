# Synthetic DiD (overall ATT; covariate-residualised) benchmark for "B1_serial_corr".
# Emits DiD-BCF-schema summaries (ATT row + broadcast CATT surface). See helpers.
# synthdid needs a single block adoption time, so eventually-treated units switch
# on at the earliest post period; the estimand is the overall ATT.
library(synthdid)
library(progress)
sink("output_synthdid.txt")
DGP <- "canonical"; SETTING <- "B1_serial_corr"; METHOD <- "synthdid"

# ---- DiD-BCF-schema summary emission (shared helpers) -----------------------
# Emit one row per (rep x estimand) in the *exact* schema of
# DiD_BCF/summaries_<scenario>_lin_<d>.csv, so the benchmark feeds
# did_bcf_revision.metrics.{compute_metrics,surface_metrics} the same way the
# DiD-BCF posterior summaries do. No estimator is modified.
SCHEMA <- c("dgp","setting","linearity_degree","N","rep","estimand_type",
            "estimand_id","g","t","k","method","post_mean","sd","q025","q05",
            "q95","q975","p_bayes","surf_rmse","surf_mae","surf_n","surf_mape",
            "surf_cover95","surf_len95","surf_cover90","surf_len90","true")
Z95 <- 1.959964; Z90 <- 1.644854
# one-sided Wald tail prob: frequentist analogue of DiD-BCF's p_bayes (smaller
# posterior tail); reject05 <=> tail < 0.025 <=> |est/se| > 1.96.
wald_tail <- function(est, se) if (is.finite(se) && se > 0) pnorm(-abs(est / se)) else NA_real_

# A scalar (averaged) estimand row. The interval is the Wald interval rebuilt
# from the reported SE, so coverage/length/rejection compute as for DiD-BCF.
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

# The CATT-surface row. The model has no individual CATT, so its scalar estimate
# (GATT(g,t) / ES(k) / ATT) is broadcast to every treated-post observation and
# compared to the individual true CATE -- exactly DiD-BCF's surface_summary().
new_surface <- function(true_vec, est_vec, se_vec) {
  ok <- is.finite(true_vec) & is.finite(est_vec)
  true_vec <- true_vec[ok]; est_vec <- est_vec[ok]; se_vec <- se_vec[ok]
  if (!length(true_vec)) return(NULL)
  err <- est_vec - true_vec
  lo95 <- est_vec - Z95 * se_vec; hi95 <- est_vec + Z95 * se_vec
  lo90 <- est_vec - Z90 * se_vec; hi90 <- est_vec + Z90 * se_vec
  nz <- abs(true_vec) > 1e-8
  data.frame(estimand_type = "CATT", estimand_id = "surface",
             g = NA_real_, t = NA_real_, k = NA_real_,
             post_mean = NA_real_, sd = NA_real_, q025 = NA_real_, q05 = NA_real_,
             q95 = NA_real_, q975 = NA_real_, p_bayes = NA_real_,
             surf_rmse = sqrt(mean(err ^ 2)), surf_mae = mean(abs(err)),
             surf_n = length(err),
             surf_mape = if (any(nz)) mean(abs(err[nz] / true_vec[nz])) else NA_real_,
             surf_cover95 = mean(lo95 <= true_vec & true_vec <= hi95),
             surf_len95 = mean(hi95 - lo95),
             surf_cover90 = mean(lo90 <= true_vec & true_vec <= hi90),
             surf_len90 = mean(hi90 - lo90), true = NA_real_,
             stringsAsFactors = FALSE)
}

finalize <- function(rows, rep, N, dgp, setting, lin_degree, method) {
  if (!length(rows)) return(NULL)
  df <- do.call(rbind, rows)
  df$dgp <- dgp; df$setting <- setting; df$linearity_degree <- lin_degree
  df$N <- N; df$rep <- rep; df$method <- method
  df[, SCHEMA]
}

lin_folders <- c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")
options(warn = -1)
for (lin in lin_folders) {
  lin_degree <- as.integer(sub(".*=", "", lin))
  files <- list.files(lin, pattern = "^iteration_", full.names = TRUE)
  files <- files[grepl("csv$", files)]
  files <- files[order(as.integer(gsub("[^0-9]", "", basename(files))))]  # numeric rep order
  if (length(files) == 0) { cat("No files in", lin, "- skipping\n"); next }
  all_rows <- list()
  pb <- progress_bar$new(total = length(files), format = paste0(lin, " [:bar] :current/:total"))
  for (ii in seq_along(files)) {
    pb$tick(); rep <- ii - 1
    d <- read.csv(files[ii])
    d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
    N <- length(unique(d$unit_id))
    ols <- lm(Y ~ X_1+X_2+X_3+X_4+X_5, data = d)
    d$res_Y <- d$Y - predict(ols, newdata = d)
    d$treated <- d$eventually_treated * d$post_treatment
    panel <- data.frame(unit_id = d$unit_id, time = d$time, res_Y = d$res_Y, treated = d$treated)
    est <- NA_real_; se <- NA_real_
    try({
      setup <- panel.matrices(panel)
      tau <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
      est <- as.numeric(tau)
      se <- tryCatch(sqrt(vcov(tau, method = "jackknife")[1, 1]), error = function(e) NA_real_)
    }, silent = TRUE)
    truth <- mean(d$CATE[d$D == 1], na.rm = TRUE)
    rows <- list()
    rows[[1]] <- new_scalar("ATT", "ATT", NA_real_, NA_real_, NA_real_, est, se, truth)
    tp <- d[d$D == 1, ]
    surf <- new_surface(tp$CATE, rep(est, nrow(tp)), rep(se, nrow(tp)))
    if (!is.null(surf)) rows[[length(rows)+1]] <- surf
    fr <- finalize(rows, rep, N, DGP, SETTING, lin_degree, METHOD)
    if (!is.null(fr)) all_rows[[length(all_rows)+1]] <- fr
  }
  res <- if (length(all_rows)) do.call(rbind, all_rows) else data.frame()
  fn <- sprintf("summaries_%s_%s_lin_%d.csv", METHOD, SETTING, lin_degree)
  write.csv(res, fn, row.names = FALSE, na = ""); cat("\nwrote", fn, "(", nrow(res), "rows )\n")
}
sink()
