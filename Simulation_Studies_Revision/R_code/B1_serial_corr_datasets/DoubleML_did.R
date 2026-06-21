# DoubleML doubly-robust DiD (random-forest nuisances) benchmark for scenario "B1_serial_corr".
# Run from inside R_code/B1_serial_corr_datasets/  (the folder holding the
# linearity_degree=1/2/3 sub-folders written by DGPs/data_creation_B1_serial_corr.py).
# Estimands are detected from the data, so this works for both the single-cohort
# (canonical) and 3-cohort (staggered) scenarios.

library(did)
library(progress)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(lgr)
lgr::get_logger("mlr3")$set_threshold("fatal")

# Plug DoubleML's ATTE estimator into att_gt as a custom est_method (as Chang 2020).
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

sink("output_DoubleML_did.txt")
lin_folders <- c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")
options(warn = -1)

for (lin in lin_folders) {
  files <- list.files(lin, pattern = "^iteration_.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) { cat("No files in", lin, "- skipping\n"); next }
  niter <- length(files)
  RMSE <- numeric(niter); MAE <- numeric(niter); MAPE <- rep(NA_real_, niter)
  est_rows <- list()
  pb <- progress_bar$new(total = niter, format = paste0(lin, " [:bar] :current/:total"))
  for (ii in seq_along(files)) {
    pb$tick()
    d <- read.csv(files[ii])
    d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
    out <- tryCatch(
      att_gt(yname = "Y", tname = "time", idname = "unit_id",
             gname = "first_treat_period", xformla = ~ X_1+X_2+X_3+X_4+X_5,
             data = d, est_method = doubleml_did_rf),
      error = function(e) NULL)
    if (is.null(out)) next
    groups <- sort(unique(out$group[out$group > 0]))
    errs <- c(); aerr <- c(); ape <- c()
    for (g in groups) {
      post_t <- sort(unique(out$t[out$group == g & out$t >= g]))
      for (t in post_t) {
        sel <- which(out$group == g & out$t == t)
        if (length(sel) == 0) next
        est <- out$att[sel]; se <- out$se[sel]
        truth <- mean(d$CATE[d$first_treat_period == g & d$time == t], na.rm = TRUE)
        sig <- as.integer(abs(est / se) > 1.96)
        errs <- c(errs, est - truth); aerr <- c(aerr, abs(est - truth))
        if (truth != 0) ape <- c(ape, abs((est - truth) / truth))
        est_rows[[length(est_rows) + 1]] <- data.frame(
          iteration = ii - 1, group = g, t = t, k = t - g,
          estimate = est, se = se, true = truth, sig = sig)
      }
    }
    if (length(errs)) { RMSE[ii] <- sqrt(mean(errs^2)); MAE[ii] <- mean(aerr) }
    if (length(ape)) MAPE[ii] <- mean(ape)
  }
  cat(sprintf("\n%s : mean RMSE=%.4f (sd %.4f)  MAE=%.4f  MAPE=%.4f\n",
              lin, mean(RMSE), sd(RMSE), mean(MAE), mean(MAPE, na.rm = TRUE)))
  metrics_df <- data.frame(iteration = 0:(niter - 1), RMSE = RMSE, MAE = MAE, MAPE = MAPE)
  est_df <- if (length(est_rows)) do.call(rbind, est_rows) else data.frame()
  fn <- paste0("DoubleML_did_GATE_and_PValues_", lin, ".xlsx")
  if (requireNamespace("openxlsx", quietly = TRUE)) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Metrics");  openxlsx::writeData(wb, "Metrics", metrics_df)
    openxlsx::addWorksheet(wb, "Estimates"); openxlsx::writeData(wb, "Estimates", est_df)
    openxlsx::saveWorkbook(wb, fn, overwrite = TRUE); cat("wrote", fn, "\n")
  } else {
    write.csv(metrics_df, paste0("DoubleML_did_Metrics_", lin, ".csv"), row.names = FALSE)
    write.csv(est_df,     paste0("DoubleML_did_Estimates_", lin, ".csv"), row.names = FALSE)
  }
}
sink()
