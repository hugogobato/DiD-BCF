# Synthetic DiD (overall ATT; covariate-residualised) benchmark for scenario "B1_null".
# Run from inside R_code/B1_null_datasets/  (the folder holding the
# linearity_degree=1/2/3 sub-folders written by DGPs/data_creation_B1_null.py).
# Estimands are detected from the data, so this works for both the single-cohort
# (canonical) and 3-cohort (staggered) scenarios.

library(synthdid)
library(progress)
sink("output_synthdid.txt")

# synthdid requires a single (block) adoption time, so -- as in the original
# study -- eventually-treated units are treated as switching on at the earliest
# post period; the estimand is the overall ATT.
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
    ols <- lm(Y ~ X_1+X_2+X_3+X_4+X_5, data = d)
    d$res_Y <- d$Y - predict(ols, newdata = d)
    d$treated <- d$eventually_treated * d$post_treatment
    panel <- data.frame(unit_id = d$unit_id, time = d$time,
                        res_Y = d$res_Y, treated = d$treated)
    est <- NA_real_; se <- NA_real_
    try({
      setup <- panel.matrices(panel)
      tau <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
      est <- as.numeric(tau)
      se <- tryCatch(sqrt(vcov(tau, method = "placebo")[1, 1]), error = function(e) NA_real_)
    }, silent = TRUE)
    truth <- mean(d$CATE[d$D == 1], na.rm = TRUE)
    sig <- as.integer(is.finite(se) & abs(est / se) > 1.96)
    if (is.finite(est)) {
      RMSE[ii] <- abs(est - truth); MAE[ii] <- abs(est - truth)
      if (truth != 0) MAPE[ii] <- abs((est - truth) / truth)
    }
    est_rows[[length(est_rows) + 1]] <- data.frame(
      iteration = ii - 1, estimand = "ATT", estimate = est, se = se,
      true = truth, sig = sig)
  }
  cat(sprintf("\n%s : mean |err|=%.4f (sd %.4f)  MAPE=%.4f\n",
              lin, mean(RMSE), sd(RMSE), mean(MAPE, na.rm = TRUE)))
  metrics_df <- data.frame(iteration = 0:(niter - 1), RMSE = RMSE, MAE = MAE, MAPE = MAPE)
  est_df <- if (length(est_rows)) do.call(rbind, est_rows) else data.frame()
  fn <- paste0("synthdid_GATE_and_PValues_", lin, ".xlsx")
  if (requireNamespace("openxlsx", quietly = TRUE)) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Metrics");  openxlsx::writeData(wb, "Metrics", metrics_df)
    openxlsx::addWorksheet(wb, "Estimates"); openxlsx::writeData(wb, "Estimates", est_df)
    openxlsx::saveWorkbook(wb, fn, overwrite = TRUE); cat("wrote", fn, "\n")
  } else {
    write.csv(metrics_df, paste0("synthdid_Metrics_", lin, ".csv"), row.names = FALSE)
    write.csv(est_df,     paste0("synthdid_Estimates_", lin, ".csv"), row.names = FALSE)
  }
}
sink()
