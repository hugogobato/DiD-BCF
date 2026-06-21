# Gardner two-stage (R `did2s`) event study benchmark for scenario "B2_sweep_serial".
# Run from inside R_code/B2_sweep_serial_datasets/  (the folder holding the
# linearity_degree=1/2/3 sub-folders written by DGPs/data_creation_B2_sweep_serial.py).
# Estimands are detected from the data, so this works for both the single-cohort
# (canonical) and 3-cohort (staggered) scenarios.

library(did2s)
library(progress)
sink("output_did2s.txt")

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
      event_study(yname = "Y", tname = "time", idname = "unit_id",
                  gname = "first_treat_period", xformla = ~ X_1+X_2+X_3+X_4+X_5,
                  data = d, estimator = "did2s"),
      error = function(e) NULL)
    if (is.null(out)) next
    sub <- out[out$estimator == "did2s", ]
    sub$k <- suppressWarnings(as.integer(as.character(sub$term)))
    errs <- c(); aerr <- c(); ape <- c()
    for (kk in sort(unique(sub$k[is.finite(sub$k) & sub$k >= 0]))) {
      row <- sub[sub$k == kk, ]
      est <- row$estimate[1]; se <- row$std.error[1]
      truth <- mean(d$CATE[d$D == 1 & d$event_time == kk], na.rm = TRUE)
      sig <- as.integer(abs(est / se) > 1.96)
      errs <- c(errs, est - truth); aerr <- c(aerr, abs(est - truth))
      if (truth != 0) ape <- c(ape, abs((est - truth) / truth))
      est_rows[[length(est_rows) + 1]] <- data.frame(
        iteration = ii - 1, k = kk, estimate = est, se = se, true = truth, sig = sig)
    }
    if (length(errs)) { RMSE[ii] <- sqrt(mean(errs^2)); MAE[ii] <- mean(aerr) }
    if (length(ape)) MAPE[ii] <- mean(ape)
  }
  cat(sprintf("\n%s : mean RMSE=%.4f (sd %.4f)  MAE=%.4f  MAPE=%.4f\n",
              lin, mean(RMSE), sd(RMSE), mean(MAE), mean(MAPE, na.rm = TRUE)))
  metrics_df <- data.frame(iteration = 0:(niter - 1), RMSE = RMSE, MAE = MAE, MAPE = MAPE)
  est_df <- if (length(est_rows)) do.call(rbind, est_rows) else data.frame()
  fn <- paste0("did2s_GATE_and_PValues_", lin, ".xlsx")
  if (requireNamespace("openxlsx", quietly = TRUE)) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Metrics");  openxlsx::writeData(wb, "Metrics", metrics_df)
    openxlsx::addWorksheet(wb, "Estimates"); openxlsx::writeData(wb, "Estimates", est_df)
    openxlsx::saveWorkbook(wb, fn, overwrite = TRUE); cat("wrote", fn, "\n")
  } else {
    write.csv(metrics_df, paste0("did2s_Metrics_", lin, ".csv"), row.names = FALSE)
    write.csv(est_df,     paste0("did2s_Estimates_", lin, ".csv"), row.names = FALSE)
  }
}
sink()
