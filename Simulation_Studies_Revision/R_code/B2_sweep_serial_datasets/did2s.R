# Gardner two-stage (R `did2s`) event-study benchmark for scenario "B2_sweep_serial".
# Uses the DOCUMENTED did2s() call (FE-only first stage, event-study second stage),
# NOT the event_study() wrapper: the wrapper puts the covariates in the first stage,
# but X_1..X_5 are time-invariant (absorbed by the unit FE) and that path trips a
# fixef/sparse_model_matrix failure. FE-only is the vignette's spec and is identical
# here. Emits DiD-BCF-schema summaries (ES rows + broadcast CATT surface).
library(did2s)
library(progress)
sink("output_did2s.txt")
DGP <- "canonical"; SETTING <- "B2_sweep_serial"; METHOD <- "did2s"

# ---- DiD-BCF-schema summary emission (shared helpers) -----------------------
SCHEMA <- c("dgp","setting","linearity_degree","N","rep","estimand_type",
            "estimand_id","g","t","k","method","post_mean","sd","q025","q05",
            "q95","q975","p_bayes","surf_rmse","surf_mae","surf_n","surf_mape",
            "surf_cover95","surf_len95","surf_cover90","surf_len90","true")
Z95 <- 1.959964; Z90 <- 1.644854
wald_tail <- function(est, se) if (is.finite(se) && se > 0) pnorm(-abs(est / se)) else NA_real_

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
    # relative event time; never-treated -> Inf (the second-stage reference level)
    d$rel <- ifelse(d$eventually_treated == 1, d$event_time, Inf)
    es <- tryCatch(
      did2s(d, yname = "Y",
            first_stage  = ~ 0 | unit_id + time,
            second_stage = ~ i(rel, ref = c(-1, Inf)),
            treatment    = "D",
            cluster_var  = "unit_id"),
      error = function(e) NULL)
    if (is.null(es)) next
    td <- tryCatch(broom::tidy(es), error = function(e) NULL)
    if (is.null(td)) next
    td$kk <- suppressWarnings(as.integer(sub("rel::", "", td$term)))
    rows <- list(); est_lk <- list(); se_lk <- list()
    for (kk in sort(unique(td$kk[is.finite(td$kk) & td$kk >= 0]))) {
      row <- td[td$kk == kk, ]
      est <- row$estimate[1]; se <- row$std.error[1]
      truth <- mean(d$CATE[d$D == 1 & d$event_time == kk], na.rm = TRUE)
      rows[[length(rows)+1]] <- new_scalar("ES", sprintf("k=%d", kk),
        NA_real_, NA_real_, as.integer(kk), est, se, truth)
      est_lk[[as.character(kk)]] <- est; se_lk[[as.character(kk)]] <- se
    }
    tp <- d[d$D == 1, ]
    keys <- as.character(tp$event_time)
    est_i <- vapply(keys, function(k) if (!is.null(est_lk[[k]])) est_lk[[k]] else NA_real_, numeric(1))
    se_i  <- vapply(keys, function(k) if (!is.null(se_lk[[k]]))  se_lk[[k]]  else NA_real_, numeric(1))
    surf <- new_surface(tp$CATE, est_i, se_i); if (!is.null(surf)) rows[[length(rows)+1]] <- surf
    fr <- finalize(rows, rep, N, DGP, SETTING, lin_degree, METHOD)
    if (!is.null(fr)) all_rows[[length(all_rows)+1]] <- fr
  }
  res <- if (length(all_rows)) do.call(rbind, all_rows) else data.frame()
  fn <- sprintf("summaries_%s_%s_lin_%d.csv", METHOD, SETTING, lin_degree)
  write.csv(res, fn, row.names = FALSE, na = ""); cat("\nwrote", fn, "(", nrow(res), "rows )\n")
}
sink()
