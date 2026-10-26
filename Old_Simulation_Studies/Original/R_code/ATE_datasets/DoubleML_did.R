library("did")
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(lgr)

lgr::get_logger("mlr3")$set_threshold("fatal")
library(progress)


doubleml_did_rf <- function(y1, y0, D, covariates,
  ml_g = lrn("regr.ranger", num.trees = 500), #as done in Chang (2020) https://doi.org/10.1093/ectj/utaa001
  ml_m = lrn("classif.ranger", num.trees = 500), #as done in Chang (2020) https://doi.org/10.1093/ectj/utaa001
  n_folds = 10, n_rep = 1, ...) {

# warning if n_rep > 1 to handle mapping from psi to inf.func
if (n_rep > 1) {
warning("n_rep > 1 is not supported.")
}
# Compute difference in outcomes
delta_y <- y1 - y0
# Prepare data backend
dml_data = DoubleML::double_ml_data_from_matrix(X = covariates, y = delta_y, d = D)
# Compute the ATT
dml_obj = DoubleML::DoubleMLIRM$new(dml_data, ml_g = ml_g, ml_m = ml_m, score = "ATTE", n_folds = n_folds)
dml_obj$fit()
att = dml_obj$coef[1]
# Return results
inf.func <- dml_obj$psi[, 1, 1]
output <- list(ATT = att, att.inf.func = inf.func)
return(output)
}

calculate_error_metrics <- function(estimated_ATE, accumulated_p_values, true_ATE = 0.5, suffix = "") {
  #' Calculates RMSE, MAE, and MAPE for the estimated ATE values,
  #' both overall and per time period (row). Adds columns for each simulation's p-values
  #' directly to the output CSV.
  #'
  #' @param estimated_ATE A matrix of shape (num_post_periods, num_simulations)
  #'                     containing the estimated ATE values.
  #' @param accumulated_p_values A matrix of shape (num_post_periods, num_simulations)
  #'                           containing the accumulated p-values.
  #' @param true_ATE The true ATE value (default: 0.5).
  #' @param suffix A suffix to add to the output CSV filename.
  #'
  #' @return A list containing:
  #'         - overall_metrics: A list containing overall RMSE, MAE, and MAPE for ATE.
  #'         - per_time_period_metrics: A list where names are time period indices
  #'           and values are lists containing RMSE, MAE, and MAPE for ATE for that time period.
  
  # Helper functions for metrics
  mean_squared_error <- function(y_true, y_pred) {
    mean((y_true - y_pred)^2)
  }
  
  mean_absolute_error <- function(y_true, y_pred) {
    mean(abs(y_true - y_pred))
  }
  
  # --- Calculate Metrics for Estimated ATE ---
  # Overall metrics for ATE
  overall_rmse_ate <- sqrt(mean_squared_error(rep(true_ATE, length(estimated_ATE)), as.vector(estimated_ATE)))
  overall_mae_ate <- mean_absolute_error(rep(true_ATE, length(estimated_ATE)), as.vector(estimated_ATE))
  
  if (true_ATE != 0) {
    overall_mape_ate <- mean(abs((as.vector(estimated_ATE) - true_ATE) / true_ATE)) * 100
  } else {
    overall_mape_ate <- NA
  }
  
  # Calculate standard deviations of metrics across simulations
  rmse_by_simulation <- sqrt(apply((estimated_ATE - true_ATE)^2, 2, mean))
  mae_by_simulation <- apply(abs(estimated_ATE - true_ATE), 2, mean)
  
  std_rmse_ate <- sd(rmse_by_simulation)
  std_mae_ate <- sd(mae_by_simulation)
  
  if (true_ATE != 0) {
    mape_by_simulation <- apply(abs((estimated_ATE - true_ATE) / true_ATE), 2, mean) * 100
    std_mape_ate <- sd(mape_by_simulation)
  } else {
    std_mape_ate <- NA
  }
  
  overall_metrics_ate <- list(
    ATE_rmse = overall_rmse_ate,
    ATE_mae = overall_mae_ate,
    ATE_mape = overall_mape_ate,
    ATE_std_rmse = std_rmse_ate,
    ATE_std_mae = std_mae_ate,
    ATE_std_mape = std_mape_ate
  )
  
  # Per-time-period metrics for ATE
  per_time_period_metrics_ate <- list()
  for (i in 1:nrow(estimated_ATE)) {  # Iterate over rows (time periods)
    rmse_ate <- sqrt(mean_squared_error(rep(true_ATE, ncol(estimated_ATE)), estimated_ATE[i, ]))
    mae_ate <- mean_absolute_error(rep(true_ATE, ncol(estimated_ATE)), estimated_ATE[i, ])
    
    if (true_ATE != 0) {
      mape_ate <- mean(abs((estimated_ATE[i, ] - true_ATE) / true_ATE)) * 100
    } else {
      mape_ate <- NA
    }
    
    std_mse_ate <- sd((estimated_ATE[i, ] - true_ATE)^2)
    std_mae_ate <- sd(abs(estimated_ATE[i, ] - true_ATE))
    
    if (true_ATE != 0) {
      std_mape_ate <- sd(abs((estimated_ATE[i, ] - true_ATE) / true_ATE) * 100)
    } else {
      std_mape_ate <- NA
    }
    
    per_time_period_metrics_ate[[as.character(i)]] <- list(
      ATE_rmse = rmse_ate,
      ATE_mae = mae_ate,
      ATE_mape = mape_ate,
      ATE_std_mse = std_mse_ate,
      ATE_std_mae = std_mae_ate,
      ATE_std_mape = std_mape_ate
    )
  }
  
  # Calculate the three specific metrics to save to CSV for ATE
  rmse_values_ate <- sqrt(apply((estimated_ATE - true_ATE)^2, 2, mean))
  mae_values_ate <- apply(abs(estimated_ATE - true_ATE), 2, mean)
  
  if (true_ATE != 0) {
    mape_values_ate <- apply(abs((estimated_ATE - true_ATE) / true_ATE), 2, mean) * 100
  } else {
    mape_values_ate <- rep(NA, ncol(estimated_ATE))
  }
  
  # Create column names that will be used consistently for both data frames
  col_names <- paste0("V", 1:ncol(estimated_ATE))
  
  # Create a matrix for combined results
  metrics_rows <- 3  # ATE_RMSE, ATE_MAE, ATE_MAPE
  p_value_rows <- nrow(accumulated_p_values)
  total_rows <- metrics_rows + p_value_rows
  
  # Create an empty matrix with the right dimensions
  result_matrix <- matrix(NA, nrow = total_rows, ncol = ncol(estimated_ATE))
  
  # Fill in the matrix
  result_matrix[1,] <- rmse_values_ate
  result_matrix[2,] <- mae_values_ate
  result_matrix[3,] <- mape_values_ate
  result_matrix[4:total_rows,] <- accumulated_p_values
  
  # Convert to data frame
  df <- as.data.frame(result_matrix)
  colnames(df) <- col_names
  
  # Set row names
  p_value_rownames <- paste0('PValue_', 0:(p_value_rows-1))
  rownames(df) <- c('ATE_RMSE', 'ATE_MAE', 'ATE_MAPE', p_value_rownames)
  
  # Save to CSV with the specified filename format
  filename <- paste0("DoubleML_did_ATE_and_PValues", suffix, ".csv")
  write.table(df, file = filename, sep = ",", col.names = FALSE, row.names = TRUE)
  
  # Return only ATE metrics
  return(list(
    overall_metrics_ate = overall_metrics_ate,
    per_time_period_metrics_ate = per_time_period_metrics_ate
  ))
}


sink("output_DoubleML_did.txt")

# Define the base directory containing the folders
base_dir <- "." # Assuming the folders are in the current working directory

# Define the linearity degrees folders
linearity_folders <- c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")

# Define the iteration numbers
iterations <- 0:99
num_iterations <- 100
num_pre_periods <- 4
num_post_periods <- 4

options(warn = -1)

for (linearity_folder in linearity_folders) {
  estimated_ATE <- matrix(0, nrow = num_post_periods, ncol = num_iterations)
accumulated_p_values <- matrix(0, nrow = num_post_periods, ncol = num_iterations)

total_PTA_respected <- 0
  # Construct the full path to the linearity degree folder
  folder_path <- file.path(base_dir, linearity_folder)
  pb <- progress_bar$new(total = num_iterations, format = "Progress [:bar] :current/:total (:percent)")
  cat(linearity_folder, "\n")
for (iteration in iterations) {
  pb$tick()  # Update the progress bar

   # Construct the filename for the current iteration
    filename <- paste0("iteration_", iteration, ".csv")

    # Construct the full path to the CSV file
    file_path <- file.path(folder_path, filename)

    data_linear <- read.csv(file_path)

data_linear$treated_group[data_linear$treated_group == 1] <- num_pre_periods

out1 <- att_gt(yname="Y",
tname="time",
idname = "unit_id",
gname="treated_group",
xformla = ~ X_1+X_2+X_3+X_4+X_5+X_6+X_7,
data=data_linear,
est_method = doubleml_did_rf)

estimated_ATE[,iteration+1] <- tail(out1$att, num_post_periods)
out1$sig <- ifelse(abs(out1$att / out1$se) > out1$c, 1, 0)
accumulated_p_values[,iteration+1] <- tail(out1$sig, num_post_periods)

total_PTA_respected <- total_PTA_respected + ifelse(out1$Wpval > 0.05, 1, 0)

}

cat("Number of iterations (out of 100) with at least two p-values above 0.05 ",
    "(i.e., we assume that the Parallel Trend Assumption holds): ", 
    total_PTA_respected, "\n", sep = "")

if (linearity_folder== "linearity_degree=3") {
error_metrics <- calculate_error_metrics(estimated_ATE, accumulated_p_values, true_ATE = 5.0, suffix = linearity_folder)
} else {
error_metrics <- calculate_error_metrics(estimated_ATE, accumulated_p_values, true_ATE = 3.0, suffix = linearity_folder)
}

overall_metrics_ate <- error_metrics$overall_metrics_ate
per_time_period_metrics_ate <- error_metrics$per_time_period_metrics_ate

# Overall Metrics
cat("Overall Metrics:\n")
for (metric in names(overall_metrics_ate)) {
  value <- overall_metrics_ate[[metric]]
  cat(sprintf("  %s: %.4f\n", metric, value))
}

# Per Time Period Metrics
cat("\nPer Time Period Metrics:\n")
for (time_period in names(per_time_period_metrics_ate)) {
  # Convert time_period to numeric and add num_pre_periods
  time_period_num <- as.numeric(time_period) + num_pre_periods
  cat(sprintf("  Time Period %d:\n", time_period_num))
  
  # Nested loop over metrics for this time period
  metrics <- per_time_period_metrics_ate[[time_period]]
  for (metric in names(metrics)) {
    value <- metrics[[metric]]
    cat(sprintf("    %s: %.4f\n", metric, value))
  }
}

print("\n")
print("\n")
}

sink()

