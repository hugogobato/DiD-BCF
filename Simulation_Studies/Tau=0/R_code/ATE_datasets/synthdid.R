library(synthdid)
library(progress)

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
  filename <- paste0("synthdid_ATE_and_PValues", suffix, ".csv")
  write.table(df, file = filename, sep = ",", col.names = FALSE, row.names = TRUE)
  
  # Return only ATE metrics
  return(list(
    overall_metrics_ate = overall_metrics_ate,
    per_time_period_metrics_ate = per_time_period_metrics_ate
  ))
}


sink("output_synthdid.txt")

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

ols_model <- lm(Y ~ X_1 + X_2 + X_3 + X_4 + X_5 + X_6 + X_7, data = data_linear)
data_linear$Y_hat <- predict(ols_model, newdata = data_linear)
data_linear$residualized_Y <- data_linear$Y - data_linear$Y_hat
data_linear$treated <- data_linear$treated_group * data_linear$post_treatment
data_linear_CATE <- data_linear$CATE
data_linear <- data.frame(
unit_id = data_linear$unit_id,
 time = data_linear$time,
  residualized_Y = data_linear$residualized_Y,
  #CATE = data_linear$CATE,
  treated=data_linear$treated
)

setup = panel.matrices(data_linear)
tau.hat = synthdid_estimate(setup$Y, setup$N0, setup$T0)


tau_hat_output <- capture.output(print(tau.hat))

# The relevant string is likely the first line of the output
output_string <- tau_hat_output[1]

ate_match <- regmatches(output_string, regexec("synthdid:\\s*([-]?\\d+\\.\\d+)\\s*\\+\\-", output_string))

# Extract the matched number and assign it to estimated_ATE
estimated_ATE_value <- as.numeric(ate_match[[1]][2])

# Use regular expressions to find the number after "+-"
se_match <- regmatches(output_string, regexec("\\+\\-\\s*(\\d+\\.\\d+)", output_string))

# Extract the matched number
standard_error <- as.numeric(se_match[[1]][2])

estimated_ATE_vector <- rep(estimated_ATE_value, times = num_post_periods)
standard_error_vector <- rep(standard_error, times = num_post_periods)

estimated_ATE[,iteration+1] <- estimated_ATE_vector

accumulated_p_values[,iteration+1] <- ifelse(abs(estimated_ATE_vector / standard_error_vector) > 1.96, 1, 0)


}

if (linearity_folder== "linearity_degree=3") {
error_metrics <- calculate_error_metrics(estimated_ATE, accumulated_p_values, true_ATE = 0, suffix = linearity_folder)
} else {
error_metrics <- calculate_error_metrics(estimated_ATE, accumulated_p_values, true_ATE = 0, suffix = linearity_folder)
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

