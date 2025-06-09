library(synthdid)
library(progress)

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
num_groups <- 3


options(warn = -1)

for (linearity_folder in linearity_folders) {

estimated_ATE <- array(0, dim = c(num_post_periods, num_iterations, num_groups))
accumulated_p_values <- array(0, dim = c(num_post_periods, num_iterations, num_groups))

total_PTA_respected <- 0

RMSE_per_group <- matrix(0, nrow = num_iterations, ncol = num_groups)
MAE_per_group <- matrix(0, nrow = num_iterations, ncol = num_groups)
MAPE_per_group <- matrix(0, nrow = num_iterations, ncol = num_groups)

RMSE_overall <- numeric(num_iterations)
MAE_overall <- numeric(num_iterations)
MAPE_overall <- numeric(num_iterations)


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

#data_linear$treated_group[data_linear$treated_group == 1] <- num_pre_periods
data_linear$first_treat_period[!is.finite(data_linear$first_treat_period)] <- 0

ols_model <- lm(Y ~ X_1 + X_2 + X_3 + X_4 + X_5 + X_6 + X_7, data = data_linear)
data_linear$Y_hat <- predict(ols_model, newdata = data_linear)
data_linear$residualized_Y <- data_linear$Y - data_linear$Y_hat
data_linear$treated <- data_linear$eventually_treated * data_linear$post_treatment

data_linear_new <- data.frame(
unit_id = data_linear$unit_id,
 time = data_linear$time,
  residualized_Y = data_linear$residualized_Y,
  #CATE = data_linear$CATE,
  treated=data_linear$treated
)

setup = panel.matrices(data_linear_new)
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

true_ATE_list <- list()
SE_per_group <- list()
AE_per_group <- list()
APE_per_group <- list()

for (group in 1:num_groups) {

estimated_ATE[,iteration+1,group] <- estimated_ATE_vector
accumulated_p_values[,iteration+1,group] <- ifelse(abs(estimated_ATE_vector / standard_error_vector) > 1.96, 1, 0)

    true_ATE_group <- data_linear$CATE[data_linear$first_treat_period == group+3 &
                                         data_linear$post_treatment == 1]
    true_ATE_list[[group]] <- matrix(true_ATE_group,
                                     nrow = length(true_ATE_group) / 4,
                                     ncol = 4,
                                     byrow = TRUE)

RMSE_per_group[iteration+1, group] <- sqrt(mean((estimated_ATE[,iteration+1,group] - true_ATE_group)^2)) 
MAE_per_group[iteration+1, group] <- mean(abs(estimated_ATE[,iteration+1,group] - true_ATE_group))
APE <- abs((estimated_ATE[, iteration + 1, group] - true_ATE_group) / true_ATE_group)
APE <- APE[!is.infinite(APE)]
MAPE_per_group[iteration+1, group] <- mean(APE, na.rm = TRUE)

SE_per_group[[group]] <- (estimated_ATE[,iteration+1,group] - true_ATE_group)^2
AE_per_group[[group]] <- abs(estimated_ATE[,iteration+1,group] - true_ATE_group)
APE_per_group[[group]] <- abs((estimated_ATE[, iteration + 1, group] - true_ATE_group) / true_ATE_group)
                      
}

SE <- unlist(SE_per_group)
AE <- unlist(AE_per_group)
APE_overall <- unlist(APE_per_group)
APE_overall <- APE_overall[!is.infinite(APE_overall)]

RMSE_overall[iteration+1] <- sqrt(mean(SE))
MAE_overall[iteration+1] <- mean(AE)
MAPE_overall[iteration+1] <- mean(APE_overall, na.rm = TRUE)


}

    
mean_RMSE_overall <- mean(RMSE_overall)
mean_MAE_overall <- mean(MAE_overall)
mean_MAPE_overall <- mean(MAPE_overall)
std_RMSE_overall <- sd(RMSE_overall)
std_MAE_overall <- sd(MAE_overall)
std_MAPE_overall <- sd(MAPE_overall)

cat(sprintf("Mean RMSE for %d simulations: %f\n", num_iterations, mean_RMSE_overall))
cat(sprintf("Standard Deviation RMSE for %d simulations: %f\n", num_iterations, std_RMSE_overall))
cat(sprintf("Mean MAE for %d simulations: %f\n", num_iterations, mean_MAE_overall))
cat(sprintf("Standard Deviation MAE for %d simulations: %f\n", num_iterations, std_MAE_overall))
cat(sprintf("Mean MAPE for %d simulations: %f\n", num_iterations, mean_MAPE_overall))
cat(sprintf("Standard Deviation MAPE for %d simulations: %f\n", num_iterations, std_MAPE_overall))

for (h in 1:num_groups) {
  mean_RMSE_group <- mean(RMSE_per_group[, h])
  std_RMSE_group <- sd(RMSE_per_group[, h])
  mean_MAE_group <- mean(MAE_per_group[, h])
  std_MAE_group <- sd(MAE_per_group[, h])
  mean_MAPE_group <- mean(MAPE_per_group[, h])
  std_MAPE_group <- sd(MAPE_per_group[, h])

  cat(sprintf("Mean RMSE for %d simulations for group %d: %f\n", num_iterations, h, mean_RMSE_group))
  cat(sprintf("Standard Deviation RMSE for %d simulations for group %d: %f\n", num_iterations, h, std_RMSE_group))
  cat(sprintf("Mean MAE for %d simulations for group %d: %f\n", num_iterations, h, mean_MAE_group))
  cat(sprintf("Standard Deviation MAE for %d simulations for group %d: %f\n", num_iterations, h, std_MAE_group))
  cat(sprintf("Mean MAPE for %d simulations for group %d: %f\n", num_iterations, h, mean_MAPE_group))
  cat(sprintf("Standard Deviation MAPE for %d simulations for group %d: %f\n", num_iterations, h, std_MAPE_group))
}

# 1. Prepare data frame for metrics (overall and group)
metrics_data <- data.frame(
  RMSE_overall = RMSE_overall,
  MAE_overall = MAE_overall,
  MAPE_overall = MAPE_overall
)

# 2. Flatten the group arrays into columns
for (i in 1:num_groups) {
  metrics_data[[paste0('RMSE_group_', i - 1)]] <- RMSE_per_group[, i]
  metrics_data[[paste0('MAE_group_', i - 1)]] <- MAE_per_group[, i]
  metrics_data[[paste0('MAPE_group_', i - 1)]] <- MAPE_per_group[, i]
}

# 3. The 'metrics_data' data frame is now ready

# --- Sheet 2: P-Values Data (Handling Variable Lengths) ---

df_p_values <- NULL # Initialize in case the list is empty

if (length(accumulated_p_values) == 0) {
  cat("Warning: 'accumulated_p_values' is empty. P-Value sheet will not be created.\n")
} else {
  # 1. Find the maximum length of the p-value vectors
  max_len <- 0
  for (vec in accumulated_p_values) {
    if (is.list(vec) || is.vector(vec)) {
      max_len <- max(max_len, length(vec))
    }
    # Handle potential non-iterable elements if necessary (R doesn't have strict typing like Python)
  }

  if (max_len == 0 && length(accumulated_p_values) > 0) {
    cat("Warning: accumulated_p_values contains elements but none have length > 0.\n")
    # Decide how to handle this - maybe create an empty df?
  }

  # 2. Create padded data
  padded_p_values <- list()
  for (i in 1:length(accumulated_p_values)) {
    vec <- accumulated_p_values[[i]]
    current_vec <- if (is.list(vec) || is.vector(vec)) as.numeric(vec) else numeric(0) # Ensure numeric

    padding_length <- max_len - length(current_vec)
    padded_row <- c(current_vec, rep(NA_real_, padding_length)) # Use NA for missing values
    padded_p_values[[i]] <- padded_row
  }

  # 3. Create column names for the p-values sheet
  p_value_columns <- paste0('p_value_', 0:(max_len - 1))

  # 4. Create the second data frame for p-values
  if (length(padded_p_values) > 0 && max_len > 0) {
    df_p_values <- as.data.frame(do.call(rbind, padded_p_values))
    colnames(df_p_values) <- p_value_columns
  } else {
    cat("P-Values data frame will be empty due to empty or zero-length input.\n")
    df_p_values <- data.frame() # Create an empty data frame
  }
}

# --- Save to Excel File with Multiple Sheets ---

output_filename_excel <- paste0('synthdid_GATE_and_PValues_', linearity_folder, '.xlsx')

# Check if the openxlsx package is installed
if (!requireNamespace("openxlsx", quietly = TRUE)) {
  cat("\nError: Cannot write Excel file. Please install the 'openxlsx' package.\n")
  cat("You can install it using: install.packages('openxlsx')\n")
} else {
  tryCatch({
    # Create a workbook
    wb <- openxlsx::createWorkbook()

    # Write the metrics data frame to the first sheet
    openxlsx::addWorksheet(wb, sheetName = "Metrics")
    openxlsx::writeData(wb, sheet = "Metrics", x = metrics_data, rowNames = FALSE, colNames = TRUE)
    cat(sprintf("Metrics data saved to sheet 'Metrics' in %s\n", output_filename_excel))

    # Write the p-values data frame to the second sheet (if it exists and is not empty)
    if (!is.null(df_p_values) && nrow(df_p_values) > 0) {
      openxlsx::addWorksheet(wb, sheetName = "P_Values")
      openxlsx::writeData(wb, sheet = "P_Values", x = df_p_values, rowNames = FALSE, colNames = TRUE)
      cat(sprintf("P-Values data saved to sheet 'P_Values' in %s\n", output_filename_excel))
    } else {
      cat("P-Values sheet was not created as the source list was empty or contained no vectors.\n")
    }

    # Save the workbook
    openxlsx::saveWorkbook(wb, output_filename_excel, overwrite = TRUE)

  }, error = function(e) {
    cat(sprintf("\nAn error occurred while writing the Excel file: %s\n", e$message))
  })
}


}