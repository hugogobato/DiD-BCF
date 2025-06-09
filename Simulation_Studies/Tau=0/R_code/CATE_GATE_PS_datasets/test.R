library(readxl)
library(dplyr)

# Loop over different linearity degrees
for (degree in 1:3) {
  # Construct the filename based on the current degree
  filename <- paste0("DoubleML_did_CATE_GATE_PS_and_PValues_linearity_degree=", degree, ".xlsx")

  # Check if the file exists
  if (file.exists(filename)) {
    # Read the Excel file
    data <- read_excel(filename)

    # Calculate mean and standard deviation for RMSE and MAE
    summary_stats <- data %>%
      summarise(
        mean_rmse = mean(.[[1]], na.rm = TRUE),
        sd_rmse = sd(.[[1]], na.rm = TRUE),
        mean_mae = mean(.[[2]], na.rm = TRUE),
        sd_mae = sd(.[[2]], na.rm = TRUE)
      )

    # Print the summary statistics in the desired format
    cat("Setting ", degree, ":", "\n")
    cat("Mean RMSE for 100 simulations:", summary_stats$mean_rmse, "\n")
    cat("Standard Deviation RMSE for 100 simulations:", summary_stats$sd_rmse, "\n")
    cat("Mean MAE for 100 simulations:", summary_stats$mean_mae, "\n")
    cat("Standard Deviation MAE for 100 simulations:", summary_stats$sd_mae, "\n")
    cat("\n") # Add a blank line for separation
  } else {
    cat("File not found:", filename, "\n")
  }
}