import os
import pandas as pd
import numpy as np
from tqdm import tqdm # Optional: for progress bars


def generate_did_data(
    n_units=200,
    num_x_covariates=5,
    num_pre_periods=5,
    num_post_periods=5,
    linearity_degree=1, # 1: fully linear, 2: half X non-linear, 3: treatment + all X non-linear
    pre_trend_bias_delta=0.2,
    epsilon_scale=1,
    seed=42
):
    """
    Generates panel data for Difference-in-Differences analysis with controllable pre-trends and non-linearity.

    Args:
        n_units (int): Number of units (e.g., individuals, firms).
        num_x_covariates (int): Number of control covariates (X) (not counting the two covariates W1 and W2 where W1 ~ Bernoulli(0.66)
        and W2 takes the values 1,2,3,4 with the following probabilities 0.3, 0.1, 0.2, 0.4 respectively.
        num_pre_periods (int): Number of periods before treatment.
        num_post_periods (int): Number of periods after treatment.
        treatment_effect_beta (float): True treatment effect size.
        linearity_degree (int): Degree of linearity in the DGP:
            1: Fully linear.
            2: Half of X covariates have non-linear relationship with Y.
            3: Treatment and all X covariates have non-linear relationship with Y.
        pre_trend_bias_delta (float): Bias parameter to induce pre-trends in the treated group.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated panel data in long format.
    """
    np.random.seed(seed)

    # --- Set treatment_effect_beta based on linearity_degree (like R code) ---
    if linearity_degree == 1 or linearity_degree == 2:
        treatment_effect_beta = 0
    elif linearity_degree == 3:
        treatment_effect_beta = 0
    else:
        # Handle cases where linearity_degree is not 1, 2, or 3
        print(f"Warning: linearity_degree ({linearity_degree}) has an unexpected value. Setting treatment_effect_beta to NaN.")
        treatment_effect_beta = np.nan

    periods = num_pre_periods + num_post_periods
    unit_ids = range(n_units)
    time_periods = range(periods)

    # Create base data frame
    data = pd.DataFrame({
        'unit_id': np.repeat(unit_ids, periods),
        'time': np.tile(time_periods, n_units)
    })

    # Treatment assignment (randomly assign half to treatment)
    treated_units = np.random.choice(unit_ids, size=n_units // 2, replace=False) #TO DO: add more complex propensity score
    data['treated_group'] = np.where(data['unit_id'].isin(treated_units), 1, 0)

    # Time indicators
    treatment_period = num_pre_periods # Period when treatment starts
    data['post_treatment'] = np.where(data['time'] >= treatment_period, 1, 0)
    data['time_trend'] = data['time'] # Simple linear time trend

    X = np.random.normal(0, 1, size=(len(data), num_x_covariates))


    # Add Bernoulli random variable with p=0.66
    bernoulli_values = np.random.binomial(n=1, p=0.66, size=len(data))
    # Add to both X matrix and dataframe
    X = np.column_stack((bernoulli_values,X))

   # Add categorical variable with values 1,2,3,4 with probabilities 0.3, 0.1, 0.2, 0.4
    categories = [1, 2, 3, 4]
    probabilities = [0.3, 0.1, 0.2, 0.4]
    categorical_values = np.random.choice(categories, size=len(data), p=probabilities)
    # Add to both X matrix and dataframe
    X = np.column_stack((X, categorical_values))

    for i in range(num_x_covariates+2):
        data[f'X_{i+1}'] = X[:,i]

    # Generate error term
    data['epsilon'] = np.random.normal(scale=epsilon_scale,size=len(data))

    # DGP parameters (can be adjusted for more complex DGPs)
    beta_0 = -0.5 # Intercept
    beta_treated = 0.75 # Main effect of treated group (alpha_i)
    beta_time = 0.2 # Main effect of time trend (gamma_t)
    beta_interaction = treatment_effect_beta # Treatment effect
    beta_x = np.array([-0.75, 0.5, -0.5, -1.30, 1.8, 2.5, -1.0])


    # Non-linear components based on linearity_degree
    if linearity_degree == 1: # Half covariates non-linear
        linear_x_contribution = np.sum([beta_x[i] * data[f'X_{i+1}'] for i in range(num_x_covariates+2)], axis=0)
        data['Y'] = beta_0 + beta_treated * data['treated_group'] + beta_time * data['time_trend']+linear_x_contribution+ beta_interaction * data['treated_group'] * data['post_treatment']
        data['CATE'] = beta_interaction * data['treated_group'] * data['post_treatment']

    elif linearity_degree == 2: # Half covariates non-linear
        half = num_x_covariates+2 // 2
        cov_effect = (np.sum(beta_x[:int(half/2)] * (X[:, :int(half/2)] ** 2),axis=1) + np.sum(beta_x[int(half/2):half] * np.exp(X[:, int(half/2):half]),axis=1)+
                              np.sum(beta_x[half:] * X[:, half:],axis=1))
        data['Y'] = beta_0 + beta_treated * data['treated_group'] + beta_time * data['time_trend']+cov_effect+beta_interaction * data['treated_group'] * data['post_treatment']
        data['CATE'] = beta_interaction * data['treated_group'] * data['post_treatment']

    elif linearity_degree == 3: 
        half = num_x_covariates+2 // 2
        cov_effect = (np.sum(beta_x[:int(half/2)] * (X[:, :int(half/2)] ** 2),axis=1) + np.sum(beta_x[int(half/2):half] * np.exp(X[:, int(half/2):half]),axis=1)+
                              np.sum(beta_x[half:half+int(half/2)] * np.abs(X[:, half:half+int(half/2)]),axis=1) + np.sum(beta_x[half+int(half/2):] * np.sqrt(np.abs(X[:, half+int(half/2):])),axis=1))
        data['Y'] = beta_0 + beta_treated * data['treated_group'] + beta_time * data['time_trend']**2+cov_effect+beta_interaction * data['treated_group'] * data['post_treatment']
        data['CATE'] = beta_interaction * data['treated_group'] * data['post_treatment']


    # Add pre-trend bias (differential trend for treated group in pre-treatment)
    if pre_trend_bias_delta != 0:
        if linearity_degree == 3:
            # Example parameters for seasonality
            seasonal_amplitude = 1.0  # Amplitude of the seasonal effect
            seasonal_period = 4      # Period of the seasonal effect (e.g., 12 for monthly data)

            # Calculate the seasonal effect
            seasonal_effect = seasonal_amplitude * np.sin(2 * np.pi * data['time'] / seasonal_period)
            data['Y'] += pre_trend_bias_delta * data['treated_group'] * seasonal_effect
        else:
            data['Y'] += pre_trend_bias_delta * data['treated_group'] * (data['time'] - treatment_period)
        # (data['time'] - treatment_period) will be negative in pre-treatment, 0 at treatment period, and positive in post-treatment.
        # (1 - data['post_treatment']) ensures this bias only applies in pre-treatment periods.


    # Add error term
    data['Y'] += data['epsilon']

    return data



# --- Simulation Parameters ---
num_iterations = 100
# Define the linearity degrees you want to simulate
linearity_degrees = [1, 2, 3]
# Define the main directory to store all results
main_output_dir = './data' # Changed from './data/ATE' to just './data'
# Define the specific subdirectory for this simulation set
simulation_set_dir = os.path.join(main_output_dir, 'ATE_datasets')

gen_params = {
    'n_units': 200,
    'num_x_covariates': 5, 
    'num_pre_periods': 4,
    'num_post_periods': 4,
    'pre_trend_bias_delta': 0,
    'epsilon_scale': 1
}

# --- Main Loop ---
print(f"Starting data generation...")
print(f"Output directory: {simulation_set_dir}")
print(f"Linearity Degrees: {linearity_degrees}")
print(f"Iterations per degree: {num_iterations}")

# Create the main simulation set directory if it doesn't exist
os.makedirs(simulation_set_dir, exist_ok=True)

# Outer loop for different linearity degrees
for degree in tqdm(linearity_degrees, desc="Linearity Degrees"):
    print(f"\nProcessing linearity_degree={degree}...")

    # 1. Create the specific folder for this linearity degree
    #    Folder name format: linearity_degree=X
    degree_folder_name = f'linearity_degree={degree}'
    degree_folder_path = os.path.join(simulation_set_dir, degree_folder_name)
    os.makedirs(degree_folder_path, exist_ok=True)
    # print(f"  Output folder: {degree_folder_path}") # Optional: print folder path

    # 2. Inner loop for iterations
    for i in tqdm(range(num_iterations), desc=f"  Iterations (Degree {degree})", leave=False):
        # Generate data with current degree and iteration number as seed
        # Pass other parameters using dictionary unpacking
        current_data_df = generate_did_data(
            **gen_params,
            linearity_degree=degree,
            seed=i # Use iteration number 'i' as the seed
        )

        # Define filename and full path for the CSV
        filename = f'iteration_{i}.csv'
        filepath = os.path.join(degree_folder_path, filename)

        # Save the generated data DataFrame as a CSV file
        try:
            current_data_df.to_csv(filepath, index=False)
        except Exception as e:
            print(f"\nError saving file {filepath}: {e}")

print("\nData generation complete.")
print(f"Generated data saved in subfolders under: {simulation_set_dir}")