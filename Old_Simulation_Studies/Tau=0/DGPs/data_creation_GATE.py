import os
import pandas as pd
import numpy as np
from tqdm import tqdm # Optional: for progress bars

def generate_staggered_did_data(
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
    Generates panel data for Difference-in-Differences analysis with staggered adoption,
    controllable pre-trends, and non-linearity.

    Creates 4 groups:
    - Group 0: Never treated (Control)
    - Group 1: Treated starting at num_pre_periods (First Treatment Time)
    - Group 2: Treated starting at num_pre_periods + 1
    - Group 3: Treated starting at num_pre_periods + 2

    Args:
        n_units (int): Total number of units (e.g., individuals, firms). Should be divisible by 4 ideally.
        num_x_covariates (int): Number of control covariates (X) (not counting W1 and W2).
        num_pre_periods (int): Number of periods before the *earliest* treatment.
        num_post_periods (int): Number of periods after the *earliest* treatment.
        linearity_degree (int): Degree of linearity in the DGP:
            1: Fully linear.
            2: Half of X covariates have non-linear relationship with Y.
            3: Treatment and half of X covariates have non-linear relationship with Y.
            4: Treatment and all X covariates have non-linear relationship with Y.
        pre_trend_bias_delta (float): Bias parameter to induce pre-trends in eventually treated groups.
        epsilon_scale (float): Standard deviation of the error term.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated panel data in long format. Includes columns:
            'unit_id': Unique identifier for each unit.
            'time': Time period index.
            'treatment_group': Integer indicating the unit's group (0: Control, 1: T0, 2: T0+1, 3: T0+2).
            'first_treat_period': The period when treatment starts for the unit (np.inf for control).
            'eventually_treated': Binary indicator (1 if unit belongs to groups 1, 2, or 3, 0 otherwise).
            'D': Binary treatment indicator (1 if unit is treated *in the current period*, 0 otherwise).
            'X_1', 'X_2', ...: Covariates.
            'Y': Outcome variable.
            'CATE': True Conditional Average Treatment Effect for the unit-period.
            'epsilon': Error term component.
            'time_trend': Linear time trend index.
    """
    np.random.seed(seed)

    # --- Set treatment_effect_beta based on linearity_degree ---
    if linearity_degree == 1 or linearity_degree == 2:
        treatment_effect_beta = 0
    elif linearity_degree == 3:
        treatment_effect_beta = 0
    else:
        print(f"Warning: linearity_degree ({linearity_degree}) has an unexpected value. Setting treatment_effect_beta to NaN.")
        treatment_effect_beta = np.nan

    periods = num_pre_periods + num_post_periods
    unit_ids = np.arange(n_units)
    time_periods = np.arange(periods)

    # Create base data frame
    data = pd.DataFrame({
        'unit_id': np.repeat(unit_ids, periods),
        'time': np.tile(time_periods, n_units)
    })

    # --- Staggered Treatment Assignment ---
    # Divide units into 4 roughly equal groups
    shuffled_unit_ids = np.random.permutation(unit_ids)
    group_size = n_units // 4
    group_assignments = {}
    group_assignments[0] = shuffled_unit_ids[0 * group_size : 1 * group_size] # Control
    group_assignments[1] = shuffled_unit_ids[1 * group_size : 2 * group_size] # Treat at T0
    group_assignments[2] = shuffled_unit_ids[2 * group_size : 3 * group_size] # Treat at T0 + 1
    # Assign remaining units (if n_units % 4 != 0) to the last group
    group_assignments[3] = shuffled_unit_ids[3 * group_size :] # Treat at T0 + 2

    # Map unit_id to treatment group
    unit_to_group = {}
    for group_id, units_in_group in group_assignments.items():
        for unit in units_in_group:
            unit_to_group[unit] = group_id
    data['treatment_group'] = data['unit_id'].map(unit_to_group)

    # Determine the first treatment period for each unit
    earliest_treatment_period = num_pre_periods # Period when the *first* group gets treated (Group 1)
    conditions = [
        data['treatment_group'] == 0,
        data['treatment_group'] == 1,
        data['treatment_group'] == 2,
        data['treatment_group'] == 3
    ]
    choices = [
        np.inf, # Never treated
        earliest_treatment_period,
        earliest_treatment_period + 1,
        earliest_treatment_period + 2
    ]
    data['first_treat_period'] = np.select(conditions, choices, default=np.nan)

    # Indicator for being *eventually* treated (used for pre-trend bias)
    data['eventually_treated'] = (data['treatment_group'] > 0).astype(int)
    data['post_treatment'] = (data['time'] >= num_pre_periods).astype(int)

    # Dynamic treatment indicator 'D': 1 if treated in the current period, 0 otherwise
    data['D'] = (data['time'] >= data['first_treat_period']).astype(int)

    # --- Covariates ---
    data['time_trend'] = data['time'] # Simple linear time trend

    # Generate X covariates
    X_numeric = np.random.normal(0, 1, size=(len(data), num_x_covariates))
    bernoulli_values = np.random.binomial(n=1, p=0.66, size=len(data))
    categories = [1, 2, 3, 4]
    probabilities = [0.3, 0.1, 0.2, 0.4]
    categorical_values = np.random.choice(categories, size=len(data), p=probabilities)

    # Combine all covariates into a single matrix X for easier processing later
    X = np.column_stack((bernoulli_values, X_numeric, categorical_values))

    # Add covariates to the DataFrame with names X_1, X_2, ...
    total_covariates = num_x_covariates + 2
    for i in range(total_covariates):
        data[f'X_{i+1}'] = X[:, i]

    # --- Generate Outcome Variable (Y) ---
    data['epsilon'] = np.random.normal(scale=epsilon_scale, size=len(data))

    # DGP parameters
    beta_0 = -0.5 # Intercept
    beta_group_effect = 0.75 # Main effect of treated group (alpha_i)
    beta_time = 0.2 # Main effect of time trend (gamma_t)
    beta_interaction = treatment_effect_beta # Treatment effect magnitude
    # Ensure beta_x has the correct length
    beta_x = np.array([-0.75, 0.5, -0.5, -1.30, 1.8, 2.5, -1.0])[:total_covariates] # Adjust length if num_x_covariates changes

    # Baseline Y components (common across linearity degrees)
    Y_base = (beta_0 +
              beta_group_effect * data['eventually_treated'] + # Group fixed effect for those eventually treated
              beta_time * data['time_trend']) # Common time trend

    # Covariate effects
    if linearity_degree == 1:
        linear_x_contribution = np.sum([beta_x[i] * data[f'X_{i+1}'] for i in range(total_covariates)], axis=0)
        Y_covariates = linear_x_contribution
        Y_treatment = beta_interaction * data['D']
        data['CATE'] = beta_interaction * data['D']

    elif linearity_degree == 2:
        half = total_covariates // 2
        cov_effect = (np.sum(beta_x[:int(half/2)] * (X[:, :int(half/2)] ** 2),axis=1) +
                      np.sum(beta_x[int(half/2):half] * np.exp(X[:, int(half/2):half]),axis=1)+
                      np.sum(beta_x[half:] * X[:, half:],axis=1))
        Y_covariates = cov_effect
        Y_treatment = beta_interaction * data['D']
        data['CATE'] = beta_interaction * data['D']

    elif linearity_degree == 3:
        half = total_covariates // 2
        # Non-linear time trend and covariates
        Y_base = (beta_0 +
                  beta_group_effect * data['eventually_treated'] +
                  beta_time * data['time_trend']**2) # Non-linear time trend
        cov_effect = (np.sum(beta_x[:int(half/2)] * (X[:, :int(half/2)] ** 2),axis=1) +
                      np.sum(beta_x[int(half/2):half] * np.exp(X[:, int(half/2):half]),axis=1)+
                      np.sum(beta_x[half:half+int(half/2)] * np.abs(X[:, half:half+int(half/2)]),axis=1) +
                      np.sum(beta_x[half+int(half/2):] * np.sqrt(np.abs(X[:, half+int(half/2):])),axis=1))
        Y_covariates = cov_effect
        # Linear treatment effect (as per original code structure for degree 4)
        Y_treatment = beta_interaction * data['D']
        data['CATE'] = beta_interaction * data['D']
    else: # Handle unexpected linearity_degree
         Y_covariates = 0
         Y_treatment = 0
         data['CATE'] = 0

    data['Y'] = Y_base + Y_covariates + Y_treatment

    # --- Add pre-trend bias ---
    # Apply bias to *eventually treated* units during the pre-period *relative to the first treatment time*
    if pre_trend_bias_delta != 0:
        # Apply bias only before the *earliest* treatment period
        pre_period_mask = data['time'] < earliest_treatment_period
        # Apply bias only to units that will eventually be treated
        bias_mask = pre_period_mask & (data['eventually_treated'] == 1)

        if linearity_degree == 3: # Non-linear pre-trend (e.g., seasonal)
            seasonal_amplitude = 1.0
            seasonal_period = 4
            seasonal_effect = seasonal_amplitude * np.sin(2 * np.pi * data['time'] / seasonal_period)
            data.loc[bias_mask, 'Y'] += pre_trend_bias_delta * seasonal_effect[bias_mask]
        else: # Linear pre-trend bias
             # Difference relative to the earliest treatment time
            time_diff = data['time'] - earliest_treatment_period
            data.loc[bias_mask, 'Y'] += pre_trend_bias_delta * time_diff[bias_mask]

    # Add final error term
    data['Y'] += data['epsilon']

    # Remove intermediate columns if desired, or keep for clarity
    # data = data.drop(columns=['epsilon'])

    return data


# --- Simulation Parameters ---
num_iterations = 100
# Define the linearity degrees you want to simulate
linearity_degrees = [1, 2, 3]
# Define the main directory to store all results
main_output_dir = './data' # Changed from './data/ATE' to just './data'
# Define the specific subdirectory for this simulation set
simulation_set_dir = os.path.join(main_output_dir, 'GATE_datasets')

gen_params = {
    'n_units': 200,
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
        current_data_df = generate_staggered_did_data(
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