import os
import pandas as pd
import numpy as np
from tqdm import tqdm # Optional: for progress bars


def generate_did_data(
    n_units=200,
    num_pre_periods=5,
    num_post_periods=5,
    linearity_degree=1, # 1: fully linear, 2: half X non-linear, 3: treatment + all X non-linear
    propensity_coeffs={'intercept': 0.0, 'X1': 0.5, 'X7': -0.5}, # Coefficients for propensity score (using static X1 and X7)
    pre_trend_bias_delta=0.2,
    epsilon_scale=1,
    seed=42
):
    """
    Generates panel data for Difference-in-Differences analysis with controllable pre-trends,
    non-linearity, conditional treatment effects, and propensity score based treatment assignment.

    Generates exactly 8 covariates:
    - X1: Bernoulli(p=0.66) - Static (unit-level)
    - X2: Bernoulli(p=0.45) - Time-varying
    - X3: Normal(0,1) - Time-varying (Used in CATE)
    - X4: Normal(0,1) - Time-varying
    - X5: Normal(0,1) - Time-varying
    - X6: Normal(0,1) - Time-varying
    - X7: Normal(0,1) - Static (unit-level, Used in Propensity Score)
    - X8: Categorical{1,2,3,4} probs {0.3, 0.1, 0.2, 0.4} - Time-varying (Used in CATE)

    Args:
        n_units (int): Number of units (e.g., individuals, firms).
        num_pre_periods (int): Number of periods before treatment.
        num_post_periods (int): Number of periods after treatment.
        linearity_degree (int): Degree of linearity in the DGP:
            1: Fully linear Y, conditional treatment effect based on X3/X8.
            2: X1,X2,X3,X4 non-linear, conditional treatment effect based on X3/X8.
            3: All X covariates non-linear, conditional treatment effect based on X3/X8.
        propensity_coeffs (dict): Dictionary with coefficients for the propensity score calculation.
                                  Expected keys: 'intercept', 'X1', 'X7'.
                                  p = sigmoid(intercept + X1*coeff_X1 + X7*coeff_X7)
        pre_trend_bias_delta (float): Bias parameter to induce pre-trends in the treated group.
        epsilon_scale (float): Scale (standard deviation) of the error term.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated panel data in long format with 'Y', covariates (X1-X8), 'treated_group',
                      'post_treatment', 'CATE', and 'propensity_score'.
    """
    np.random.seed(seed)
    total_periods = num_pre_periods + num_post_periods
    total_observations = n_units * total_periods
    unit_ids = np.arange(n_units)
    time_periods = np.arange(total_periods)

    # --- Fixed Covariate Betas ---
    beta_x = np.array([-0.75, 0.5, -0.5, -1.30, 1.8, 2.5, -1.0, 0.3])
    if len(beta_x) != 8:
        raise ValueError("beta_x must have exactly 8 elements.")

    # --- Set base treatment_effect_beta based on linearity_degree ---
    if linearity_degree == 1 or linearity_degree == 2:
        treatment_effect_beta = 3.0
    elif linearity_degree == 3:
        treatment_effect_beta = 5.0
    else:
        raise ValueError(f"linearity_degree ({linearity_degree}) must be 1, 2, or 3.")

    # --- Generate Static Covariates (Unit-Level) ---
    X1_unit = np.random.binomial(n=1, p=0.66, size=n_units) # Static Bernoulli for Propensity
    X7_unit = np.random.normal(0, 1, size=n_units)          # Static Numerical for Propensity

    # --- Propensity Score Calculation and Treatment Assignment ---
    z_unit = (propensity_coeffs['intercept'] +
              propensity_coeffs['X1'] * X1_unit +
              propensity_coeffs['X7'] * X7_unit)
    propensity_scores_unit = 1 / (1 + np.exp(-z_unit))
    treatment_assignment_random = np.random.uniform(0, 1, size=n_units)
    treated_units_mask = treatment_assignment_random < propensity_scores_unit
    treated_units = unit_ids[treated_units_mask]

    # --- Create Base DataFrame and Map Static Data ---
    data = pd.DataFrame({
        'unit_id': np.repeat(unit_ids, total_periods),
        'time': np.tile(time_periods, n_units)
    })
    unit_map = data['unit_id'].values # Index mapper from panel row to unit

    data['X1'] = X1_unit[unit_map]
    data['X7'] = X7_unit[unit_map]
    data['propensity_score'] = propensity_scores_unit[unit_map]
    data['treated_group'] = np.isin(data['unit_id'], treated_units).astype(int)

    # --- Generate Time-Varying Covariates (Panel-Level) ---
    data['X2'] = np.random.binomial(n=1, p=0.45, size=total_observations) # Time-varying Bernoulli
    data['X3'] = np.random.normal(0, 1, size=total_observations)         # Time-varying Numerical (for CATE)
    data['X4'] = np.random.normal(0, 1, size=total_observations)         # Time-varying Numerical
    data['X5'] = np.random.normal(0, 1, size=total_observations)         # Time-varying Numerical
    data['X6'] = np.random.normal(0, 1, size=total_observations)         # Time-varying Numerical

    # Time-varying Categorical (for CATE)
    cat_categories = [1, 2, 3, 4]
    cat_probabilities = [0.3, 0.1, 0.2, 0.4]
    data['X8'] = np.random.choice(cat_categories, size=total_observations, p=cat_probabilities)

    # --- Time indicators ---
    treatment_period = num_pre_periods
    data['post_treatment'] = np.where(data['time'] >= treatment_period, 1, 0)
    data['time_trend'] = data['time']

    # --- Generate error term ---
    data['epsilon'] = np.random.normal(scale=epsilon_scale, size=total_observations)

    # --- DGP parameters ---
    beta_0 = -0.5 # Intercept
    beta_treated = 0.75 # Main effect of treated group (alpha_i)
    beta_time = 0.2 # Main effect of time trend (gamma_t)

    # --- Calculate Conditional Treatment Effect (CATE) ---
    # Depends on X3 (first numerical, time-varying) and X8 (categorical, time-varying)
    sqrt_abs_X3 = np.sqrt(np.abs(data['X3']))
    cate_conditions = [
        (data['X8'] == 1) | (data['X8'] == 3),
        (data['X8'] == 2),
        (data['X8'] == 4)
    ]
    cate_choices = [
        treatment_effect_beta + 1.5 * sqrt_abs_X3,
        treatment_effect_beta,
        treatment_effect_beta - 0.5 * sqrt_abs_X3
    ]
    potential_cate = np.select(cate_conditions, cate_choices, default=treatment_effect_beta)



    actual_cate_contribution = potential_cate * data['treated_group'] * data['post_treatment']
    data['CATE'] = potential_cate # Store potential effect magnitude

    # --- Calculate Outcome Y based on Linearity Degree ---
    covariate_names = [f'X{i+1}' for i in range(8)]

    # Define non-linear functions for flexibility
    def nl_func1(x): return x**2
    def nl_func2(x): return np.exp(x / 2) # Scaled exp
    def nl_func3(x): return np.abs(x)
    def nl_func4(x): return np.sqrt(np.abs(x))

    cov_effect = 0

    if linearity_degree == 1: # Fully linear
        for i in range(8):
            cov_effect += beta_x[i] * data[covariate_names[i]]
        time_term = beta_time * data['time_trend']

    elif linearity_degree == 2: # X1, X2, X3, X4 non-linear
        # Apply specific non-linear functions to first 4 covariates
        cov_effect += beta_x[0] * nl_func1(data['X1']) # X1^2 (still 0 or 1)
        cov_effect += beta_x[1] * nl_func2(data['X2']) # exp(X2/2) (more distinct for 0/1)
        cov_effect += beta_x[2] * nl_func3(data['X3']) # abs(X3)
        cov_effect += beta_x[3] * nl_func4(data['X4']) # sqrt(abs(X4))
        # Linear term for remaining covariates
        for i in range(4, 8):
            cov_effect += beta_x[i] * data[covariate_names[i]]
        time_term = beta_time * data['time_trend']

    elif linearity_degree == 3: 
        # Apply different non-linear functions across all covariates
        nl_funcs = [nl_func1, nl_func2, nl_func3, nl_func4, nl_func1, nl_func2, nl_func3, nl_func4] # Example pattern
        for i in range(8):
            cov_effect += beta_x[i] * nl_funcs[i](data[covariate_names[i]])
        # Non-linear time trend
        time_term = beta_time * (data['time_trend']**2)

    # Combine all components for Y
    data['Y'] = (beta_0 + beta_treated * data['treated_group'] +
                 time_term + cov_effect +
                 actual_cate_contribution)

    # --- Add pre-trend bias ---
    if pre_trend_bias_delta != 0:
        pre_trend_effect = pre_trend_bias_delta * data['treated_group'] * (data['time'] - treatment_period) * (1 - data['post_treatment'])
        data['Y'] += pre_trend_effect

    # --- Add final error term ---
    data['Y'] += data['epsilon']

    # --- Final Touches ---
    # Reorder columns for clarity
    cols_order = ['unit_id', 'time', 'treated_group', 'post_treatment', 'propensity_score'] + \
                 covariate_names + ['CATE', 'Y']
    data = data[cols_order]

    return data

# --- Simulation Parameters ---
num_iterations = 100
# Define the linearity degrees you want to simulate
linearity_degrees = [1, 2, 3]
# Define the main directory to store all results
main_output_dir = './data' # Changed from './data/ATE' to just './data'
# Define the specific subdirectory for this simulation set
simulation_set_dir = os.path.join(main_output_dir, 'CATE_datasets')

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