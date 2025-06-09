import os
import pandas as pd
import numpy as np
from tqdm import tqdm # Optional: for progress bars

def generate_staggered_did_data_hete_effect(
    n_units=200,
    num_pre_periods=5,
    num_post_periods=5,
    linearity_degree=1,
    pre_trend_bias_delta=0.2,
    propensity_noise_scale=0.5, # Scale of noise added to utility for assignment randomness
    epsilon_scale=1,
    seed=42
):
    """
    Generates panel data for DiD with staggered adoption, propensity scores,
    fixed covariates (8 total, mixed static/dynamic), and HETEROGENEOUS
    treatment effects conditional on X1 (static Bern) and X3 (dynamic Cat).

    Covariates (8 total):
    - X1: Bernoulli(p=0.66) - STATIC, influences propensity & treatment effect
    - X2: Bernoulli(p=0.45) - Time-varying
    - X3: Categorical({1,2,3,4} p={0.3,0.1,0.2,0.4}) - Time-varying, influences treatment effect
    - X4-X7: Numerical (Normal(0,1)) - Time-varying
    - X8: Numerical (Normal(0,1)) - STATIC, influences propensity

    Treatment Effect Logic:
    - Base effect size ('base_beta') determined by linearity_degree.
    - Modifier term = sqrt(abs(X1))
    - If X3 is 1 or 3: Effect = 1.5 * modifier + base_beta
    - If X3 is 2:       Effect = base_beta
    - If X3 is 4:       Effect = base_beta - 0.5 * modifier

    Args:
        n_units (int): Total number of units.
        num_pre_periods (int): Periods before the *earliest* treatment.
        num_post_periods (int): Periods after the *earliest* treatment.
        linearity_degree (int): Degree of linearity in the DGP (1-4). Also sets base treatment effect.
        pre_trend_bias_delta (float): Bias for pre-trends in eventually treated groups.
        propensity_noise_scale (float): Std deviation of noise added to group utility.
        epsilon_scale (float): Std deviation of the outcome error term.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated panel data with heterogeneous treatment effects.
    """
    np.random.seed(seed)
    total_covariates = 8 # Fixed number of covariates

    # --- 1. Generate STATIC Unit-Level Covariates (X1, X8) ---
    unit_ids = np.arange(n_units)
    unit_X1_bern = np.random.binomial(n=1, p=0.66, size=n_units)
    unit_X8_num = np.random.normal(0, 1, size=n_units)

    # --- 2. Calculate Group Utilities and Assign Groups based on STATIC X1, X8 ---
    coeffs = {
        'g1': {'intercept': 0.1, 'x1_bern': 0.8, 'x8_num': 0.6},
        'g2': {'intercept': 0.0, 'x1_bern': -0.5, 'x8_num': -0.7},
        'g3': {'intercept': -0.1, 'x1_bern': 0.3, 'x8_num': 0.4}
    }
    V0 = np.zeros(n_units)
    V1 = coeffs['g1']['intercept'] + coeffs['g1']['x1_bern'] * unit_X1_bern + coeffs['g1']['x8_num'] * unit_X8_num
    V2 = coeffs['g2']['intercept'] + coeffs['g2']['x1_bern'] * unit_X1_bern + coeffs['g2']['x8_num'] * unit_X8_num
    V3 = coeffs['g3']['intercept'] + coeffs['g3']['x1_bern'] * unit_X1_bern + coeffs['g3']['x8_num'] * unit_X8_num
    noise = np.random.normal(0, propensity_noise_scale, size=(n_units, 4))
    U = np.column_stack((V0, V1, V2, V3)) + noise
    unit_treatment_group = np.argmax(U, axis=1)

    # --- 3. Create Panel DataFrame and Merge STATIC Unit-Level Info ---
    periods = num_pre_periods + num_post_periods
    time_periods = np.arange(periods)
    data = pd.DataFrame({
        'unit_id': np.repeat(unit_ids, periods),
        'time': np.tile(time_periods, n_units)
    })
    df_unit_static = pd.DataFrame({
        'unit_id': unit_ids,
        'treatment_group': unit_treatment_group,
        'X1': unit_X1_bern,
        'X8': unit_X8_num
    })
    data = pd.merge(data, df_unit_static, on='unit_id', how='left')

    # --- 4. Generate Time-Varying Covariates (X2-X7) ---
    n_observations = len(data)
    data['X2'] = np.random.binomial(n=1, p=0.45, size=n_observations)
    cat_choices = [1, 2, 3, 4]; cat_probs = [0.3, 0.1, 0.2, 0.4]
    data['X3'] = np.random.choice(cat_choices, size=n_observations, p=cat_probs)
    X_num_time_varying = np.random.normal(0, 1, size=(n_observations, 4))
    data['X4'] = X_num_time_varying[:, 0]
    data['X5'] = X_num_time_varying[:, 1]
    data['X6'] = X_num_time_varying[:, 2]
    data['X7'] = X_num_time_varying[:, 3]

    # --- 5. Define Treatment Timing and Indicators ---
    earliest_treatment_period = num_pre_periods
    conditions_timing = [
        data['treatment_group'] == 0, data['treatment_group'] == 1,
        data['treatment_group'] == 2, data['treatment_group'] == 3
    ]
    choices_timing = [ np.inf, earliest_treatment_period, earliest_treatment_period + 1, earliest_treatment_period + 2 ]
    data['first_treat_period'] = np.select(conditions_timing, choices_timing, default=np.nan)
    data['post_treatment'] = (data['time'] >= num_pre_periods).astype(int)
    data['eventually_treated'] = (data['treatment_group'] > 0).astype(int)
    data['D'] = (data['time'] >= data['first_treat_period']).astype(int)
    data['time_trend'] = data['time']

    # --- 6. Generate Outcome Variable (Y) with HETEROGENEOUS Effects ---

    # Determine BASE treatment effect based on linearity degree
    if linearity_degree == 1 or linearity_degree == 2: base_treatment_effect_beta = 0
    elif linearity_degree == 3: base_treatment_effect_beta = 0
    else: base_treatment_effect_beta = np.nan

    # Calculate the DYNAMIC treatment effect based on X1 and X3
    effect_modifier_term = np.sqrt(np.abs(data['X4'])) # sqrt(abs(Bernoulli)) is just the Bernoulli value itself, but use formula
    conditions_effect = [
        data['X3'].isin([1, 3]),
        data['X3'] == 2,
        data['X3'] == 4
    ]
    choices_effect = [
        1.5 * effect_modifier_term + base_treatment_effect_beta,
        base_treatment_effect_beta,
        base_treatment_effect_beta - 0.5 * effect_modifier_term
    ]
    data['dynamic_treatment_effect'] = np.select(conditions_effect, choices_effect, default=np.nan)

    # --- DGP Parameters (excluding treatment interaction) ---
    data['epsilon'] = np.random.normal(scale=epsilon_scale, size=len(data))

    beta_0 = -0.5 # Intercept
    beta_group_effect = 0.75 # Main effect of treated group (alpha_i)
    beta_time = 0.2 # Main effect of time trend (gamma_t)
    beta_x = np.array([-0.75, 0.5, -0.5, -1.30, 1.8, 2.5, -1.0, 0.3]) # Fixed covariate effects
    if len(beta_x) != total_covariates: raise ValueError("beta_x length mismatch")


    # Prepare covariate matrix X (order X1 to X8)
    X_cols = [f'X{i}' for i in range(1, total_covariates + 1)]
    X = data[X_cols].values

    # --- Calculate Y based on linearity_degree ---
    Y_base = (beta_0 + beta_group_effect * data['eventually_treated'] + beta_time * data['time_trend'])
    half = total_covariates // 2 # half = 4

    # Calculate Covariate Effects (Y_covariates) based on linearity degree
    if linearity_degree == 1: # Fully Linear
        Y_covariates = np.sum(beta_x * X, axis=1)
    elif linearity_degree == 2: # Half X non-linear
        Y_covariates = (np.sum(beta_x[:2] * (X[:, :2] ** 2), axis=1) +
                        np.sum(beta_x[2:4] * np.exp(X[:, 2:4]), axis=1) +
                        np.sum(beta_x[4:] * X[:, 4:], axis=1))
    elif linearity_degree == 3: 
        Y_base = (beta_0 + beta_group_effect * data['eventually_treated'] + beta_time * data['time_trend']**2)
        Y_covariates = (np.sum(beta_x[:2] * (X[:, :2] ** 2), axis=1) +
                        np.sum(beta_x[2:4] * np.exp(X[:, 2:4]), axis=1) +
                        np.sum(beta_x[4:6] * np.abs(X[:, 4:6]), axis=1) +
                        np.sum(beta_x[6:] * np.sqrt(np.abs(X[:, 6:])), axis=1))
    else:
         Y_covariates = 0


    Y_treatment = data['dynamic_treatment_effect'] * data['D']
        # Calculate CATE
    data['CATE'] = data['dynamic_treatment_effect'] * data['D']


    # Combine components for final Y
    data['Y'] = Y_base + Y_covariates + Y_treatment

    # --- Add pre-trend bias ---
    if pre_trend_bias_delta != 0:
        pre_period_mask = data['time'] < earliest_treatment_period
        bias_mask = pre_period_mask & (data['eventually_treated'] == 1)
        if linearity_degree == 3: # Trigger non-linear pre-trend if degree is 4
            seasonal_amplitude = 1.0; seasonal_period = 4
            seasonal_effect = seasonal_amplitude * np.sin(2 * np.pi * data['time'] / seasonal_period)
            data.loc[bias_mask, 'Y'] += pre_trend_bias_delta * seasonal_effect[bias_mask]
        else:
            time_diff = data['time'] - earliest_treatment_period
            data.loc[bias_mask, 'Y'] += pre_trend_bias_delta * time_diff[bias_mask]

    # Add final error term
    data['Y'] += data['epsilon']

    # --- 7. Finalize DataFrame ---
    final_cols = (['unit_id', 'time', 'treatment_group', 'first_treat_period', 'eventually_treated', 'D','post_treatment'] +
                   X_cols + ['dynamic_treatment_effect', 'Y', 'CATE', 'time_trend', 'epsilon']) # Add dynamic_treatment_effect column
    final_cols = [col for col in final_cols if col in data.columns]
    data = data[final_cols]

    return data


# --- Simulation Parameters ---
num_iterations = 100
# Define the linearity degrees you want to simulate
linearity_degrees = [1, 2, 3]
# Define the main directory to store all results
main_output_dir = './data' # Changed from './data/ATE' to just './data'
# Define the specific subdirectory for this simulation set
simulation_set_dir = os.path.join(main_output_dir, 'CATE_GATE_PS_datasets')

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
        current_data_df = generate_staggered_did_data_hete_effect(
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