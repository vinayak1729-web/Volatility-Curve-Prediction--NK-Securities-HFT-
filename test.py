import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

# Load test data
print("Loading test data from 'data/test-data.csv'...")
test_data = pd.read_csv('data/test-data.csv', na_values='\\N')
print(f"Test data loaded: {test_data.shape[0]} rows, {test_data.shape[1]} columns")

# Load sample submission to inspect its structure
print("Loading sample submission file...")
sample_submission = pd.read_csv('data/sample_submission.csv')
print(f"Sample submission loaded: {sample_submission.shape[0]} rows")
print(f"Sample submission columns: {list(sample_submission.columns)}")

# Pre-process IVs: Stricter outlier removal and log-transform
print("Pre-processing IV data...")
call_iv_cols = [col for col in test_data.columns if col.startswith('call_iv_')]
put_iv_cols = [col for col in test_data.columns if col.startswith('put_iv_')]
iv_columns = call_iv_cols + put_iv_cols

# Clip IVs to [0, 0.5] and log-transform
test_data_clean = test_data.copy()
for col in iv_columns:
    test_data_clean[col] = test_data_clean[col].clip(lower=0, upper=0.5)
    test_data_clean[col] = np.log1p(test_data_clean[col])  # Log-transform to handle outliers
print("IV data pre-processed: clipped and log-transformed")

# Compute global median log(IV) per strike as fallback
print("Computing global median log(IV) per strike...")
global_median_ivs = test_data_clean[iv_columns].median().fillna(0)

# Compute row-level adjustment from X0 to X41
print("Computing row-level adjustment from X features...")
feature_cols = [f'X{i}' for i in range(42)]
# Mean of non-zero X values per row
x_adjustments = test_data_clean[feature_cols].apply(lambda row: row[row != 0].mean() if row[row != 0].size > 0 else 1, axis=1).fillna(1)
test_data_clean['x_adjust'] = x_adjustments.clip(lower=0.5, upper=2.0)  # Cap adjustment to avoid extreme scaling

# Function to extract strike price from column name
def extract_strike(col_name):
    return int(col_name.split('_')[-1])

# Function to predict IVs for a single row
def predict_row(row, iv_columns, global_median_ivs):
    U = row['underlying']
    # Collect available (moneyness, log(IV)) pairs
    available_ivs = []
    for col in iv_columns:
        if not pd.isna(row[col]):
            K = extract_strike(col)
            k = np.log(K / U)
            available_ivs.append((k, row[col]))
    
    # Compute row's IV level for scaling
    row_iv_level = np.median([iv for _, iv in available_ivs]) if available_ivs else 0
    
    # Predict base IVs using quadratic fit
    if len(available_ivs) < 3:  # Need at least 3 points for quadratic fit
        # Fallback: Use global median log(IV), scaled by row's IV level and X adjustment
        predictions = {}
        for col in iv_columns:
            if pd.isna(row[col]):
                pred_log_iv = global_median_ivs[col]
                # Scale by row's IV level and X adjustment
                if row_iv_level > 0:
                    pred_log_iv = pred_log_iv * (row_iv_level / global_median_ivs[col]) if global_median_ivs[col] > 0 else pred_log_iv
                pred_log_iv = pred_log_iv * row['x_adjust']
                pred_iv = np.expm1(pred_log_iv)  # Reverse log-transform
                predictions[col] = max(pred_iv, 0)
    else:
        # Fit quadratic model to log(IVs)
        k_values = np.array([k for k, _ in available_ivs]).reshape(-1, 1)
        log_iv_values = np.array([iv for _, iv in available_ivs])
        
        model = make_pipeline(PolynomialFeatures(degree=2), HuberRegressor())
        model.fit(k_values, log_iv_values)
        
        # Predict missing log(IVs)
        predictions = {}
        for col in iv_columns:
            if pd.isna(row[col]):
                K = extract_strike(col)
                k = np.log(K / U)
                pred_log_iv = model.predict([[k]])[0]
                pred_log_iv = pred_log_iv * row['x_adjust']
                pred_iv = np.expm1(pred_log_iv)  # Reverse log-transform
                predictions[col] = max(pred_iv, 0)
    
    return predictions

# Generate predictions with progress bar
print("Predicting missing IVs for each row...")
submission_data = test_data[['timestamp']].copy()
for col in iv_columns:
    submission_data[col] = test_data[col]  # Copy original IVs

for idx, row in tqdm(test_data_clean.iterrows(), total=test_data_clean.shape[0], desc="Processing rows"):
    row_predictions = predict_row(row, iv_columns, global_median_ivs)
    for col, pred in row_predictions.items():
        submission_data.loc[idx, col] = pred
print(f"Updated {submission_data.shape[0]} rows with predictions")

# Ensure submission matches sample_submission columns
print("Aligning submission with sample submission format...")
submission_cols = ['timestamp'] + [col for col in sample_submission.columns if col != 'timestamp']
submission = submission_data[submission_cols]
print(f"Submission DataFrame created with {submission.shape[0]} rows and {submission.shape[1]} columns")

# Save submission
print("Saving submission to 'ttestsubmission.csv'...")
submission.to_csv('ttestsubmission.csv', index=False)
print("Submission file 'ttestsubmission.csv' has been generated.")