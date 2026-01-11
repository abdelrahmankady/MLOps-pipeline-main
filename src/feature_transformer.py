import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.feature_extraction import FeatureHasher  # <--- NEW IMPORT
from src.utils import stable_hash_to_bucket, add_date_features

# Must match the buckets used in feature_engineering.py
BUCKETS = {
    "item": 2000,
    "branch": 200,
    "invoice": 5000,
    "item_branch": 5000,
    "item_month": 5000
}

def transform_raw_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms a single transaction dictionary into a DataFrame 
    with EXACTLY the features expected by the XGBoost model.
    """
    
    # 1. Create Initial DataFrame
    df = pd.DataFrame([data])
    
    # 2. Date Features
    df = add_date_features(df, date_col="Date")
    
    # 3. Hashing Features (Recreating logic from feature_engineering.py)
    df["h_item"] = df["ItemCode"].astype(str).apply(lambda x: stable_hash_to_bucket(x, BUCKETS["item"]))
    df["h_branch"] = df["BranchID"].astype(str).apply(lambda x: stable_hash_to_bucket(x, BUCKETS["branch"]))
    df["h_invoice"] = df["InvoiceNumber"].astype(str).apply(lambda x: stable_hash_to_bucket(x, BUCKETS["invoice"]))
    
    # Cross Features
    df["h_item_branch"] = (df["ItemCode"] + df["BranchID"]).apply(lambda x: stable_hash_to_bucket(x, BUCKETS["item_branch"]))
    df["h_item_month"] = (df["ItemCode"] + df["month"].astype(str)).apply(lambda x: stable_hash_to_bucket(x, BUCKETS["item_month"]))
    
    # 4. Time-Series Features with SMART DEFAULTS
    qty_sold = float(data.get("QuantitySold", 0))
    
    # Lag features
    df['qty_lag_1'] = qty_sold
    df['qty_lag_2'] = qty_sold
    df['qty_lag_3'] = qty_sold
    df['qty_lag_7'] = qty_sold * 0.9 
    df['qty_lag_14'] = qty_sold * 0.8
    
    # Rolling means
    df['qty_roll_mean_3'] = qty_sold
    df['qty_roll_mean_7'] = qty_sold
    df['qty_roll_mean_14'] = qty_sold
    df['qty_roll_mean_30'] = qty_sold
    
    # Rolling std
    df['qty_roll_std_3'] = qty_sold * 0.1
    df['qty_roll_std_7'] = qty_sold * 0.1
    df['qty_roll_std_14'] = qty_sold * 0.1
    df['qty_roll_std_30'] = qty_sold * 0.1

    # --- 5. NEW: Add Feature Hashing (The Fix for the Error) ---
    # We must replicate the hashing logic from train.py exactly
    
    # A. Create the source column
    df['branch_item_cross'] = df['h_item_branch'].astype(str)
    
    # B. Apply Hashing
    hasher = FeatureHasher(n_features=20, input_type='string')
    # Transform needs list of strings, so we apply lambda
    hashed_features = hasher.transform(df['branch_item_cross'].apply(lambda x: [x])).toarray()
    
    # C. Create DataFrame for hashes and join
    hashed_df = pd.DataFrame(hashed_features, columns=[f'hash_{i}' for i in range(20)], index=df.index)
    df = pd.concat([df, hashed_df], axis=1)
    # -----------------------------------------------------------
        
    # 6. Select & Order Columns Exactly as Model Expects
    # Note: We append the 20 hash columns at the end
    expected_cols = [
        'year', 'month', 'day', 'dayofweek', 'is_weekend', 
        'h_item', 'h_branch', 'h_invoice', 'h_item_branch', 'h_item_month',
        'qty_lag_1', 'qty_lag_2', 'qty_lag_3', 'qty_lag_7', 'qty_lag_14',
        'qty_roll_mean_3', 'qty_roll_std_3', 
        'qty_roll_mean_7', 'qty_roll_std_7', 
        'qty_roll_mean_14', 'qty_roll_std_14', 
        'qty_roll_mean_30', 'qty_roll_std_30'
    ] + [f'hash_{i}' for i in range(20)]  # <--- This adds the missing columns
    
    # Ensure purely numeric types for XGBoost
    df_final = df[expected_cols].astype(float)
    
    return df_final