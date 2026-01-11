import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from src.utils import add_date_features, stable_hash_to_bucket

DEFAULT_BUCKETS = {"item": 2000, "branch": 200, "invoice": 5000, "item_branch": 5000, "item_month": 5000}

def run_feature_engineering():
    print("🚀 [1/4] Starting Feature Engineering...")
    base_dir = Path(os.getcwd())
    in_path = base_dir / "data" / "raw" / "sales_raw.csv"
    out_train = base_dir / "data" / "processed" / "train.csv"
    out_test = base_dir / "data" / "processed" / "test.csv"
    
    # Check input
    if not in_path.exists(): raise FileNotFoundError(f"Missing {in_path}")
    
    # Load & Clean
    df = pd.read_csv(in_path, sep=";", dtype=str) if in_path.suffix == ".csv" else pd.read_excel(in_path, dtype=str)
    df = df.dropna(subset=["Date"])
    df["QuantitySold"] = pd.to_numeric(df["QuantitySold"], errors='coerce')
    df = df[df["QuantitySold"] > 0]
    
    # Process
    df = add_date_features(df)
    # Add hashes
    df["h_item"] = df["ItemCode"].apply(lambda x: stable_hash_to_bucket(x, DEFAULT_BUCKETS["item"]))
    df["h_branch"] = df["BranchID"].apply(lambda x: stable_hash_to_bucket(x, DEFAULT_BUCKETS["branch"]))
    df["h_invoice"] = df["InvoiceNumber"].apply(lambda x: stable_hash_to_bucket(x, DEFAULT_BUCKETS["invoice"]))
    # Add Crosses
    df["h_item_branch"] = (df["ItemCode"] + df["BranchID"]).apply(lambda x: stable_hash_to_bucket(x, DEFAULT_BUCKETS["item_branch"]))
    df["h_item_month"] = (df["ItemCode"] + df["month"].astype(str)).apply(lambda x: stable_hash_to_bucket(x, DEFAULT_BUCKETS["item_month"]))

    # Target
    low, high = df["QuantitySold"].quantile([0.33, 0.66])
    df["y_class"] = np.where(df["QuantitySold"] <= low, 0, np.where(df["QuantitySold"] <= high, 1, 2))
    
    # Split
    df = df.sort_values("Date")
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    
    # Save
    cols = ["y_class", "QuantitySold", "year", "month", "day", "dayofweek", "is_weekend", "h_item", "h_branch", "h_invoice", "h_item_branch", "h_item_month"]
    os.makedirs(out_train.parent, exist_ok=True)
    train[cols].to_csv(out_train, index=False)
    test[cols].to_csv(out_test, index=False)
    
    # Save Artifacts
    with open(base_dir / "configs" / "feature_artifacts.json", "w") as f:
        json.dump({"buckets": DEFAULT_BUCKETS, "thresholds": {"low": low, "high": high}}, f)
        
    return {"train_path": str(out_train), "test_path": str(out_test)}

if __name__ == "__main__":
    run_feature_engineering()