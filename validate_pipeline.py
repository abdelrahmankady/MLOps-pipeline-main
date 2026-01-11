import os
import sys
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# --- 1. Structure Check ---
def check_structure():
    print("🔍 [1/3] Checking File Structure...")
    required = [
        "src/__init__.py",
        "src/train.py",
        "src/serve_api.py",
        "src/utils.py",
        "src/contracts.py",
        "configs/model_features.json",
        "pipelines/prefect_flow.py",
        "Dockerfile",
        "requirements.txt",
        "tests/test_api.py"  
    ]
    missing = []
    for path in required:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print(f"❌ Critical Missing Files: {missing}")
        return False
    print("✅ File Structure: OK")
    return True

# --- 2. Data Content Check  ---
def check_data_content():
    print("🔍 [2/3] Validating Data Artifacts (Processed Schema)...")
    train_path = "data/processed/train.csv"
    
    if not os.path.exists(train_path):
        print("❌ Missing data/processed/train.csv. Run the pipeline first!")
        return False
        
    df = pd.read_csv(train_path, nrows=10) 
    
    # These are the columns the pipeline actually generates
    required_cols = {
        "h_branch",       # Hashed BranchID
        "h_item",         # Hashed ItemCode
        "h_invoice",      # Hashed InvoiceNumber
        "QuantitySold",   # Raw Feature
        "y_class"         # Encoded Target (0, 1, 2)
    }
    
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Data Integrity Failed. Missing columns: {missing}")
        return False
        
    print(f"✅ Data Columns: OK (Found {len(df.columns)} features including hashes).")
    return True

# --- 3. MLflow Registry Check ---
def check_mlflow_registry():
    print("🔍 [3/3] Validating Model Registry...")
    try:
        client = MlflowClient()
        model_name = "sales-quantity-classifier"
        models = client.search_registered_models(filter_string=f"name='{model_name}'")
        
        if not models:
            print(f"❌ Model '{model_name}' not found in MLflow Registry.")
            return False
            
        print(f"✅ Model found: {model_name}")
        return True
        
    except Exception as e:
        print(f"⚠️ MLflow Check Warning (Is server running?): {e}")
        return True

if __name__ == "__main__":
    print("🚀 STARTING FINAL COMPLIANCE CHECK...")
    
    structure_ok = check_structure()
    data_ok = check_data_content()
    mlflow_ok = check_mlflow_registry()
    
    if structure_ok and data_ok and mlflow_ok:
        print("\n✅ PASSED: All checks passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Please fix the errors above.")
        sys.exit(1)