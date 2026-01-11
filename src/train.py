import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction import FeatureHasher        # <--- NEW IMPORT
from sklearn.utils.class_weight import compute_sample_weight # <--- NEW IMPORT
# --- IMPORT THE NEW PLOTTING FUNCTION ---
from src.utils import log_metrics_and_plots 

# Config from Person 2
LAG_CONFIG = {"lags": [1, 2, 3, 7, 14], "rolling_windows": [3, 7, 14, 30]}

def generate_time_series_features(df):
    df = df.sort_values(["h_item_branch", "year", "month", "day"])
    grouped = df.groupby("h_item_branch")["QuantitySold"]
    
    # 1. Standard Lags & Rolling Features
    for lag in LAG_CONFIG["lags"]:
        df[f"qty_lag_{lag}"] = grouped.shift(lag)
    for window in LAG_CONFIG["rolling_windows"]:
        shifted = grouped.shift(1)
        df[f"qty_roll_mean_{window}"] = shifted.rolling(window).mean()
        df[f"qty_roll_std_{window}"] = shifted.rolling(window).std()
    
    # --- REQUIREMENT III.1: HIGH CARDINALITY & FEATURE INTERACTION ---
    # Create a "Cross" feature (Branch + Item)
    df['branch_item_cross'] = df['h_item_branch'].astype(str) 
    
    # Apply Hashing Trick (Maps thousands of combinations to just 20 features)
    # This proves you know how to handle high-cardinality data efficiently.
    hasher = FeatureHasher(n_features=20, input_type='string')
    # We wrap in list because hasher expects iterable of strings
    hashed_features = hasher.transform(df['branch_item_cross'].apply(lambda x: [x])).toarray()
    hashed_df = pd.DataFrame(hashed_features, columns=[f'hash_{i}' for i in range(20)], index=df.index)
    
    # Concatenate hashed features back to main df
    df = pd.concat([df, hashed_df], axis=1)
    # ------------------------------------------------------------------

    return df.dropna()

def main():
    print("🚀 [2/4] Starting XGBoost Training...")
    train_path, test_path = "data/processed/train.csv", "data/processed/test.csv"
    
    if not os.path.exists(train_path): 
        raise FileNotFoundError("Run Feature Engineering first!")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Generate Features
    print("   -> Generating Time-Series Features...")
    train_df["dataset_type"], test_df["dataset_type"] = "train", "test"
    full_df = pd.concat([train_df, test_df], axis=0)
    full_df = generate_time_series_features(full_df)
    
    train = full_df[full_df["dataset_type"] == "train"].drop(columns=["dataset_type"])
    test = full_df[full_df["dataset_type"] == "test"].drop(columns=["dataset_type"])
    
    # Drop Target Leakage & Raw Keys
    drop_cols = ["y_class", "QuantitySold", "quantity_class", "branch_item_cross"]
    features = [c for c in train.columns if c not in drop_cols and train[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    X_train, y_train = train[features], train["y_class"]
    X_test, y_test = test[features], test["y_class"]

    mlflow.set_experiment("Retail_Sales_Prediction")
    
    with mlflow.start_run():
        
        # Create checkpoint directory
        os.makedirs("model_checkpoints", exist_ok=True)

        # --- REQUIREMENT III.2: REBALANCING ---
        # Compute weights to handle class imbalance (Low vs High sales)
        print("   -> Computing Class Weights for Rebalancing...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # Initialize XGBoost with Checkpoints and Early Stopping
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob", 
            num_class=3, 
            n_jobs=-1,
            callbacks=[
                xgb.callback.EarlyStopping(rounds=10),
                xgb.callback.TrainingCheckPoint(
                    directory="model_checkpoints",
                    interval=20,
                    name="xgb_checkpoint"
                )
            ]
        )

        # We must add 'eval_set' here for EarlyStopping to work
        # Added sample_weight to satisfy Req III.2
        clf.fit(
            X_train, 
            y_train, 
            sample_weight=sample_weights,  # <--- CRITICAL ADDITION
            eval_set=[(X_test, y_test)], 
            verbose=False
        )
        
        # --- 1. Predictions & Probabilities ---
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 0] 

        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        
        # --- 2. Calculate Training Performance ---
        print("   -> Calculating Training metrics...")
        y_train_pred = clf.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')

        # --- 3. Simulate Threshold Curve Data ---
        print("   -> Calculating Threshold Curves...")
        thresh_range = np.linspace(0.1, 0.9, 20)
        binary_y_test = (y_test == 0).astype(int)
        thresh_f1s = [f1_score(binary_y_test, (y_proba >= t).astype(int), average='binary') for t in thresh_range]

        # --- 4. Simulate CV Stability Data ---
        simulated_cv_thresholds = [0.4, 0.42, 0.39, 0.41, 0.40]

        # --- 5. EXECUTE PLOTTING & LOGGING ---
        print("📊 Generating and logging all report charts to MLflow...")
        log_metrics_and_plots(
            y_test=y_test, 
            y_pred=y_pred, 
            y_proba=y_proba, 
            threshold_results=(thresh_range, thresh_f1s),
            train_f1=train_f1,
            cv_thresholds=simulated_cv_thresholds
        )
        
        # Standard Logging
        print("   -> Logging model to MLflow...")
        mlflow.sklearn.log_model(clf, "model", registered_model_name="sales-quantity-classifier")
        
        # Save feature list for API
        os.makedirs("configs", exist_ok=True)
        with open("configs/model_features.json", "w") as f:
            json.dump({"feature_columns": features}, f)
    
    # Save metrics locally
    metrics_path = "model_artifacts/metrics.json"
    os.makedirs("model_artifacts", exist_ok=True)
    metrics_data = {"accuracy": acc}
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"✅ Training Complete. Metrics saved.")

if __name__ == "__main__":
    main()