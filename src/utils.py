from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. Feature Engineering Functions (EXISTING)
# ==========================================

def stable_hash_to_bucket(value, num_buckets: int, salt: str = "") -> int:
    """Deterministic hash -> [0, num_buckets)."""
    if not isinstance(num_buckets, int) or num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive int, got {num_buckets}")

    s = (salt + str(value)).encode("utf-8")
    return int(hashlib.md5(s).hexdigest(), 16) % num_buckets


def add_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add basic time features from a date column."""
    if date_col not in df.columns:
        raise ValueError(f"Missing column: {date_col}")

    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        before = len(out)
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col])
        if len(out) < before:
            print(f"⚠️ Dropped {before - len(out)} invalid dates")

    out["year"] = out[date_col].dt.year.astype(int)
    out["month"] = out[date_col].dt.month.astype(int)
    out["day"] = out[date_col].dt.day.astype(int)
    out["dayofweek"] = out[date_col].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    return out


def add_hashed_features(
    df: pd.DataFrame,
    buckets: Dict[str, int],
    item_col: str = "ItemCode",
    branch_col: str = "BranchID",
    invoice_col: str = "InvoiceNumber",
) -> pd.DataFrame:
    """Add hashed ID features and simple crosses."""
    required = ["item", "branch", "invoice", "item_branch", "item_month"]
    for k in required:
        if k not in buckets or buckets[k] <= 0:
            raise ValueError(f"Invalid bucket for '{k}'")

    if "month" not in df.columns:
        raise ValueError("Run add_date_features() before hashing")

    out = df.copy()

    out[item_col] = out[item_col].astype(str)
    out[branch_col] = out[branch_col].astype(str)
    out[invoice_col] = out[invoice_col].astype(str)

    out["h_item"] = out[item_col].map(lambda x: stable_hash_to_bucket(x, buckets["item"], "item:"))
    out["h_branch"] = out[branch_col].map(lambda x: stable_hash_to_bucket(x, buckets["branch"], "branch:"))
    out["h_invoice"] = out[invoice_col].map(lambda x: stable_hash_to_bucket(x, buckets["invoice"], "inv:"))

    out["h_item_branch"] = (
        out[item_col].str.cat(out[branch_col], sep="|")
        .map(lambda x: stable_hash_to_bucket(x, buckets["item_branch"], "ib:"))
    )

    out["h_item_month"] = (
        out[item_col].str.cat(out["month"].astype(str), sep="|")
        .map(lambda x: stable_hash_to_bucket(x, buckets["item_month"], "im:"))
    )

    return out


# ==========================================
# 2. Reporting & Plotting Functions (NEW)
# ==========================================

def log_metrics_and_plots(
    y_test: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: np.ndarray, 
    threshold_results: Optional[Tuple[List[float], List[float]]] = None,
    train_f1: Optional[float] = None,       # NEW: For Page 19
    cv_thresholds: Optional[List[float]] = None # NEW: For Page 20
):
    """
    Generates and logs ALL 8 charts/metrics required by the report.
    """
    
    # --- A. Standard Metrics (Page 25) ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"📊 Logging Metrics: Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}, F1={f1:.2f}")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # --- B. Metrics Bar Chart (Page 25) ---
    metrics_dict = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylim(0, 1.0)
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', fontweight='bold')
    mlflow.log_figure(plt.gcf(), "evaluation_metrics_barchart.png")
    plt.close()

    # --- C. Confusion Matrix (Page 13/15) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()

    # --- D. Probability Histogram (Page 14) ---
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title("P(LOW) Prediction Probabilities Histogram")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    mlflow.log_figure(plt.gcf(), "probability_histogram.png")
    plt.close()

    # --- E. Prediction Confidence Charts (Page 20) ---
    mean_prob = np.mean(y_proba)
    plt.figure(figsize=(6, 2))
    plt.barh(['Mean Probability'], [mean_prob], color='#1f77b4')
    plt.xlim(0, 1)
    plt.title(f"Mean Predicted Probability: {mean_prob:.2f}")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "mean_probability_chart.png")
    plt.close()

    std_prob = np.std(y_proba)
    plt.figure(figsize=(6, 2))
    plt.barh(['Std Deviation'], [std_prob], color='#1f77b4')
    plt.xlim(0, 0.5)
    plt.title(f"Std Deviation of Probability: {std_prob:.2f}")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "std_probability_chart.png")
    plt.close()

    # --- F. Threshold Curve (Page 12/18) ---
    if threshold_results:
        thresholds, f1_scores = threshold_results
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, linewidth=2, marker='o', markersize=4)
        plt.title("CV mean F1 vs Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Mean F1 Score")
        plt.grid(True)
        mlflow.log_figure(plt.gcf(), "threshold_f1_curve.png")
        plt.close()

    # --- G. Training Performance Chart (Page 19) ---
    if train_f1 is not None:
        plt.figure(figsize=(6, 4))
        plt.bar(["Train F1"], [train_f1], color="#1f77b4")
        plt.ylim(0, 1.0)
        plt.title("Training Performance (F1 Score)")
        plt.ylabel("Score")
        plt.text(0, train_f1 + 0.02, f"{train_f1:.2f}", ha='center', fontweight='bold')
        mlflow.log_figure(plt.gcf(), "training_performance_f1.png")
        plt.close()

    # --- H. CV Stability Chart (Page 20) ---
    if cv_thresholds:
        plt.figure(figsize=(8, 4))
        folds = [f"Fold {i+1}" for i in range(len(cv_thresholds))]
        plt.plot(folds, cv_thresholds, marker='o', linestyle='-', color='#1f77b4')
        plt.title("Cross-Validation Stability Across Folds (Best Threshold)")
        plt.ylabel("Best Threshold")
        plt.ylim(0, 1.0)
        plt.grid(True)
        mlflow.log_figure(plt.gcf(), "cv_stability_folds.png")
        plt.close()