import os
import logging
import mlflow.pyfunc
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.feature_transformer import transform_raw_input

# --- Configuration ---
MODEL_NAME = "sales-quantity-classifier"
MODEL_STAGE = "Production" 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales Quantity Classifier API")

# --- Model Loading Logic ---
class ModelLoader:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the model from MLflow with Debugging."""
        print(f"🔍 DEBUG: Connecting to MLflow tracking URI...")
        try:
            # Load by ALIAS as per MLflow best practices
            model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
            print(f"🔍 DEBUG: Attempting to load: {model_uri}")
            
            # Load the model
            self.model = mlflow.sklearn.load_model(model_uri)
            print("✅ DEBUG: Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ DEBUG: Failed to load model. Error details: {e}")
            self.model = None

    def predict(self, features: pd.DataFrame):
        if not self.model:
            raise RuntimeError("Model not loaded.")
        return self.model.predict(features)

# Initialize Loader
model_loader = ModelLoader()

# --- API Schemas ---
class TransactionInput(BaseModel):
    Date: str
    BranchID: str
    InvoiceNumber: str
    ItemCode: str
    QuantitySold: float
    # Optional fields
    BranchName: Optional[str] = "Unknown"
    ItemName: Optional[str] = "Unknown"

class PredictionOutput(BaseModel):
    prediction: str
    class_id: int
    status: str = "success"  # Added status field for observability
    note: Optional[str] = None

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "active"}

@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(transaction: TransactionInput):
    # Check if model is loaded, but DON'T crash. Use fallback if model is missing.
    if not model_loader.model:
        logger.warning("⚠️ Model missing. Using heuristic fallback.")
        return {
            "prediction": "LOW", 
            "class_id": 0, 
            "status": "fallback", 
            "note": "Model not loaded, used default safety stock."
        }
    
    try:
        # 1. Convert Input to Dict (Updated to model_dump to fix warning)
        data_dict = transaction.model_dump()
        
        # 2. Transform Features
        features_df = transform_raw_input(data_dict)
        
        # 3. Predict
        pred_idx = model_loader.predict(features_df)[0]
        
        # 4. Map to Label
        labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        result_label = labels.get(int(pred_idx), "UNKNOWN")
        
        return {
            "prediction": result_label, 
            "class_id": int(pred_idx),
            "status": "success"
        }

    except Exception as e:
        # --- ALGORITHMIC FALLBACK STRATEGY (Requirement III.3) ---
        # If the ML model crashes (e.g., bad feature conversion), 
        # do NOT return 500 error. Return a safe business default.
        logger.error(f"🚨 Prediction Error: {e}. Switching to Fallback Strategy.")
        
        return {
            "prediction": "LOW",     # Safe default (minimize risk of overstocking)
            "class_id": 0,
            "status": "fallback",
            "note": f"Error during inference: {str(e)}"
        }