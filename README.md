```markdown
# 📈 Retail Sales Prediction System (End-to-End MLOps)

A production-grade MLOps system demonstrating **Level 2 Maturity (CI/CD Automation)**. This project predicts sales quantity categories (`LOW`, `MEDIUM`, `HIGH`) for high-cardinality retail data using a resilient, containerized architecture.

---

## 🚀 Key Technical Features (MLOps Components)

### 1. 🧠 Advanced Modeling & Feature Engineering
* **High Cardinality Handling:** Implements **Feature Hashing** (20-bucket strategy) to handle thousands of `ItemCode` and `BranchID` combinations without exploding memory.
* **Feature Crosses:** Captures interactions between `Branch` and `Item` explicitly.
* **Class Rebalancing:** Uses **Weighted Loss Functions** in XGBoost to handle class imbalance.

### 2. 🛡️ Resilience & Reliability
* **Algorithmic Fallback:** The API implements a `try-except` safety net. If the ML model fails, a heuristic fallback (`LOW` stock safe-mode) is triggered.
* **Training Checkpoints:** Saves model state every 20 iterations (`model_checkpoints/`) to allow resumption and fault tolerance.
* **Input Validation:** Strict Pydantic schemas ensure data integrity before inference.

### 3. 👁️ Continuous Evaluation & Monitoring
* **Drift Detection:** Statistical monitoring (**Kolmogorov-Smirnov** and **Chi-Square** tests) compares incoming production data against training baselines to detect concept drift.
* **Automated Testing:** CI pipeline runs unit tests (`pytest`) and health checks on every commit.

---

## 📂 Project Structure

```text
MLOPS-PIPELINE/
├── .github/workflows/   # CI/CD Pipeline (GitHub Actions)
├── configs/             # JSON configs for features and thresholds
├── data/                # Raw & Processed datasets
├── docs/                # Architecture diagrams & reports
│   ├── 01_experiments.png
│   ├── 02_run_details.png
│   ├── 03_model_registry.png
│   └── 04_production_alias.png
├── mlruns/              # MLflow local tracking store
├── mlruns_docker/       # MLflow store for Dockerized runs
├── model_artifacts/     # Serialized metrics and plots
├── model_checkpoints/   # XGBoost checkpoints (fault tolerance)
├── monitoring/          # Drift detection scripts
│   └── data_drift_monitor.py
├── pipelines/           # Prefect orchestration flows
│   └── prefect_flow.py
├── src/                 # Source Code
│   ├── feature_engineering.py  # Hashing & transformations
│   ├── serve_api.py            # FastAPI serving + fallback
│   ├── train.py                # XGBoost training
│   ├── utils.py                # Metrics & plotting utilities
│   ├── feature_transformer.py  # Input validation helper
│   ├── evaluate.py             # Model evaluation logic
│   └── register.py             # Model registry logic
├── tests/               # Integration tests
│   └── test_api.py
├── .dockerignore        # Docker exclusion rules
├── .gitignore           # Git exclusion rules
├── Dockerfile           # Train-on-build image
├── mlflow.db            # Local SQLite database for MLflow
├── requirements.txt     # Python dependencies
├── set_alias.py         # Script to tag models as "Production"
└── validate_pipeline.py # System health check script

```

---

## 🐳 Quick Start (Production Mode via Docker)

This image follows the **"Train-on-Build"** pattern to ensure the model matches the exact code version.

### 1️⃣ Build the Image

```bash
# This executes the full training pipeline inside the container
docker build --no-cache -t sales-classifier:prod .

```

### 2️⃣ Run the API

```bash
docker run -p 8000:8000 sales-classifier:prod

```

Access the API documentation at: `http://localhost:8000/docs`

---

## 🛠️ Development Guide (Local Execution)

### 1️⃣ Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2️⃣ Run the Orchestrated Pipeline

We use **Prefect** to manage the workflow DAG.

```bash
python -m pipelines.prefect_flow

```

### 3️⃣ Run Drift Monitoring

Check if the test data has statistically drifted from the training baseline.

```bash
python monitoring/data_drift_monitor.py

```

### 4️⃣ Start Local API

```bash
uvicorn src.serve_api:app --host 0.0.0.0 --port 8000 --reload

```

---

## 📊 Pipeline Results

* **Accuracy:** ~51% (vs 33% random baseline)
* **F1 Score:** ~0.52 (Balanced across classes)
* **Latency:** < 50ms per prediction

---

## ⚡ API Usage Example

```bash
1. Test for LOW Category (Class 0)Criteria: Quantity <= 3.0
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "Date": "2025-12-30",
           "BranchID": "1",
           "InvoiceNumber": "INV-LOW-001",
           "ItemCode": "10001",
           "QuantitySold": 1.5
         }'

2. Test for HIGH Category (Class 2)Criteria: Quantity > 8.0
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "Date": "2025-12-30",
           "BranchID": "10",
           "InvoiceNumber": "INV-HIGH-003",
           "ItemCode": "30003",
           "QuantitySold": 15.0
         }'
```

--- 

# 📊 Model Governance & Monitoring

# This project uses MLflow as the central hub for experiment tracking and stage management. 
# To view the model registry and performance metrics:

# 1. Start the MLflow UI:
# Run this command in the project root:
mlflow ui --port 5000

# 2. Access the Dashboard:
# Navigate to http://127.0.0.1:5000 in your web browser.

# 3. What to Look For:
# - Experiments Tab: View every logged parameter, metric (Accuracy, F1), and model version.
# - Models Tab (Registry): Observe the Model Registry where we demonstrate stage management 
#   by promoting models to the @production alias.