# Use Python 3.11 (Compatible with your MLflow version)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and data
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# --- THE FIX: Train the model INSIDE the container ---
# This generates a fresh mlflow.db with correct Linux paths
RUN python -m pipelines.prefect_flow

# Expose port and start
EXPOSE 8000
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]