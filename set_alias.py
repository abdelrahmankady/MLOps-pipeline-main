import mlflow
from mlflow import MlflowClient

# Initialize Client
client = MlflowClient()
model_name = "sales-quantity-classifier"

# Get the latest version of the model you just trained
latest_version_info = client.get_latest_versions(model_name, stages=["None", "Staging", "Production", "Archived"])[-1]
latest_version = latest_version_info.version

# Set the alias "Production" to this version
client.set_registered_model_alias(model_name, "Production", latest_version)

print(f"✅ Fixed! Version {latest_version} is now marked as 'Production'.")