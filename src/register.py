import mlflow
from mlflow.tracking import MlflowClient

def main():
    print("📝 [4/4] Promoting Model...")
    client = MlflowClient()
    # Find the run we just created
    runs = client.search_runs(
        [mlflow.get_experiment_by_name("Retail_Sales_Prediction").experiment_id],
        order_by=["metrics.accuracy DESC"], max_results=1
    )
    if runs:
        run_id = runs[0].info.run_id
        mv = mlflow.register_model(f"runs:/{run_id}/model", "sales-quantity-classifier")
        client.set_registered_model_alias("sales-quantity-classifier", "Production", mv.version)
        print("✅ Model promoted to Production.")

if __name__ == "__main__":
    main()