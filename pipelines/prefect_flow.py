from prefect import flow, task
from src import feature_engineering, train, evaluate, register

@task
def run_features():
    return feature_engineering.run_feature_engineering()

@task
def run_training(paths):
    # Note: train.main() loads data from hardcoded paths in contracts.py, 
    # but passing 'paths' ensures Prefect knows this task depends on the previous one.
    train.main()

@task
def run_evaluation():
    # This prints the metrics report required by the project scope
    evaluate.evaluate_model(model_uri=None)

@task
def run_register():
    register.main()

@flow(name="sales-mlops-pipeline")
def main_flow():
    # 1. Preprocess
    paths = run_features()
    
    # 2. Train (Wait for features)
    run_training(paths)
    
    # 3. Evaluate (The Missing Step)
    run_evaluation()
    
    # 4. Register
    run_register()

if __name__ == "__main__":
    main_flow()