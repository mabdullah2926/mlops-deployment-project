import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("MLOps_PreProd_Pipeline")
client = MlflowClient(tracking_uri=TRACKING_URI)

X_clf, y_clf = load_breast_cancer(return_X_y=True)
_, X_clf_test, _, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

X_reg, y_reg = fetch_california_housing(return_X_y=True)
_, X_reg_test, _, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

models_cfg = [
    ("Disease_RF_Model", "clf", 0.85),
    ("Disease_LR_Model", "clf", 0.85),
    ("Boston_RF_Model", "reg", 0.50),
]

for model_name, task_type, threshold in models_cfg:
    model_uri = f"models:/{model_name}@Challenger"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    with mlflow.start_run(run_name=f"pre_prod_{model_name}"):
        model_info = mlflow.sklearn.log_model(
            sk_model=loaded_model,
            name="model",
            registered_model_name=model_name,
        )
        new_ver = model_info.registered_model_version
        client.set_registered_model_alias(model_name, "Challenger-pre-test", new_ver)
        print(f"Registered {model_name} v{new_ver} with alias 'Challenger-pre-test'")

        if task_type == "clf":
            preds = loaded_model.predict(X_clf_test)
            score = accuracy_score(y_clf_test, preds)
            mlflow.log_metric("pre_prod_accuracy", score)
        else:
            preds = loaded_model.predict(X_reg_test)
            score = r2_score(y_reg_test, preds)
            mlflow.log_metric("pre_prod_r2", score)

        passed = score >= threshold
        mlflow.log_param("test_passed", passed)
        if not passed:
            raise RuntimeError(f"Pre-prod test failed for {model_name} (score={score:.4f} < {threshold}). Notify through Email.")

        client.set_registered_model_alias(model_name, "Challenger-post-test", new_ver)
        print(f"SUCCESS: {model_name} v{new_ver} passed test (score={score:.4f}) -> alias 'Challenger-post-test'")
