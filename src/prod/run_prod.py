import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.metrics import accuracy_score, r2_score

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("MLOps_Prod_Pipeline")
client = MlflowClient(tracking_uri=TRACKING_URI)

X_clf, y_clf = load_breast_cancer(return_X_y=True)
X_reg, y_reg = fetch_california_housing(return_X_y=True)

models_cfg = [
    ("Disease_RF_Model", "clf", X_clf, y_clf),
    ("Disease_LR_Model", "clf", X_clf, y_clf),
    ("Boston_RF_Model", "reg", X_reg, y_reg),
]

for model_name, task_type, X_data, y_data in models_cfg:
    model_uri = f"models:/{model_name}@Challenger-post-test"
    prod_model = mlflow.sklearn.load_model(model_uri)

    with mlflow.start_run(run_name=f"prod_deploy_{model_name}"):
        model_info = mlflow.sklearn.log_model(
            sk_model=prod_model,
            name="model",
            registered_model_name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        prod_ver = model_info.registered_model_version
        client.set_registered_model_alias(model_name, "Champion", prod_ver)
        print(f"Registered {model_name} v{prod_ver} with alias 'Champion'")

        preds = prod_model.predict(X_data[:5])
        all_preds = prod_model.predict(X_data)
        if task_type == "clf":
            metric_val = accuracy_score(y_data, all_preds)
            mlflow.log_metric("prod_full_accuracy", metric_val)
        else:
            metric_val = r2_score(y_data, all_preds)
            mlflow.log_metric("prod_full_r2", metric_val)

        print(f"DEPLOYED: {model_name} Champion v{prod_ver} (sample={preds.tolist()}, score={metric_val:.4f})")
