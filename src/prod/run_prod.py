import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MLOps_Prod_Pipeline")
client = MlflowClient()

MODEL_NAME = "Disease_RF_Model"

# 1. Load model with 'Challenger-post-test' alias
model_uri = f"models:/{MODEL_NAME}@Challenger-post-test"
prod_model = mlflow.sklearn.load_model(model_uri)

# 2. Log and register model - assign alias 'Champion'
with mlflow.start_run(run_name="prod_deployment"):
    model_info = mlflow.sklearn.log_model(
        prod_model,
        name="prod_model",
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    prod_ver = model_info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "Champion", prod_ver)
    print(f"Registered {MODEL_NAME} v{prod_ver} with alias 'Champion'")

    # 3. Model Deploy / Production smoke check
    X_dis, y_dis = load_breast_cancer(return_X_y=True)
    preds = prod_model.predict(X_dis[:10])
    full_acc = accuracy_score(y_dis, prod_model.predict(X_dis))
    mlflow.log_metric("prod_full_accuracy", full_acc)
    print(f"DEPLOYED: Champion v{prod_ver} ready for inference (sample preds={preds.tolist()}, full_acc={full_acc:.4f})")
