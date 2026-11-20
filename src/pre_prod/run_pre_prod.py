import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MLOps_PreProd_Pipeline")
client = MlflowClient()

MODEL_NAME = "Disease_RF_Model"

# 1. Load model with 'Challenger' alias
model_uri = f"models:/{MODEL_NAME}@Challenger"
loaded_model = mlflow.sklearn.load_model(model_uri)

# 2. Log and register model - assign alias 'Challenger-pre-test'
with mlflow.start_run(run_name="pre_prod_validation"):
    model_info = mlflow.sklearn.log_model(
        loaded_model,
        name="pre_prod_model",
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    new_ver = model_info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "Challenger-pre-test", new_ver)
    print(f"Registered {MODEL_NAME} v{new_ver} with alias 'Challenger-pre-test'")

    # 3. Model Test on holdout/test split
    X_dis, y_dis = load_breast_cancer(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X_dis, y_dis, test_size=0.2, random_state=42)
    test_acc = accuracy_score(y_test, loaded_model.predict(X_test))
    mlflow.log_metric("pre_prod_accuracy", test_acc)

    if test_acc >= 0.85:
        client.set_registered_model_alias(MODEL_NAME, "Challenger-post-test", new_ver)
        print(f"SUCCESS: Model test passed (acc={test_acc:.4f}). Updated alias to 'Challenger-post-test' on v{new_ver}")
    else:
        print(f"FAILED: Model test failed (acc={test_acc:.4f}). Notify through Email.")
