import sys
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "DiseaseClassifierModel"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

action = sys.argv[1]  # register_challenger | pre_test | post_test | champion

if action == "register_challenger":
    with open("best_run.txt") as f:
        run_id = f.read().strip()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "Challenger", mv.version)
    print(f"Registered version {mv.version} with alias 'Challenger'")

elif action == "pre_test":
    mv = client.get_model_version_by_alias(MODEL_NAME, "Challenger")
    client.set_registered_model_alias(MODEL_NAME, "Challenger-pre-test", mv.version)
    print(f"Version {mv.version} assigned alias 'Challenger-pre-test'")

elif action == "post_test":
    mv = client.get_model_version_by_alias(MODEL_NAME, "Challenger-pre-test")
    client.set_registered_model_alias(MODEL_NAME, "Challenger-post-test", mv.version)
    print(f"Version {mv.version} updated to alias 'Challenger-post-test'")

elif action == "champion":
    mv = client.get_model_version_by_alias(MODEL_NAME, "Challenger-post-test")
    client.set_registered_model_alias(MODEL_NAME, "Champion", mv.version)
    print(f"Version {mv.version} promoted to 'Champion'")
