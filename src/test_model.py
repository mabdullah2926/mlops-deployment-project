import sys
import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "DiseaseClassifierModel"
mlflow.set_tracking_uri(TRACKING_URI)

mode = sys.argv[1]  # "run" or "alias"

if mode == "run":
    with open("best_run.txt") as f:
        run_id = f.read().strip()
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
else:
    alias = sys.argv[2]
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")

df = pd.read_csv("data/disease.csv")
X = df.drop(columns=["target"])
y = df["target"]
preds = model.predict(X)
acc = accuracy_score(y, preds)

print(f"Model Test Accuracy: {acc:.4f}")
if acc < 0.85:
    raise ValueError(f"Model test failed! Accuracy {acc:.4f} is below threshold 0.85")
print("Model Test PASSED!")
