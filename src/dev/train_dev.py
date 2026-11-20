import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, r2_score

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("MLOps_Dev_Pipeline")
client = MlflowClient()

X_dis, y_dis = load_breast_cancer(return_X_y=True)
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_dis, y_dis, test_size=0.2, random_state=42)

X_hou, y_hou = fetch_california_housing(return_X_y=True)
X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(X_hou[:2000], y_hou[:2000], test_size=0.2, random_state=42)

with mlflow.start_run(run_name="rf_disease_dev"):
    rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_clf.fit(X_tr_d, y_tr_d)
    acc = accuracy_score(y_te_d, rf_clf.predict(X_te_d))
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", acc)

    if acc >= 0.80:
        model_info = mlflow.sklearn.log_model(
            rf_clf,
            name="model",
            registered_model_name="Disease_RF_Model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        client.set_registered_model_alias("Disease_RF_Model", "Challenger", model_info.registered_model_version)
        print(f"SUCCESS: Disease_RF_Model v{model_info.registered_model_version} -> alias 'Challenger' (acc={acc:.4f})")
    else:
        print(f"FAILED: Model test failed (acc={acc:.4f}). Trigger email notification step.")

with mlflow.start_run(run_name="logreg_disease_dev"):
    lr_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    lr_clf.fit(X_tr_d, y_tr_d)
    acc_lr = accuracy_score(y_te_d, lr_clf.predict(X_te_d))
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc_lr)
    mlflow.sklearn.log_model(lr_clf, name="model")
    print(f"Logged LogisticRegression accuracy={acc_lr:.4f}")

with mlflow.start_run(run_name="rf_housing_dev"):
    rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_reg.fit(X_tr_h, y_tr_h)
    r2 = r2_score(y_te_h, rf_reg.predict(X_te_h))
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(rf_reg, name="housing_model", skops_trusted_types=["sklearn.tree._tree.Tree"])
    print(f"Logged Housing R2={r2:.4f}")
