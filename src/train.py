import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Project_02_MLOps")

# 1. Disease Dataset: Logistic Regression & Random Forest Classifier
df_disease = pd.read_csv("data/disease.csv")
X_d = df_disease.drop(columns=["target"])
y_d = df_disease["target"]
X_train, X_test, y_train, y_test = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

classifiers = {
    "LogisticRegression_Disease": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "RandomForest_Disease": RandomForestClassifier(n_estimators=100, random_state=42),
}

best_acc = -1.0
best_run_id = None

for name, clf in classifiers.items():
    with mlflow.start_run(run_name=name) as run:
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        mlflow.log_param("model_type", name)
        mlflow.log_param("dataset", "Disease")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            clf,
            name="model",
            serialization_format="pickle",
        )

        print(f"{name} -> accuracy: {acc:.4f} | run_id: {run.info.run_id}")
        if acc >= best_acc:
            best_acc = acc
            best_run_id = run.info.run_id

# 2. Housing Dataset: Random Forest Regressor
df_boston = pd.read_csv("data/boston.csv").apply(pd.to_numeric, errors="coerce").dropna()
X_b = df_boston.drop(columns=["MEDV"])
y_b = df_boston["MEDV"]
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_b, y_b, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="RandomForest_Boston"):
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(Xb_tr, yb_tr)
    r2 = r2_score(yb_te, reg.predict(Xb_te))

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("dataset", "BostonHousing")
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(
        reg,
        name="model",
        serialization_format="pickle",
    )
    print(f"RandomForest_Boston -> r2_score: {r2:.4f}")

with open("best_run.txt", "w") as f:
    f.write(best_run_id)
print("Saved best_run.txt:", best_run_id)
