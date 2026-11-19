import os
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing

os.makedirs("data", exist_ok=True)

# 1. Disease Dataset (Classification)
disease = load_breast_cancer(as_frame=True)
disease.frame.to_csv("data/disease.csv", index=False)

# 2. Housing Dataset for Regression (saved as data/boston.csv with MEDV target)
housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame.rename(columns={"MedHouseVal": "MEDV"})
housing_df.to_csv("data/boston.csv", index=False)

print("Done: data/disease.csv and data/boston.csv saved.")
