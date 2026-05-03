import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/training_data.csv")
X = df.drop("job_completion_min", axis=1)
y = df["job_completion_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("gpuforge-job-completion")

results = []

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Train Ridge
with mlflow.start_run(run_name="Ridge"):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape_val = mape(y_test.values, preds)
    mlflow.log_param("alpha", 1.0)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape_val)
    mlflow.set_tag("experiment_type", "baseline_comparison")
    results.append({"name": "Ridge", "mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4), "mape": round(mape_val,4)})

# Train GradientBoosting
with mlflow.start_run(run_name="GradientBoosting"):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape_val = mape(y_test.values, preds)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape_val)
    mlflow.set_tag("experiment_type", "baseline_comparison")
    results.append({"name": "GradientBoosting", "mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4), "mape": round(mape_val,4)})

# Pick best model
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "gpuforge-job-completion",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

os.makedirs("results", exist_ok=True)
with open("results/step1_tracking.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 1 done!")
print(json.dumps(output, indent=2))