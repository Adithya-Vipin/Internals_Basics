import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("data/training_data.csv")
X = df.drop("gas_fee_gwei", axis=1)
y = df["gas_fee_gwei"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("coinpulse-gas-fee-gwei")

results = []

# Train Lasso
with mlflow.start_run(run_name="Lasso"):
    model = Lasso(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mlflow.log_param("alpha", 1.0)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.set_tag("experiment_type", "baseline_comparison")
    results.append({"name": "Lasso", "mae": round(mae,4), "rmse": round(rmse,4)})

# Train GradientBoosting
with mlflow.start_run(run_name="GradientBoosting"):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.set_tag("experiment_type", "baseline_comparison")
    results.append({"name": "GradientBoosting", "mae": round(mae,4), "rmse": round(rmse,4)})

# Save best model
best = min(results, key=lambda x: x["rmse"])
mlflow.sklearn.save_model(model, "models/best_model")

output = {
    "experiment_name": "coinpulse-gas-fee-gwei",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 1 done!")
print(json.dumps(output, indent=2))