import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import shutil
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")
combined_df = pd.concat([train_df, new_df], ignore_index=True)

# Same test set as Task 1
X_orig = train_df.drop("job_completion_min", axis=1)
y_orig = train_df["job_completion_min"]
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

# Champion RMSE from Task 1
champion_model = mlflow.sklearn.load_model("models/best_model")
champion_preds = champion_model.predict(X_test)
champion_rmse = np.sqrt(mean_squared_error(y_test, champion_preds))

# Retrain on combined data
X_combined = combined_df.drop("job_completion_min", axis=1)
y_combined = combined_df["job_completion_min"]

mlflow.set_experiment("gpuforge-job-completion")

with mlflow.start_run(run_name="retrain-combined"):
    retrained_model = GradientBoostingRegressor(random_state=42)
    retrained_model.fit(X_combined, y_combined)
    retrained_preds = retrained_model.predict(X_test)
    retrained_rmse = np.sqrt(mean_squared_error(y_test, retrained_preds))
    mlflow.log_metric("retrained_rmse", retrained_rmse)
    mlflow.log_metric("champion_rmse", champion_rmse)

improvement = champion_rmse - retrained_rmse
min_threshold = 0.5

if improvement >= min_threshold:
    action = "promoted"
    shutil.rmtree("models/best_model", ignore_errors=True)
    mlflow.sklearn.save_model(retrained_model, "models/best_model")
else:
    action = "kept_champion"

output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": round(champion_rmse, 4),
    "retrained_rmse": round(retrained_rmse, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("results", exist_ok=True)
with open("results/step4_retraining.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 4 done!")
print(json.dumps(output, indent=2))