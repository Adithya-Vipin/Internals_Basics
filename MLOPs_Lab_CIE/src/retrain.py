import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import shutil
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")
combined_df = pd.concat([train_df, new_df], ignore_index=True)

# Same test set as Task 1
X_orig = train_df.drop("gas_fee_gwei", axis=1)
y_orig = train_df["gas_fee_gwei"]
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

# Champion MAE from Task 1
champion_model = mlflow.sklearn.load_model("models/best_model")
champion_preds = champion_model.predict(X_test)
champion_mae = mean_absolute_error(y_test, champion_preds)

# Retrain on combined data
X_combined = combined_df.drop("gas_fee_gwei", axis=1)
y_combined = combined_df["gas_fee_gwei"]

mlflow.set_experiment("coinpulse-gas-fee-gwei")

with mlflow.start_run(run_name="retrain-combined"):
    retrained_model = Lasso(alpha=1.0)
    retrained_model.fit(X_combined, y_combined)
    retrained_preds = retrained_model.predict(X_test)
    retrained_mae = mean_absolute_error(y_test, retrained_preds)
    mlflow.log_metric("retrained_mae", retrained_mae)
    mlflow.log_metric("champion_mae", champion_mae)

improvement = champion_mae - retrained_mae

if improvement >= 0.5:
    action = "promoted"
    shutil.rmtree("models/best_model", ignore_errors=True)
    mlflow.sklearn.save_model(retrained_model, "models/best_model")
else:
    action = "kept_champion"

output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_mae": round(champion_mae, 4),
    "retrained_mae": round(retrained_mae, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "mae"
}

os.makedirs("results", exist_ok=True)
with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 4 done!")
print(json.dumps(output, indent=2))