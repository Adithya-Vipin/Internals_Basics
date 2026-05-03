import requests
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Load training data to get means
df = pd.read_csv("data/training_data.csv")
train_mean_params = df["model_params_millions"].mean()
train_mean_queue = df["queue_depth"].mean()

# Normal requests (from training data ranges)
normal_inputs = [
    {"gpu_memory_gb": 40, "batch_size": 64, "model_params_millions": 1500, "queue_depth": 8},
    {"gpu_memory_gb": 16, "batch_size": 32, "model_params_millions": 125, "queue_depth": 3},
    {"gpu_memory_gb": 80, "batch_size": 256, "model_params_millions": 7000, "queue_depth": 18},
    {"gpu_memory_gb": 24, "batch_size": 32, "model_params_millions": 250, "queue_depth": 5},
    {"gpu_memory_gb": 64, "batch_size": 128, "model_params_millions": 3500, "queue_depth": 12},
    {"gpu_memory_gb": 16, "batch_size": 64, "model_params_millions": 175, "queue_depth": 2},
    {"gpu_memory_gb": 48, "batch_size": 64, "model_params_millions": 1000, "queue_depth": 9},
    {"gpu_memory_gb": 32, "batch_size": 128, "model_params_millions": 800, "queue_depth": 6},
    {"gpu_memory_gb": 8, "batch_size": 8, "model_params_millions": 30, "queue_depth": 1},
    {"gpu_memory_gb": 72, "batch_size": 256, "model_params_millions": 5000, "queue_depth": 15},
] * 4  # 40 normal requests

# Drifted requests (from new_data ranges)
drifted_inputs = [
    {"gpu_memory_gb": 80, "batch_size": 256, "model_params_millions": 7000, "queue_depth": 20},
    {"gpu_memory_gb": 64, "batch_size": 256, "model_params_millions": 5500, "queue_depth": 18},
    {"gpu_memory_gb": 72, "batch_size": 128, "model_params_millions": 6500, "queue_depth": 19},
    {"gpu_memory_gb": 80, "batch_size": 128, "model_params_millions": 7000, "queue_depth": 17},
    {"gpu_memory_gb": 48, "batch_size": 256, "model_params_millions": 4000, "queue_depth": 16},
    {"gpu_memory_gb": 64, "batch_size": 64, "model_params_millions": 5000, "queue_depth": 15},
    {"gpu_memory_gb": 72, "batch_size": 256, "model_params_millions": 6000, "queue_depth": 18},
    {"gpu_memory_gb": 80, "batch_size": 128, "model_params_millions": 6800, "queue_depth": 20},
    {"gpu_memory_gb": 56, "batch_size": 128, "model_params_millions": 4500, "queue_depth": 14},
    {"gpu_memory_gb": 64, "batch_size": 256, "model_params_millions": 5200, "queue_depth": 16},
]  # 10 drifted requests

all_inputs = normal_inputs + drifted_inputs

# Send requests and log
os.makedirs("logs", exist_ok=True)
predictions = []

with open("logs/predictions.jsonl", "w") as f:
    for inp in all_inputs:
        resp = requests.post("http://127.0.0.1:8500/estimate", json=inp).json()
        pred = resp["prediction"]
        predictions.append(pred)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": inp,
            "prediction": pred,
            "endpoint": "/estimate"
        }
        f.write(json.dumps(log_entry) + "\n")

# Monitor drift
live_mean_params = np.mean([x["model_params_millions"] for x in all_inputs])
live_mean_queue = np.mean([x["queue_depth"] for x in all_inputs])

shift_params = abs(live_mean_params - train_mean_params)
shift_queue = abs(live_mean_queue - train_mean_queue)

alerts = [
    {
        "feature": "model_params_millions",
        "train_mean": round(train_mean_params, 4),
        "live_mean": round(live_mean_params, 4),
        "shift": round(shift_params, 4),
        "threshold": 500,
        "status": "ALERT" if shift_params > 500 else "OK"
    },
    {
        "feature": "queue_depth",
        "train_mean": round(train_mean_queue, 4),
        "live_mean": round(live_mean_queue, 4),
        "shift": round(shift_queue, 4),
        "threshold": 5,
        "status": "ALERT" if shift_queue > 5 else "OK"
    }
]

drift_detected = any(a["status"] == "ALERT" for a in alerts)

output = {
    "total_predictions": len(predictions),
    "mean_prediction": round(np.mean(predictions), 4),
    "drift_detected": drift_detected,
    "alerts": alerts
}

with open("results/step3_monitoring.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 3 done!")
print(json.dumps(output, indent=2))