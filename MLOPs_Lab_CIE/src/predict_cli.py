import argparse
import mlflow.sklearn
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--network_congestion", type=float, required=True)
parser.add_argument("--txn_size_bytes", type=float, required=True)
parser.add_argument("--is_priority", type=float, required=True)
parser.add_argument("--block_fullness_pct", type=float, required=True)
args = parser.parse_args()

model = mlflow.sklearn.load_model("models/best_model")

data = pd.DataFrame([{
    "network_congestion": args.network_congestion,
    "txn_size_bytes": args.txn_size_bytes,
    "is_priority": args.is_priority,
    "block_fullness_pct": args.block_fullness_pct
}])

pred = model.predict(data)[0]
print(json.dumps({"prediction": round(float(pred), 4)}))