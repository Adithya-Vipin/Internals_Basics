import mlflow.sklearn
import pandas as pd
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

app = FastAPI()
model = mlflow.sklearn.load_model("models/best_model")

class Transaction(BaseModel):
    network_congestion: float
    txn_size_bytes: float
    is_priority: float
    block_fullness_pct: float

    @validator("network_congestion")
    def check_network(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("network_congestion must be 0.0-1.0")
        return v

    @validator("txn_size_bytes")
    def check_txn(cls, v):
        if not 200 <= v <= 2000:
            raise ValueError("txn_size_bytes must be 200-2000")
        return v

    @validator("is_priority")
    def check_priority(cls, v):
        if v not in [0, 1]:
            raise ValueError("is_priority must be 0 or 1")
        return v

    @validator("block_fullness_pct")
    def check_block(cls, v):
        if not 50 <= v <= 100:
            raise ValueError("block_fullness_pct must be 50-100")
        return v

@app.get("/health")
def health():
    return {"status": "operational", "service": "CoinPulse API"}

@app.post("/forecast")
def forecast(txn: Transaction):
    data = pd.DataFrame([txn.dict()])
    pred = model.predict(data)[0]
    return {"prediction": round(float(pred), 4)}