import mlflow.sklearn
import pandas as pd
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import uvicorn

app = FastAPI()

# Load best model
model = mlflow.sklearn.load_model("models/best_model")

class JobRequest(BaseModel):
    gpu_memory_gb: float
    batch_size: float
    model_params_millions: float
    queue_depth: float

    @validator("gpu_memory_gb")
    def check_gpu(cls, v):
        if not 8 <= v <= 80:
            raise ValueError("gpu_memory_gb must be 8-80")
        return v

    @validator("batch_size")
    def check_batch(cls, v):
        if not 8 <= v <= 256:
            raise ValueError("batch_size must be 8-256")
        return v

    @validator("model_params_millions")
    def check_params(cls, v):
        if not 10 <= v <= 7000:
            raise ValueError("model_params_millions must be 10-7000")
        return v

    @validator("queue_depth")
    def check_queue(cls, v):
        if not 1 <= v <= 20:
            raise ValueError("queue_depth must be 1-20")
        return v

@app.get("/status")
def status():
    return {"status": "running", "model": "GradientBoosting", "version": "1.0"}

@app.post("/estimate")
def estimate(job: JobRequest):
    data = pd.DataFrame([job.dict()])
    pred = model.predict(data)[0]
    return {"prediction": round(float(pred), 4)}