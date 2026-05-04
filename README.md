# MLOps CIE Submission

**USN:** 1BM23AI009
**Course:** MLOps (24AM6AEMLO)
**Date:** 04 May 2026

## Scenario
MLOps Engineer at CoinPulse — a blockchain analytics platform. The goal is to predict transaction gas fees (gas_fee_gwei) to help users optimize transaction timing.

## Dataset
### Features
| Feature | Description | Range |
|---------|-------------|-------|
| network_congestion | Network congestion level | 0.1 – 1.0 |
| txn_size_bytes | Transaction size in bytes | 200 – 2000 |
| is_priority | Priority transaction flag | 0 or 1 |
| block_fullness_pct | Block fullness percentage | 50 – 100 |

- **Target:** gas_fee_gwei (gwei)
- **Training data:** 25 rows
- **New data:** 20 rows (shifted distributions)

## Tasks Completed

### Task 1 — Experiment Tracking & Model Comparison 
- Trained Lasso and GradientBoosting models
- Logged MAE and RMSE to MLflow
- Experiment name: coinpulse-gas-fee-gwei
- Tag: experiment_type = baseline_comparison

| Model | MAE | RMSE |
|-------|-----|------|
| Lasso | 12.7012 | 15.4547 |
| GradientBoosting | 16.8365 | 22.1934 |

- **Best model: Lasso** (lowest RMSE: 15.4547)

### Task 2 — Docker Packaging 
- Created src/predict_cli.py using argparse
- Containerized using Docker with python:3.12-slim base image
- Image: coinpulse-predictor:v1
- Test prediction for input (network_congestion=0.6, txn_size_bytes=1078, is_priority=1, block_fullness_pct=75.9): **95.1087 gwei**

### Task 3 — FastAPI Serving 
- Served best model via FastAPI on port 9000
- POST /forecast — accepts features, returns prediction
- GET /health — returns operational status
- Pydantic validation with HTTP 422 for invalid inputs
- Test prediction: **95.1087 gwei**

### Task 4 — Retraining Pipeline 
| Metric | Value |
|--------|-------|
| Original data rows | 25 |
| New data rows | 20 |
| Combined data rows | 45 |
| Champion MAE | 16.8365 |
| Retrained MAE | 8.9402 |
| Improvement | 7.8963 |
| Threshold | 0.5 |
| Action | promoted |

- Retrained Lasso on combined data
- MAE improved by 7.8963 (threshold: 0.5) — **model promoted!**

## Tech Stack
- Python, MLflow, FastAPI, Uvicorn, Docker, Scikit-learn, Pandas, NumPy
