# api/fastapi_app/main.py
# FastAPI inference service for Breast Cancer SVM model

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List
from prometheus_client import Counter, Histogram, start_http_server

# ----------------------------- Metrics Setup -----------------------------
# Prometheus metrics (will be exposed at /metrics)
REQUEST_COUNT = Counter(
    "app_requests_total", "Total request count", ["endpoint", "method"]
)
PREDICTION_LATENCY = Histogram(
    "app_prediction_latency_seconds", "Prediction latency in seconds"
)
PREDICTION_COUNTER = Counter(
    "app_predictions_total", "Total predictions", ["prediction"]
)

# Start Prometheus metrics server on port 8001 (background thread)
start_http_server(8001)

# ----------------------------- FastAPI App -----------------------------
app = FastAPI(
    title="Breast Cancer Classification API",
    description="SVM model for predicting malignant (0) or benign (1) tumors",
    version="1.0.0"
)

# ----------------------------- Input Schema -----------------------------
class PredictionRequest(BaseModel):
    features: List[float]

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_features

    @classmethod
    def validate_features(cls, v):
        if len(v) != 30:
            raise ValueError("Exactly 30 features are required")
        return v

# ----------------------------- Load Artifacts at Startup -----------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    print("Model, scaler, and feature names loaded successfully")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    raise

# ----------------------------- Endpoints -----------------------------
@app.get("/health")
async def health_check():
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return {"status": "healthy", "model": "SVM loaded"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()

    try:
        # Convert input to numpy array and scale
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Time the prediction
        with PREDICTION_LATENCY.time():
            prediction = int(model.predict(features_scaled)[0])
            probability = float(model.predict_proba(features_scaled)[0, 1])

        # Update metrics
        PREDICTION_COUNTER.labels(prediction="malignant" if prediction == 0 else "benign").inc()

        return {
            "prediction": prediction,
            "diagnosis": "malignant" if prediction == 0 else "benign",
            "probability_benign": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")