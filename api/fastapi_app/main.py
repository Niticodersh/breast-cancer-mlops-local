# api/fastapi_app/main.py
# FastAPI inference service for Breast Cancer SVM model

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List
from prometheus_client import Counter, Histogram, start_http_server


VERSION = os.getenv("VERSION", "unknown")

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
# Robust path: works both locally and in Docker container
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  # Local project root
MODEL_DIR = os.path.join(PROJECT_ROOT, "model") if os.path.exists(os.path.join(PROJECT_ROOT, "model")) else os.path.join(CURRENT_DIR, "model")

try:
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_names.joblib")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    print("Model artifacts loaded successfully from:")
    print(f"   {MODEL_DIR}")

except Exception as e:
    print(f"Error loading artifacts from {MODEL_DIR}: {e}")
    raise

# ----------------------------- Endpoints -----------------------------
# @app.get("/health")
# async def health_check():
#     REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
#     return {"status": "healthy", "model": "SVM loaded"}


@app.get("/health")
async def health_check():
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return {
        "status": "healthy",
        "model": "SVM loaded",
        "version": VERSION
    }


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