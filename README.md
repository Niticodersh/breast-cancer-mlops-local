# Breast Cancer Classification MLOps Project

A complete end-to-end MLOps pipeline for breast cancer classification using machine learning. This repository demonstrates production-ready deployment of an SVM model with containerization, Kubernetes orchestration, monitoring, and multiple API interfaces.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Deployment Options](#deployment-options)
- [Monitoring](#monitoring)
- [API Documentation](#api-documentation)

---

## 🎯 Project Overview

This project implements a machine learning model for breast cancer classification (malignant vs benign) using the Wisconsin Breast Cancer Dataset. The model uses a Support Vector Machine (SVM) classifier and is deployed with:

- **FastAPI REST API** with Prometheus metrics
- **Flask Web Application** with interactive UI
- **Docker containerization**
- **Kubernetes deployment** with scaling and canary deployment support
- **Monitoring stack** (Prometheus + Grafana)

---

## 📁 Repository Structure

```
breast-cancer-mlops-local/
│
├── model/                          # Model training and artifacts
│   ├── train.py                   # Training script (SVM model)
│   ├── model.joblib               # Trained model artifact
│   ├── scaler.joblib              # Feature scaler artifact
│   └── feature_names.joblib       # Feature names artifact
│
├── api/                           # API implementations
│   ├── fastapi_app/               # FastAPI REST API
│   │   ├── main.py                # FastAPI application with metrics
│   │   └── requirements.txt       # FastAPI dependencies
│   └── flask_app/                 # Flask web application
│       └── app.py                 # Flask UI application
│
├── templates/                     # HTML templates
│   └── index.html                 # Flask web UI template
│
├── docker/                        # Containerization
│   └── Dockerfile                 # Docker image for FastAPI service
│
├── k8s/                           # Kubernetes configurations
│   ├── deployment.yaml            # Main Kubernetes deployment
│   ├── service.yaml               # Kubernetes service (NodePort)
│   └── canary-deployment.yaml     # Canary deployment strategy
│
├── monitoring/                    # Monitoring infrastructure
│   ├── docker-compose.yml         # Prometheus + Grafana stack
│   ├── prometheus.yml             # Prometheus configuration
│   └── grafana-dashboards/        # Grafana dashboard definitions
│
├── diagrams/                      # Architecture diagrams
│   └── architecture.drawio        # System architecture diagram
│
├── notebook.ipynb                 # Jupyter notebook for exploration
├── requirements.txt              # Main project dependencies
├── requirements-dev.txt           # Development dependencies
└── README.md                      # This file
```

---

## ✨ Features

### Model
- **Algorithm**: Support Vector Machine (SVM) with probability estimation
- **Dataset**: Wisconsin Breast Cancer Dataset (30 features)
- **Preprocessing**: StandardScaler for feature normalization
- **Reproducibility**: Fixed random seeds for consistent results

### APIs
- **FastAPI REST API**: Production-ready API with:
  - Health check endpoint
  - Prediction endpoint with validation
  - Prometheus metrics integration
  - Automatic API documentation (Swagger/OpenAPI)
  
- **Flask Web UI**: User-friendly web interface with:
  - Interactive form for feature input
  - Real-time predictions
  - Visual result display

### Infrastructure
- **Docker**: Containerized FastAPI service
- **Kubernetes**: 
  - Deployment with 3 replicas
  - NodePort service for external access
  - Canary deployment support
  - Health checks and readiness probes

### Monitoring
- **Prometheus**: Metrics collection (request count, latency, predictions)
- **Grafana**: Visualization dashboards
- **Custom Metrics**: 
  - Request count by endpoint
  - Prediction latency
  - Prediction distribution (malignant/benign)

---

## 🔧 Prerequisites

Before running this project, ensure you have:

- **Python 3.9+**
- **Docker** (for containerization)
- **Docker Compose** (for monitoring stack)
- **Kubernetes cluster** (minikube, kind, or cloud K8s) - optional
- **pip** package manager

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd breast-cancer-mlops-local
```

### 2. Install Python Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies (optional, for Jupyter notebook)
pip install -r requirements-dev.txt
```

### 3. Train the Model

First, train the model to generate the required artifacts:

```bash
python model/train.py
```

This will create:
- `model/model.joblib` - Trained SVM model
- `model/scaler.joblib` - Feature scaler
- `model/feature_names.joblib` - Feature names list

**Expected Output:**
```
Dataset loaded:
   Samples: 569, Features: 30
Model training completed (SVM with probability=True)
Deployment artifacts saved successfully:
   → model/model.joblib
   → model/scaler.joblib
   → model/feature_names.joblib
```

---

## 📖 Usage Guide

### Option 1: Run FastAPI REST API (Local)

Start the FastAPI service locally:

```bash
cd api/fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Access Points:**
- **API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc
- **Metrics**: http://localhost:8001/metrics
- **Health Check**: http://localhost:8000/health

**Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction (example with 30 features)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

### Option 2: Run Flask Web UI (Local)

Start the Flask web application:

```bash
cd api/flask_app
python app.py
```

**Access Point:**
- **Web UI**: http://localhost:5000

The web interface provides a form where you can input all 30 features and get predictions with visual feedback.

### Option 3: Run with Docker

Build and run the Docker container:

```bash
# Build the Docker image
docker build -t breast-cancer-fastapi:latest -f docker/Dockerfile .

# Run the container
docker run -p 8000:8000 -p 8001:8001 breast-cancer-fastapi:latest
```

**Access Points:**
- **API**: http://localhost:8000
- **Metrics**: http://localhost:8001/metrics

### Option 4: Deploy to Kubernetes

#### Prerequisites
- Kubernetes cluster running (minikube, kind, or cloud)
- kubectl configured

#### Steps

1. **Build and load Docker image** (for local K8s):

```bash
# Build image
docker build -t breast-cancer-fastapi:latest -f docker/Dockerfile .

# Load into minikube (if using minikube)
minikube image load breast-cancer-fastapi:latest
```

2. **Deploy to Kubernetes**:

```bash
# Deploy the application
kubectl apply -f k8s/deployment.yaml

# Create the service
kubectl apply -f k8s/service.yaml

# Check deployment status
kubectl get pods
kubectl get services
```

3. **Access the service**:

```bash
# For minikube
minikube service breast-cancer-service

# Or access via NodePort (port 30000)
# http://<node-ip>:30000
```

4. **Canary Deployment** (optional):

```bash
kubectl apply -f k8s/canary-deployment.yaml
```

### Option 5: Set Up Monitoring Stack

Start Prometheus and Grafana:

```bash
cd monitoring
docker-compose up -d
```

**Access Points:**
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001
  - Default credentials: `admin` / `admin` (change on first login)

**Configure Prometheus** to scrape metrics from your API by updating `monitoring/prometheus.yml` with the correct target address.

---

## 🔍 API Documentation

### FastAPI Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "SVM loaded"
}
```

#### Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [17.99, 10.38, 122.8, ...]  // Array of 30 float values
}
```

**Response:**
```json
{
  "prediction": 1,
  "diagnosis": "benign",
  "probability_benign": 0.9876
}
```

**Note:** 
- `prediction`: 0 = malignant, 1 = benign
- `probability_benign`: Probability of benign diagnosis (0-1)

### Metrics Endpoint

```http
GET /metrics
```

Returns Prometheus metrics including:
- `app_requests_total` - Total request count by endpoint
- `app_prediction_latency_seconds` - Prediction latency histogram
- `app_predictions_total` - Total predictions by class

---

## 📊 Monitoring

### Prometheus Metrics

The FastAPI service exposes the following metrics:

- **Request Count**: `app_requests_total{endpoint, method}`
- **Prediction Latency**: `app_prediction_latency_seconds`
- **Prediction Counter**: `app_predictions_total{prediction}`

### Grafana Dashboards

Import dashboards from `monitoring/grafana-dashboards/` to visualize:
- Request rates
- Prediction latency
- Prediction distribution
- Error rates

---

## 🛠️ Development

### Running Tests

```bash
# Test API endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [...]}'
```

### Model Retraining

To retrain the model with different parameters:

```bash
python model/train.py
```

The script uses fixed random seeds for reproducibility. Modify `train.py` to change model parameters.

### Jupyter Notebook

Explore the dataset and model:

```bash
jupyter notebook notebook.ipynb
```

---

## 📝 Notes

- The model expects exactly **30 features** in the correct order
- Features should be raw values (not pre-scaled) - the API handles scaling
- Model artifacts must be present in `model/` directory before running APIs
- Prometheus metrics are exposed on port **8001** (separate from API port 8000)
- Kubernetes service uses **NodePort** type for external access (ports 30000 and 30001)

---

## 🔐 Security Considerations

- Docker image runs as non-root user
- Input validation on all API endpoints
- Health checks for Kubernetes deployments
- Consider adding authentication for production deployments

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🐛 Troubleshooting

### Model artifacts not found
- Ensure you've run `python model/train.py` first
- Check that `model/` directory contains all `.joblib` files

### Port already in use
- Change ports in the application code or use different ports
- For Docker: `docker run -p <host-port>:<container-port>`

### Kubernetes pods not starting
- Check pod logs: `kubectl logs <pod-name>`
- Verify image is loaded: `kubectl describe pod <pod-name>`
- Ensure model artifacts are included in Docker image

### Prometheus not scraping metrics
- Verify API is running and accessible
- Check `monitoring/prometheus.yml` target configuration
- Ensure firewall allows connections to metrics port

---

**Happy Deploying! 🚀**
