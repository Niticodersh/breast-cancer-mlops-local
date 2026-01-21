# model/train.py
# Reproducible training script for Breast Cancer classification model
# Generates all artifacts required for deployment

import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names.tolist()

print("Dataset loaded:")
print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")

# Train-test split (same random state as notebook for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train final SVM model
model = SVC(probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

print("Model training completed (SVM with probability=True)")

# Define paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(MODEL_DIR, "model.joblib")
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
features_path = os.path.join(MODEL_DIR, "feature_names.joblib")

# Save artifacts
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(feature_names, features_path)

print("Deployment artifacts saved successfully:")
print(f"   → {model_path}")
print(f"   → {scaler_path}")
print(f"   → {features_path}")
print("\nModel is now ready for inference service.")