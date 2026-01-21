# api/flask_app/app.py
# Flask web app with beautiful UI for Breast Cancer prediction

import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="../../templates")

# Load artifacts
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model")
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

print("Flask UI app: Model loaded successfully")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Collect 30 features
            features = []
            for i in range(30):
                val = float(request.form[f"feature{i}"])
                features.append(val)
            
            # Predict
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            prediction = int(model.predict(features_scaled)[0])
            probability = float(model.predict_proba(features_scaled)[0, 1])

            result = {
                "prediction": prediction,
                "diagnosis": "Malignant" if prediction == 0 else "Benign",
                "probability_benign": probability
            }
        except Exception as e:
            result = {"diagnosis": f"Error: {str(e)}"}

    return render_template("index.html", feature_names=feature_names, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)