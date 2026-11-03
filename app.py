
import joblib
import json
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("breast_cancer_model.joblib")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

@app.route("/")
def home():
    return """
    <h1>Breast Cancer Classification</h1>
    <h3>Machine Learning Prediction Application</h3>
    <p>Our Logistic Regression model is ready.</p>
    <p>Use the /predict endpoint to submit predictions.</p>
    """

@app.route("/health")
def health():
    return jsonify({"status": "running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "JSON input required"}), 400

        missing = [f for f in feature_names if f not in input_data]

        if missing:
            return jsonify({"error": "Missing features", "features": missing}), 400

        values = np.array([[float(input_data[f]) for f in feature_names]])

        prediction = int(model.predict(values)[0])
        probabilities = model.predict_proba(values)[0]

        return jsonify({
            "prediction": "malignant" if prediction == 0 else "benign",
            "confidence": round(float(max(probabilities)) * 100, 2)
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Input values must be numeric"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
