from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Load model and scaler ---
with open("fraud_rf_smote.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return "Flask Fraud Detection API is running! Use the /predict endpoint."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]

        # Convert to DataFrame
        input_df = pd.DataFrame(
            [features],
            columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
        )

        # Scale
        X_scaled = scaler.transform(input_df)

        # Predict
        fraud_pred = model.predict(X_scaled)[0]
        fraud_prob = model.predict_proba(X_scaled)[0][1]

        return jsonify({
            "fraud_prediction": int(fraud_pred),
            "fraud_probability": float(fraud_prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
