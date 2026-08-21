import joblib
import pandas as pd

from flask import Flask, request, jsonify, render_template


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "bank_marketing_model.pkl"

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received."
            }), 400

        # Convert JSON into one-row DataFrame
        sample = pd.DataFrame([data])

        # Model prediction
        prediction = model.predict(sample)[0]

        # Result
        if prediction == 1:
            label = "Likely to subscribe"
        else:
            label = "Likely not to subscribe"

        return jsonify({
            "prediction": int(prediction),
            "label": label
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 400


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )