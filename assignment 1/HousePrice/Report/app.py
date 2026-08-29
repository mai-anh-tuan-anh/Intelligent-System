import joblib
import pandas as pd

from flask import Flask, request, jsonify


app = Flask(__name__)


# ============================================================
# LOAD MODEL ĐÃ TRAIN
# ============================================================

filename = "house_price_model.pkl"

loaded_model = joblib.load(filename)

print("Model loaded successfully!")


# ============================================================
# PREDICT API
# ============================================================

@app.route(
    "/house/v1/predict",
    methods=["POST"]
)
def predict():

    try:

        features = request.get_json()

        print("\n========== REQUEST ==========")
        print(features)


        input_data = pd.DataFrame([features])

        print("\n========== INPUT DATA ==========")
        print(input_data)

        print("\n========== COLUMNS ==========")
        print(input_data.columns.tolist())


        prediction = loaded_model.predict(
            input_data
        )

        result = float(prediction[0])

        print("\n========== PREDICTION ==========")
        print(result)


        return jsonify({
            "prediction": result
        })


    except Exception as e:

        import traceback

        print("\n========== ERROR ==========")

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500
    
@app.route("/house/v1/test", methods=["GET"])
def test_connection():
    return {
        "status": "ok",
        "message": "Android can reach Flask"
    }

@app.route("/house/v1/options", methods=["GET"])
def get_options():

    try:
        preprocessor = loaded_model.named_steps["preprocessor"]

        categorical_transformer = None
        categorical_columns = None

        for name, transformer, columns in preprocessor.transformers_:

            if hasattr(transformer, "named_steps"):

                if "encoder" in transformer.named_steps:

                    categorical_transformer = transformer
                    categorical_columns = list(columns)

                    break


        if categorical_transformer is None:

            return jsonify({
                "error": "Không tìm thấy categorical transformer"
            }), 500


        encoder = (
            categorical_transformer
            .named_steps["encoder"]
        )


        all_categories = {}

        for column, categories in zip(
            categorical_columns,
            encoder.categories_
        ):

            all_categories[str(column)] = [
                str(value)
                for value in categories
            ]


        fields = [
            "House direction",
            "Balcony direction",
            "Legal status",
            "Furniture state"
        ]


        result = {}

        for field in fields:

            result[field] = all_categories.get(
                field,
                []
            )


        return jsonify(result)


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500
# ============================================================
# RUN SERVER
# ============================================================
import os

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )