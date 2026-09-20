
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os

import joblib
import pandas as pd

from flask import (
    Flask,
    jsonify,
    request,
    render_template
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Absolute path of the folder containing app.py
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# Saved machine-learning pipeline
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "diabetes_model_pipeline.joblib"
)


# Web template directory
TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "web",
    "templates"
)


# Web static directory
STATIC_DIR = os.path.join(
    BASE_DIR,
    "web",
    "static"
)


# ============================================================
# 3. CREATE FLASK APPLICATION
# ============================================================

# Flask normally searches for:
#
#   templates/
#   static/
#
# in the same directory as app.py.
#
# Our project intentionally keeps Web files inside:
#
#   web/templates/
#   web/static/
#
# Therefore the directories are explicitly configured here.

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)


# ============================================================
# 4. CHECK PROJECT FILES AND DIRECTORIES
# ============================================================

if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        "\nModel file not found.\n"
        f"Expected location:\n{MODEL_PATH}\n"
    )


if not os.path.isdir(TEMPLATE_DIR):

    raise FileNotFoundError(
        "\nWeb template directory not found.\n"
        f"Expected location:\n{TEMPLATE_DIR}\n"
    )


if not os.path.isdir(STATIC_DIR):

    raise FileNotFoundError(
        "\nWeb static directory not found.\n"
        f"Expected location:\n{STATIC_DIR}\n"
    )


# ============================================================
# 5. LOAD THE SAVED ML PIPELINE
# ============================================================

# The saved object should contain:
#
#   preprocessing
#       +
#   feature transformation
#       +
#   trained classifier
#
# Loading the complete pipeline ensures inference uses
# exactly the same transformation process as training.

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# 6. PRINT MODEL INFORMATION
# ============================================================

print("=" * 70)

print(
    "Diabetes Prediction API"
)

print(
    "=" * 70
)

print(
    "Model loaded successfully."
)

print(
    f"Model path    : {MODEL_PATH}"
)

print(
    f"Template path : {TEMPLATE_DIR}"
)

print(
    f"Static path   : {STATIC_DIR}"
)


# ============================================================
# 7. DETERMINE MODEL NAME
# ============================================================

def get_model_name():
    """
    Return the name of the final classifier.

    The saved object is expected to be a sklearn Pipeline
    containing a step named 'model'.
    """

    # Check whether the saved object is a Pipeline
    if hasattr(
        model,
        "named_steps"
    ):

        # Check whether the final estimator is stored
        # under the expected step name.
        if "model" in model.named_steps:

            return type(
                model.named_steps["model"]
            ).__name__


    # Fallback for a model saved without Pipeline
    return type(model).__name__


MODEL_NAME = get_model_name()

print(
    f"Model name    : {MODEL_NAME}"
)


# ============================================================
# 8. EXTRACT INPUT FEATURE SCHEMA
# ============================================================

def get_feature_schema():
    """
    Extract the exact input features from the fitted
    preprocessing pipeline.

    This avoids manually maintaining a second feature list
    in the API.

    Each feature is represented as:

        {
            "name": ...,
            "type": "number" or "category",
            "categories": [...]
        }
    """

    # --------------------------------------------------------
    # Ensure the saved object is a Pipeline
    # --------------------------------------------------------

    if not hasattr(
        model,
        "named_steps"
    ):

        raise RuntimeError(
            "The saved model is not a sklearn Pipeline."
        )


    # --------------------------------------------------------
    # Find the preprocessing step
    # --------------------------------------------------------

    if "preprocessor" not in model.named_steps:

        raise RuntimeError(
            "The saved pipeline does not contain "
            "a 'preprocessor' step."
        )


    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )


    # --------------------------------------------------------
    # Ensure preprocessing has been fitted
    # --------------------------------------------------------

    if not hasattr(
        preprocessor,
        "transformers_"
    ):

        raise RuntimeError(
            "The preprocessing pipeline has not been fitted."
        )


    schema = []


    # --------------------------------------------------------
    # Inspect every fitted transformer
    # --------------------------------------------------------

    for (
        transformer_name,
        transformer,
        columns
    ) in preprocessor.transformers_:


        # ====================================================
        # Ignore dropped columns
        # ====================================================

        if transformer == "drop":

            continue


        # ====================================================
        # NUMERICAL FEATURES
        # ====================================================

        if transformer_name == "numeric":

            for column in columns:

                schema.append({

                    "name":
                        str(column),

                    "type":
                        "number",

                    "categories":
                        []

                })


        # ====================================================
        # CATEGORICAL FEATURES
        # ====================================================

        elif transformer_name == "categorical":

            try:

                # Get the fitted OneHotEncoder
                onehot = (
                    transformer
                    .named_steps[
                        "onehot"
                    ]
                )


                # categories_ contains the actual categories
                # learned during training.
                categories = (
                    onehot.categories_
                )


                for (
                    column,
                    values
                ) in zip(
                    columns,
                    categories
                ):

                    schema.append({

                        "name":
                            str(column),

                        "type":
                            "category",

                        "categories": [
                            str(value)
                            for value in values
                        ]

                    })


            except Exception:

                # Fallback:
                # if the categories cannot be extracted,
                # still expose the feature as categorical.

                for column in columns:

                    schema.append({

                        "name":
                            str(column),

                        "type":
                            "category",

                        "categories":
                            []

                    })


    return schema


# Extract schema once when Flask starts.
FEATURE_SCHEMA = get_feature_schema()


# Extract only feature names for convenient DataFrame creation.
FEATURE_COLUMNS = [
    feature["name"]
    for feature in FEATURE_SCHEMA
]


print(
    f"Number of input features: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    "\nInput feature schema:"
)

for feature in FEATURE_SCHEMA:

    print(
        f"  - {feature['name']} "
        f"[{feature['type']}]"
    )


print(
    "=" * 70
)


# ============================================================
# 9. CONVERT INPUT VALUES
# ============================================================

def convert_input_value(
    value,
    feature_type
):
    """
    Convert incoming JSON values to the appropriate
    Python representation.
    """

    # --------------------------------------------------------
    # Numerical values
    # --------------------------------------------------------

    if feature_type == "number":

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                f"Invalid numerical value: {value}"
            )


    # --------------------------------------------------------
    # Categorical values
    # --------------------------------------------------------

    return str(
        value
    )


# ============================================================
# 10. VALIDATE CLIENT INPUT
# ============================================================

def validate_request(
    data
):
    """
    Validate a JSON request and return a dictionary
    containing exactly the features expected by the model.
    """

    # --------------------------------------------------------
    # Request must be a JSON object
    # --------------------------------------------------------

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            "Request body must be a JSON object."
        )


    cleaned_data = {}


    # --------------------------------------------------------
    # Validate every expected feature
    # --------------------------------------------------------

    for feature in FEATURE_SCHEMA:

        feature_name = (
            feature["name"]
        )

        feature_type = (
            feature["type"]
        )


        # ----------------------------------------------------
        # Required feature check
        # ----------------------------------------------------

        if feature_name not in data:

            raise ValueError(
                f"Missing required feature: "
                f"{feature_name}"
            )


        value = data[
            feature_name
        ]


        # ----------------------------------------------------
        # Null check
        # ----------------------------------------------------

        if value is None:

            raise ValueError(
                f"Feature '{feature_name}' "
                f"cannot be null."
            )


        # ----------------------------------------------------
        # Empty string check
        # ----------------------------------------------------

        if (
            isinstance(
                value,
                str
            )
            and
            value.strip() == ""
        ):

            raise ValueError(
                f"Feature '{feature_name}' "
                f"cannot be empty."
            )


        # ----------------------------------------------------
        # Type conversion
        # ----------------------------------------------------

        cleaned_data[
            feature_name
        ] = convert_input_value(
            value,
            feature_type
        )


    return cleaned_data


# ============================================================
# 11. GENERATE MODEL PREDICTION
# ============================================================

def generate_prediction(
    features
):
    """
    Generate a prediction for one patient.

    Parameters
    ----------
    features : dict
        Validated patient features.

    Returns
    -------
    dict
        JSON-compatible prediction result.
    """

    # --------------------------------------------------------
    # Create one-row DataFrame
    # --------------------------------------------------------
    #
    # The column order is explicitly controlled by
    # FEATURE_COLUMNS to match training.

    input_df = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )


    # ========================================================
    # CLASS PREDICTION
    # ========================================================

    prediction = model.predict(
        input_df
    )[0]

    prediction = int(
        prediction
    )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    probability_diabetes = None

    probability_no_diabetes = None

    confidence = None


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model
            .predict_proba(
                input_df
            )[0]
        )


        # -----------------------------------------------
        # Get the class labels used by the classifier
        # -----------------------------------------------

        classes = list(
            model.classes_
        )


        # -----------------------------------------------
        # Find the positive class
        # -----------------------------------------------

        if 1 in classes:

            diabetes_index = (
                classes.index(1)
            )


            probability_diabetes = float(
                probabilities[
                    diabetes_index
                ]
            )


            probability_no_diabetes = (
                1.0 -
                probability_diabetes
            )


            # -------------------------------------------
            # Confidence
            #
            # Confidence is the probability associated
            # with the predicted class.
            # -------------------------------------------

            predicted_class_index = (
                classes.index(
                    prediction
                )
            )


            confidence = float(
                probabilities[
                    predicted_class_index
                ]
            )


    # ========================================================
    # HUMAN-READABLE DIAGNOSIS
    # ========================================================

    if prediction == 1:

        diagnosis = (
            "Diabetes Positive"
        )

        risk_label = (
            "Positive"
        )

    else:

        diagnosis = (
            "Diabetes Negative"
        )

        risk_label = (
            "Negative"
        )


    # ========================================================
    # BUILD API RESPONSE
    # ========================================================

    result = {

        # Numerical prediction
        "prediction":
            prediction,


        # Human-readable diagnosis
        "diagnosis":
            diagnosis,


        # Simplified label
        "risk_label":
            risk_label,


        # Probability assigned to predicted class
        "confidence":
            (
                round(
                    confidence * 100,
                    2
                )
                if confidence is not None
                else None
            ),


        # Probability of diabetes = 1
        "probability_diabetes":
            (
                round(
                    probability_diabetes * 100,
                    2
                )
                if probability_diabetes is not None
                else None
            ),


        # Probability of diabetes = 0
        "probability_no_diabetes":
            (
                round(
                    probability_no_diabetes * 100,
                    2
                )
                if probability_no_diabetes is not None
                else None
            ),


        # Name of the final classifier
        "model":
            MODEL_NAME

    }


    return result


# ============================================================
# 12. HEALTH CHECK ENDPOINT
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    """
    Simple endpoint for checking whether the Flask
    service is running correctly.
    """

    return jsonify({

        "status":
            "ok",

        "service":
            "Diabetes Prediction API",

        "model":
            MODEL_NAME,

        "feature_count":
            len(FEATURE_COLUMNS)

    }), 200


# ============================================================
# 13. FEATURE SCHEMA ENDPOINT
# ============================================================

@app.route(
    "/diabetes/v1/schema",
    methods=["GET"]
)
def schema():

    """
    Return the exact features expected by the model.

    Web and Android clients can use this endpoint to
    construct their input forms.
    """

    return jsonify({

        "target":
            "diagnosed_diabetes",

        "model":
            MODEL_NAME,

        "features":
            FEATURE_SCHEMA

    }), 200


# ============================================================
# 14. DIABETES PREDICTION ENDPOINT
# ============================================================

@app.route(
    "/diabetes/v1/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Read JSON data
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if data is None:

            return jsonify({

                "error":
                    "Request body must contain valid JSON."

            }), 400


        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        features = validate_request(
            data
        )


        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        result = generate_prediction(
            features
        )


        # ----------------------------------------------------
        # Return successful response
        # ----------------------------------------------------

        return jsonify(
            result
        ), 200


    except ValueError as error:

        # Input-related errors
        return jsonify({

            "error":
                str(error)

        }), 400


    except Exception as error:

        # Unexpected server-side errors
        print(
            "Prediction error:",
            error
        )


        return jsonify({

            "error":
                "Internal prediction server error."

        }), 500


# ============================================================
# 15. WEB HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    """
    Serve the Web frontend.

    Because template_folder was explicitly configured above,
    Flask searches for:

        web/templates/index.html
    """

    return render_template(
        "index.html"
    )


# ============================================================
# 16. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Run Flask development server.
    #
    # host="0.0.0.0" allows:
    #   - local browser access
    #   - Android emulator access
    #   - devices on the same LAN (if firewall/network
    #     configuration allows it)
    # --------------------------------------------------------

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )