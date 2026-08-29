# ============================================================
# London House Price Prediction Service
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


MODEL_PATH = (
    BASE_DIR
    / "model"
    / "house_price_model_pipeline.joblib"
)


# ============================================================
# Flask application
# ============================================================

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)


# ============================================================
# Load trained pipeline
#
# No training is performed by the web service.
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "Saved model pipeline was not found:\n"
        f"{MODEL_PATH}"
    )


pipeline = joblib.load(
    MODEL_PATH
)


if not hasattr(
    pipeline,
    "named_steps"
):

    raise ValueError(
        "Saved object is not a valid sklearn Pipeline."
    )


if "preprocessor" not in pipeline.named_steps:

    raise ValueError(
        "Saved pipeline does not contain a preprocessor."
    )


if "model" not in pipeline.named_steps:

    raise ValueError(
        "Saved pipeline does not contain a model."
    )


preprocessor = (
    pipeline.named_steps[
        "preprocessor"
    ]
)


model = (
    pipeline.named_steps[
        "model"
    ]
)


# ============================================================
# Fixed feature definition used by the current notebook
# ============================================================

USER_NUMERIC_FEATURES = [

    "latitude",

    "longitude",

    "bathrooms",

    "bedrooms",

    "floorAreaSqM",

    "livingRooms"

]


USER_CATEGORICAL_FEATURES = [

    "outcode",

    "tenure",

    "propertyType",

    "currentEnergyRating"

]


USER_FEATURES = (
    USER_NUMERIC_FEATURES
    +
    USER_CATEGORICAL_FEATURES
)


# ============================================================
# Hidden model inputs
#
# These fields were part of the current trained model but are
# intentionally not requested from the user.
#
# Their values are passed as missing values so the fitted
# preprocessing pipeline can handle them using imputation.
# Country is fixed because the dataset contains England.
# ============================================================

FIXED_COUNTRY = "England"


HIDDEN_NUMERIC_FEATURES = [

    "history_percentageChange",

    "history_numericChange"

]


HIDDEN_DATE_FEATURES = [

    "history_date_year",

    "history_date_month"

]


# ============================================================
# Validation metrics from the selected model in the notebook
#
# Selected model:
# Gradient Boosting Tuned
#
# These values are validation-set performance indicators.
# They are not individual prediction probabilities.
# ============================================================

MODEL_NAME = (
    "Gradient Boosting Tuned"
)


MODEL_VALIDATION_METRICS = {

    "mae":
        170394.504560,

    "rmse":
        821122.4,

    "r2":
        0.623860,

    "training_time":
        880.2323

}


MODEL_CONFIDENCE_INDICATOR = (

    MODEL_VALIDATION_METRICS["r2"]
    * 100

)


# ============================================================
# Extract fitted preprocessor information
# ============================================================

def get_transformer_columns(
    transformer_name: str
) -> list[str]:

    for (
        name,
        transformer,
        columns
    ) in preprocessor.transformers_:

        if name == transformer_name:

            return list(
                columns
            )

    return []


FITTED_NUMERIC_COLUMNS = (
    get_transformer_columns(
        "numeric"
    )
)


FITTED_CATEGORICAL_COLUMNS = (
    get_transformer_columns(
        "categorical"
    )
)


# ============================================================
# Extract imputation statistics
# ============================================================

def get_numeric_defaults() -> dict[str, float]:

    defaults = {}


    numeric_transformer = (
        preprocessor
        .named_transformers_
        .get(
            "numeric"
        )
    )


    if numeric_transformer is None:

        return defaults


    imputer = (
        numeric_transformer
        .named_steps
        .get(
            "imputer"
        )
    )


    if imputer is None:

        return defaults


    for column, value in zip(

        FITTED_NUMERIC_COLUMNS,

        imputer.statistics_

    ):

        if not pd.isna(value):

            defaults[column] = float(
                value
            )


    return defaults


NUMERIC_DEFAULTS = (
    get_numeric_defaults()
)


# ============================================================
# Extract categorical values learned by OneHotEncoder
# ============================================================

def get_categories() -> dict[str, list[str]]:

    result = {}


    categorical_transformer = (
        preprocessor
        .named_transformers_
        .get(
            "categorical"
        )
    )


    if categorical_transformer is None:

        return result


    encoder = (
        categorical_transformer
        .named_steps
        .get(
            "onehot"
        )
    )


    if encoder is None:

        return result


    for index, column in enumerate(

        FITTED_CATEGORICAL_COLUMNS

    ):

        if (
            index
            >=
            len(
                encoder.categories_
            )
        ):

            continue


        result[column] = [

            str(value)

            for value
            in encoder.categories_[index]

            if not pd.isna(value)

        ]


    return result


CATEGORIES = (
    get_categories()
)


# ============================================================
# Build default sample values
# ============================================================

def first_category(
    column: str,
    fallback: str = ""
) -> str:

    values = (
        CATEGORIES
        .get(
            column,
            []
        )
    )


    if values:

        return values[0]


    return fallback


DEFAULTS = {

    "latitude":
        round(
            NUMERIC_DEFAULTS.get(
                "latitude",
                51.50
            ),
            6
        ),

    "longitude":
        round(
            NUMERIC_DEFAULTS.get(
                "longitude",
                -0.12
            ),
            6
        ),

    "bathrooms":
        round(
            NUMERIC_DEFAULTS.get(
                "bathrooms",
                1
            ),
            1
        ),

    "bedrooms":
        round(
            NUMERIC_DEFAULTS.get(
                "bedrooms",
                2
            ),
            1
        ),

    "floorAreaSqM":
        round(
            NUMERIC_DEFAULTS.get(
                "floorAreaSqM",
                70
            ),
            2
        ),

    "livingRooms":
        round(
            NUMERIC_DEFAULTS.get(
                "livingRooms",
                1
            ),
            1
        ),

    "outcode":
        first_category(
            "outcode",
            ""
        ),

    "tenure":
        first_category(
            "tenure",
            ""
        ),

    "propertyType":
        first_category(
            "propertyType",
            ""
        ),

    "currentEnergyRating":
        first_category(
            "currentEnergyRating",
            ""
        )

}


# ============================================================
# Feature engineering
#
# This reproduces the relevant transformation used by the
# current notebook.
# ============================================================

def engineer_features(
    raw_df: pd.DataFrame
) -> pd.DataFrame:

    data = (
        raw_df
        .copy()
    )


    # --------------------------------------------------------
    # Historical variables not supplied by the user
    # --------------------------------------------------------

    data[
        "history_percentageChange"
    ] = np.nan


    data[
        "history_numericChange"
    ] = np.nan


    # --------------------------------------------------------
    # Historical date is intentionally hidden from the user.
    # The corresponding engineered values remain missing and
    # are handled by the fitted numerical imputer.
    # --------------------------------------------------------

    data[
        "history_date_year"
    ] = np.nan


    data[
        "history_date_month"
    ] = np.nan


    # --------------------------------------------------------
    # Log-transformed floor area
    # --------------------------------------------------------

    if "floorAreaSqM" in data.columns:

        data[
            "log_floorAreaSqM"
        ] = np.log1p(

            data[
                "floorAreaSqM"
            ].clip(
                lower=0
            )

        )


    return data


# ============================================================
# Validate user input
# ============================================================

def validate_payload(
    payload: dict
) -> None:

    if not isinstance(
        payload,
        dict
    ):

        raise ValueError(
            "Request body must be a JSON object."
        )


    # --------------------------------------------------------
    # Numerical fields
    # --------------------------------------------------------

    for column in USER_NUMERIC_FEATURES:

        value = payload.get(
            column,
            ""
        )


        if value == "":

            continue


        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                f"{column} must be numeric."
            )


        if not np.isfinite(
            number
        ):

            raise ValueError(
                f"{column} must be a finite number."
            )


        if column == "latitude":

            if not (
                -90
                <=
                number
                <=
                90
            ):

                raise ValueError(
                    "Latitude must be between -90 and 90."
                )


        if column == "longitude":

            if not (
                -180
                <=
                number
                <=
                180
            ):

                raise ValueError(
                    "Longitude must be between -180 and 180."
                )


        if column in [

            "bathrooms",
            "bedrooms",
            "livingRooms"

        ]:

            if number < 0:

                raise ValueError(
                    f"{column} cannot be negative."
                )


        if column == "floorAreaSqM":

            if number < 0:

                raise ValueError(
                    "floorAreaSqM cannot be negative."
                )


# ============================================================
# Prepare prediction input
# ============================================================

def prepare_input(
    payload: dict
) -> pd.DataFrame:

    validate_payload(
        payload
    )


    data = {

        "latitude":
            np.nan,

        "longitude":
            np.nan,

        "bathrooms":
            np.nan,

        "bedrooms":
            np.nan,

        "floorAreaSqM":
            np.nan,

        "livingRooms":
            np.nan,

        "outcode":
            None,

        "tenure":
            None,

        "propertyType":
            None,

        "currentEnergyRating":
            None

    }


    # --------------------------------------------------------
    # User-provided numerical values
    # --------------------------------------------------------

    for column in USER_NUMERIC_FEATURES:

        value = payload.get(
            column,
            ""
        )


        if value != "":

            data[column] = float(
                value
            )


    # --------------------------------------------------------
    # User-provided categorical values
    # --------------------------------------------------------

    for column in USER_CATEGORICAL_FEATURES:

        value = payload.get(
            column,
            ""
        )


        if value != "":

            data[column] = str(
                value
            )


    # --------------------------------------------------------
    # Fixed country value
    # --------------------------------------------------------

    data[
        "country"
    ] = FIXED_COUNTRY


    # --------------------------------------------------------
    # Raw DataFrame
    # --------------------------------------------------------

    raw_df = pd.DataFrame(

        [data]

    )


    # --------------------------------------------------------
    # Reproduce notebook feature engineering
    # --------------------------------------------------------

    engineered_df = (
        engineer_features(
            raw_df
        )
    )


    return engineered_df


# ============================================================
# Web page
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(

        "index.html",

        defaults=DEFAULTS,

        categories=CATEGORIES,

        model_name=MODEL_NAME,

        confidence=
            MODEL_CONFIDENCE_INDICATOR

    )


# ============================================================
# Model information endpoint
# ============================================================

@app.route(
    "/house/v1/info",
    methods=["GET"]
)
def model_info():

    return jsonify({

        "model":
            MODEL_NAME,

        "target":
            "history_price",

        "confidence_indicator":
            round(
                MODEL_CONFIDENCE_INDICATOR,
                2
            ),

        "confidence_note":
            "Based on validation R2. "
            "It is not a probability of prediction correctness.",

        "validation_metrics": {

            "MAE":
                MODEL_VALIDATION_METRICS[
                    "mae"
                ],

            "RMSE":
                MODEL_VALIDATION_METRICS[
                    "rmse"
                ],

            "R2":
                MODEL_VALIDATION_METRICS[
                    "r2"
                ],

            "Training Time (s)":
                MODEL_VALIDATION_METRICS[
                    "training_time"
                ]

        }

    })


# ============================================================
# Prediction API
# ============================================================

@app.route(
    "/house/v1/predict",
    methods=["POST"]
)
def predict():

    try:

        payload = request.get_json(
            silent=True
        )


        if payload is None:

            return jsonify({

                "error":
                    "Invalid JSON request."

            }), 400


        input_df = prepare_input(
            payload
        )


        prediction = (
            pipeline
            .predict(
                input_df
            )[0]
        )


        prediction = float(
            prediction
        )


        if not np.isfinite(
            prediction
        ):

            raise ValueError(
                "The model returned an invalid prediction."
            )


        return jsonify({

            "predicted_price":
                prediction,

            "predicted_price_formatted":
                f"£{prediction:,.2f}",

            "model":
                MODEL_NAME,

            "confidence_indicator":
                round(
                    MODEL_CONFIDENCE_INDICATOR,
                    2
                ),

            "confidence_note":
                "This indicator is derived from "
                "validation R2 and is not a probability.",

            "validation_metrics": {

                "MAE":
                    MODEL_VALIDATION_METRICS[
                        "mae"
                    ],

                "RMSE":
                    MODEL_VALIDATION_METRICS[
                        "rmse"
                    ],

                "R2":
                    MODEL_VALIDATION_METRICS[
                        "r2"
                    ]

            },

            "currency":
                "GBP",

            "target":
                "history_price"

        })


    except ValueError as error:

        return jsonify({

            "error":
                str(error)

        }), 400


    except Exception as error:

        app.logger.exception(
            "Prediction failed."
        )


        return jsonify({

            "error":
                "Prediction failed.",

            "details":
                str(error)

        }), 500


# ============================================================
# Health check
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "model":
            MODEL_NAME,

        "pipeline_loaded":
            True

    })


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    print(
        "============================================================"
    )

    print(
        "London House Price Prediction"
    )

    print(
        "Model:",
        MODEL_NAME
    )

    print(
        "Model path:",
        MODEL_PATH
    )

    print(
        "Web:",
        "http://127.0.0.1:5000/"
    )

    print(
        "API:",
        "http://127.0.0.1:5000/house/v1/predict"
    )

    print(
        "============================================================"
    )


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )