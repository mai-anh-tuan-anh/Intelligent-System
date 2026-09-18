from pathlib import Path
import json
import math
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
WEB_DIR = PROJECT_DIR / "web"

BEST_ML_PATH = MODEL_DIR / "housePrice_best_ml_bundle.joblib"
DL_PATH = MODEL_DIR / "housePrice_dl_model.joblib"
METADATA_PATH = MODEL_DIR / "housePrice_metadata.json"
PREDICTIONS_PATH = (
    MODEL_DIR / "housePrice_deployment_predictions.csv"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CHECK FILES
# ============================================================

required_files = {
    "Best ML model": BEST_ML_PATH,
    "Deep Learning model": DL_PATH,
    "Metadata": METADATA_PATH,
    "Prediction CSV": PREDICTIONS_PATH,
    "index.html": WEB_DIR / "index.html",
    "styles.css": WEB_DIR / "styles.css",
    "app.js": WEB_DIR / "app.js",
}

for name, path in required_files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} not found:\n{path}\n"
        )


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("=" * 70)
print("HOUSE PRICE AI - LOADING DEPLOYMENT")
print("=" * 70)

print(f"ML artifact : {BEST_ML_PATH}")
print(f"DL artifact : {DL_PATH}")
print(f"Metadata    : {METADATA_PATH}")
print(f"Predictions : {PREDICTIONS_PATH}")
print(f"Web         : {WEB_DIR}")

ML_BUNDLE = joblib.load(BEST_ML_PATH)
DL_BUNDLE = joblib.load(DL_PATH)

METADATA = json.loads(
    METADATA_PATH.read_text(
        encoding="utf-8"
    )
)

EVAL_DF = pd.read_csv(
    PREDICTIONS_PATH
)

print("Artifacts loaded successfully.")


# ============================================================
# MODEL CONFIG
# ============================================================

BEST_ML_NAME = ML_BUNDLE["model_name"]
DL_NAME = DL_BUNDLE["model_name"]

TARGET_COLUMN = ML_BUNDLE["target_column"]

CATEGORICAL_LOW = ML_BUNDLE["categorical_low"]
CATEGORICAL_HIGH = ML_BUNDLE["categorical_high"]
NUMERIC_FEATURES = ML_BUNDLE["numeric_features"]
FEATURE_COLUMNS = ML_BUNDLE["feature_columns"]

PREPROCESSING_INFO = ML_BUNDLE["preprocessing_info"]


# ============================================================
# JSON HELPERS
# ============================================================

def json_safe(value: Any):

    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def json_ready(value: Any):

    if isinstance(value, dict):

        return {
            str(key): json_ready(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            json_ready(item)
            for item in value
        ]

    return json_safe(value)


# ============================================================
# RELU
# ============================================================

def relu(
    x: np.ndarray
) -> np.ndarray:

    return np.maximum(
        0.0,
        x
    )


# ============================================================
# PREPROCESSING
# ============================================================

def transform_input(
    input_df: pd.DataFrame
) -> pd.DataFrame:

    data = input_df.copy()

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    for col in NUMERIC_FEATURES:

        if col not in data.columns:
            data[col] = np.nan

    data[NUMERIC_FEATURES] = (
        data[NUMERIC_FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    data[NUMERIC_FEATURES] = (
        data[NUMERIC_FEATURES]
        .fillna(
            PREPROCESSING_INFO[
                "numeric_medians"
            ]
        )
    )

    # --------------------------------------------------------
    # HIGH CARDINALITY
    # --------------------------------------------------------

    for col in CATEGORICAL_HIGH:

        if col not in data.columns:
            data[col] = ""

        data[col] = (
            data[col]
            .fillna("")
            .astype(str)
        )

        frequency_map = (
            PREPROCESSING_INFO[
                "frequency_maps"
            ][col]
        )

        data[
            f"{col}_freq"
        ] = (
            data[col]
            .map(frequency_map)
            .fillna(0.0)
        )

    # --------------------------------------------------------
    # LOW CARDINALITY
    # --------------------------------------------------------

    for col in CATEGORICAL_LOW:

        if col not in data.columns:
            data[col] = np.nan

        data[col] = (
            data[col]
            .fillna(
                PREPROCESSING_INFO[
                    "categorical_modes"
                ][col]
            )
            .astype(str)
        )

    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    dummies = pd.get_dummies(
        data[CATEGORICAL_LOW],
        prefix=CATEGORICAL_LOW,
        dtype=float
    )

    dummies = dummies.reindex(
        columns=(
            PREPROCESSING_INFO[
                "dummy_columns"
            ]
        ),
        fill_value=0.0
    )

    # --------------------------------------------------------
    # CONTINUOUS FEATURES
    # --------------------------------------------------------

    frequency_columns = [
        f"{col}_freq"
        for col in CATEGORICAL_HIGH
    ]

    continuous = pd.concat(
        [
            data[
                NUMERIC_FEATURES
            ].astype(float),

            data[
                frequency_columns
            ].astype(float),
        ],
        axis=1
    )

    means = pd.Series(
        PREPROCESSING_INFO[
            "continuous_means"
        ]
    )

    stds = pd.Series(
        PREPROCESSING_INFO[
            "continuous_stds"
        ]
    )

    continuous = (
        continuous - means
    ) / stds

    # --------------------------------------------------------
    # FINAL FEATURES
    # --------------------------------------------------------

    processed = pd.concat(
        [
            continuous.reset_index(
                drop=True
            ),

            dummies.reset_index(
                drop=True
            )
        ],
        axis=1
    )

    processed = processed.reindex(
        columns=(
            PREPROCESSING_INFO[
                "feature_names"
            ]
        ),
        fill_value=0.0
    )

    processed = (
        processed
        .replace(
            [np.inf, -np.inf],
            0.0
        )
        .fillna(0.0)
    )

    return processed.astype(
        np.float64
    )


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

def dl_predict(
    processed: pd.DataFrame
) -> np.ndarray:

    weights = DL_BUNDLE["weights"]

    W1 = np.asarray(
        weights["W1"],
        dtype=np.float64
    )

    b1 = np.asarray(
        weights["b1"],
        dtype=np.float64
    )

    W2 = np.asarray(
        weights["W2"],
        dtype=np.float64
    )

    b2 = np.asarray(
        weights["b2"],
        dtype=np.float64
    )

    W3 = np.asarray(
        weights["W3"],
        dtype=np.float64
    )

    b3 = np.asarray(
        weights["b3"],
        dtype=np.float64
    )

    X = processed.to_numpy(
        dtype=np.float64
    )

    # Input -> Hidden 1
    h1 = relu(
        X @ W1 + b1
    )

    # Hidden 1 -> Hidden 2
    h2 = relu(
        h1 @ W2 + b2
    )

    # Hidden 2 -> Output
    prediction_scaled = (
        h2 @ W3 + b3
    )

    target_mean = float(
        DL_BUNDLE[
            "target_scaling"
        ]["mean"]
    )

    target_std = float(
        DL_BUNDLE[
            "target_scaling"
        ]["std"]
    )

    prediction = (
        prediction_scaled
        * target_std
        + target_mean
    )

    return prediction.reshape(-1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> dict:

    error = (
        actual - predicted
    )

    absolute_error = np.abs(
        error
    )

    mae = np.mean(
        absolute_error
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(
        mse
    )

    mape = (
        np.mean(
            absolute_error
            /
            np.maximum(
                np.abs(actual),
                1e-8
            )
        )
        *
        100.0
    )

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (
            actual
            -
            actual.mean()
        ) ** 2
    )

    r2 = (
        1.0
        -
        (
            ss_res
            /
            max(
                ss_tot,
                1e-8
            )
        )
    )

    median_ae = np.median(
        absolute_error
    )

    max_ae = np.max(
        absolute_error
    )

    bias = np.mean(
        predicted
        -
        actual
    )

    return {

        "MAE":
            float(mae),

        "MSE":
            float(mse),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),

        "MAPE (%)":
            float(mape),

        "MedianAE":
            float(median_ae),

        "MaxAE":
            float(max_ae),

        "Bias":
            float(bias)
    }


# ============================================================
# BUILD EVALUATION
# ============================================================

def build_evaluation():

    actual = (
        EVAL_DF[
            "Actual"
        ]
        .astype(float)
        .to_numpy()
    )

    ml_pred = (
        EVAL_DF[
            "Best_ML_Predicted"
        ]
        .astype(float)
        .to_numpy()
    )

    dl_pred = (
        EVAL_DF[
            "Deep_Learning_Predicted"
        ]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    ml_metrics = calculate_metrics(
        actual,
        ml_pred
    )

    dl_metrics = calculate_metrics(
        actual,
        dl_pred
    )

    metrics = [

        {
            "model":
                BEST_ML_NAME,

            **ml_metrics
        },

        {
            "model":
                DL_NAME,

            **dl_metrics
        }

    ]

    # --------------------------------------------------------
    # PRICE QUARTILES
    # --------------------------------------------------------

    quartile_labels = [
        "Q1 Low",
        "Q2",
        "Q3",
        "Q4 High"
    ]

    quartiles = pd.qcut(
        actual,
        q=4,
        labels=quartile_labels,
        duplicates="drop"
    )

    # Convert directly to NumPy.
    quartile_array = np.asarray(
        quartiles.astype(str)
    )

    quartile_mae = []

    for label in quartile_labels:

        mask = (
            quartile_array
            ==
            label
        )

        if not np.any(mask):
            continue

        ml_mae = np.mean(
            np.abs(
                actual[mask]
                -
                ml_pred[mask]
            )
        )

        dl_mae = np.mean(
            np.abs(
                actual[mask]
                -
                dl_pred[mask]
            )
        )

        quartile_mae.append(
            {
                "group":
                    label,

                "actual_min":
                    float(
                        actual[mask].min()
                    ),

                "actual_max":
                    float(
                        actual[mask].max()
                    ),

                BEST_ML_NAME:
                    float(
                        ml_mae
                    ),

                DL_NAME:
                    float(
                        dl_mae
                    )
            }
        )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED SAMPLE
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    sample_size = min(
        2500,
        len(actual)
    )

    if sample_size < len(actual):

        sample_indexes = rng.choice(
            len(actual),
            size=sample_size,
            replace=False
        )

        sample_indexes.sort()

    else:

        sample_indexes = np.arange(
            len(actual)
        )

    scatter = []

    for index in sample_indexes:

        scatter.append(
            {
                "actual":
                    float(
                        actual[index]
                    ),

                "ml":
                    float(
                        ml_pred[index]
                    ),

                "dl":
                    float(
                        dl_pred[index]
                    )
            }
        )

    # --------------------------------------------------------
    # RESIDUALS
    # --------------------------------------------------------

    residual_size = min(
        5000,
        len(actual)
    )

    if residual_size < len(actual):

        residual_indexes = rng.choice(
            len(actual),
            size=residual_size,
            replace=False
        )

    else:

        residual_indexes = np.arange(
            len(actual)
        )

    residuals = {

        "ml": [

            float(value)

            for value in (
                actual[
                    residual_indexes
                ]
                -
                ml_pred[
                    residual_indexes
                ]
            )

        ],

        "dl": [

            float(value)

            for value in (
                actual[
                    residual_indexes
                ]
                -
                dl_pred[
                    residual_indexes
                ]
            )

        ]

    }

    # --------------------------------------------------------
    # TABLE PREVIEW
    # --------------------------------------------------------

    preview = EVAL_DF[
        [
            "Actual",
            "Best_ML_Predicted",
            "Deep_Learning_Predicted",
            "Best_ML_Absolute_Error",
            "Deep_Learning_Absolute_Error",
        ]
    ].head(20)

    return {

        "sample_count":
            int(
                len(actual)
            ),

        "metrics":
            metrics,

        "quartile_mae":
            quartile_mae,

        "scatter":
            scatter,

        "residuals":
            residuals,

        "preview":
            preview
            .round(2)
            .to_dict(
                orient="records"
            )

    }


EVALUATION = build_evaluation()


# ============================================================
# FORM SCHEMA
# ============================================================

def build_schema():

    dummy_columns = (
        PREPROCESSING_INFO[
            "dummy_columns"
        ]
    )

    low_categories = []

    for col in CATEGORICAL_LOW:

        prefix = (
            f"{col}_"
        )

        options = sorted(
            [
                str(name)[
                    len(prefix):
                ]

                for name
                in dummy_columns

                if str(name).startswith(
                    prefix
                )
            ]
        )

        low_categories.append(
            {
                "name":
                    col,

                "options":
                    options
            }
        )

    high_categories = []

    for col in CATEGORICAL_HIGH:

        frequency_map = (
            PREPROCESSING_INFO[
                "frequency_maps"
            ][col]
        )

        options = sorted(
            [
                str(value)
                for value
                in frequency_map.keys()
            ],

            key=lambda value:
                frequency_map.get(
                    value,
                    0
                ),

            reverse=True
        )

        high_categories.append(
            {
                "name":
                    col,

                "options":
                    options[:300],

                "allow_custom":
                    True
            }
        )

    numeric = []

    for col in NUMERIC_FEATURES:

        numeric.append(
            {
                "name":
                    col,

                "median":
                    json_safe(
                        PREPROCESSING_INFO[
                            "numeric_medians"
                        ].get(
                            col
                        )
                    )
            }
        )

    return {

        "target":
            TARGET_COLUMN,

        "numeric":
            numeric,

        "categorical_low":
            low_categories,

        "categorical_high":
            high_categories,

        "feature_count":
            len(
                PREPROCESSING_INFO[
                    "feature_names"
                ]
            )
    }


SCHEMA = build_schema()


# ============================================================
# HEALTH API
# ============================================================

@app.get("/api/health")
def health():

    return jsonify(
        {
            "status":
                "ok",

            "best_ml_model":
                BEST_ML_NAME,

            "deep_learning_model":
                DL_NAME,

            "target":
                TARGET_COLUMN,

            "feature_count":
                len(
                    PREPROCESSING_INFO[
                        "feature_names"
                    ]
                )
        }
    )


# ============================================================
# METADATA API
# ============================================================

@app.get("/api/metadata")
def metadata():

    ml_model = (
        ML_BUNDLE["model"]
    )

    ml_parameters = {}

    if hasattr(
        ml_model,
        "get_params"
    ):

        raw_params = (
            ml_model.get_params(
                deep=False
            )
        )

        allowed = {

            "max_iter",

            "learning_rate",

            "max_leaf_nodes",

            "l2_regularization",

            "max_depth",

            "min_samples_leaf",

            "n_estimators",

            "random_state",

        }

        for key, value in (
            raw_params.items()
        ):

            if key in allowed:

                ml_parameters[key] = (
                    json_safe(value)
                )

    result = {

        "task":
            METADATA.get(
                "task",
                "regression"
            ),

        "target":
            TARGET_COLUMN,

        "best_machine_learning":
            {

                "model":
                    BEST_ML_NAME,

                "metrics":
                    METADATA.get(
                        "best_machine_learning",
                        {}
                    ).get(
                        "metrics",
                        {}
                    ),

                "parameters":
                    ml_parameters,

                "selection_criterion":
                    ML_BUNDLE.get(
                        "selection_criterion"
                    )

            },

        "deep_learning":
            {

                "model":
                    DL_NAME,

                "metrics":
                    METADATA.get(
                        "deep_learning",
                        {}
                    ).get(
                        "metrics",
                        {}
                    ),

                "architecture":
                    DL_BUNDLE.get(
                        "architecture",
                        {}
                    ),

                "training":
                    {

                        "learning_rate":
                            DL_BUNDLE.get(
                                "learning_rate"
                            ),

                        "epochs":
                            DL_BUNDLE.get(
                                "epochs"
                            ),

                        "batch_size":
                            DL_BUNDLE.get(
                                "batch_size"
                            ),

                        "random_seed":
                            DL_BUNDLE.get(
                                "random_seed"
                            )

                    }

            },

        "preprocessing":
            {

                "numeric_features":
                    NUMERIC_FEATURES,

                "categorical_low":
                    CATEGORICAL_LOW,

                "categorical_high":
                    CATEGORICAL_HIGH,

                "feature_count":
                    len(
                        PREPROCESSING_INFO[
                            "feature_names"
                        ]
                    )

            },

        "probability_output":
            False,

        "probability_note":
            (
                "This is a regression problem. "
                "The models return numeric house "
                "price estimates instead of class "
                "probabilities."
            )
    }

    return jsonify(
        json_ready(
            result
        )
    )


# ============================================================
# SCHEMA API
# ============================================================

@app.get("/api/schema")
def schema():

    return jsonify(
        json_ready(
            SCHEMA
        )
    )


# ============================================================
# EVALUATION API
# ============================================================

@app.get("/api/evaluation")
def evaluation():

    return jsonify(
        json_ready(
            EVALUATION
        )
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/api/predict")
def predict():

    payload = request.get_json(
        silent=True
    )

    if not isinstance(
        payload,
        dict
    ):

        return jsonify(
            {
                "error":
                    "Request body must be a JSON object."
            }
        ), 400

    # --------------------------------------------------------
    # BUILD RAW INPUT
    # --------------------------------------------------------

    row = {}

    for col in FEATURE_COLUMNS:

        row[col] = payload.get(
            col
        )

    raw_df = pd.DataFrame(
        [row]
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    try:

        processed = transform_input(
            raw_df
        )

    except Exception as exc:

        return jsonify(
            {
                "error":
                    f"Preprocessing failed: {exc}"
            }
        ), 400

    # --------------------------------------------------------
    # BEST ML
    # --------------------------------------------------------

    try:

        ml_prediction = float(

            ML_BUNDLE[
                "model"
            ].predict(
                processed
            )[0]

        )

    except Exception as exc:

        return jsonify(
            {
                "error":
                    f"ML prediction failed: {exc}"
            }
        ), 500

    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    try:

        dl_prediction = float(
            dl_predict(
                processed
            )[0]
        )

    except Exception as exc:

        return jsonify(
            {
                "error":
                    (
                        "Deep learning prediction "
                        f"failed: {exc}"
                    )
            }
        ), 500

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    difference = abs(
        ml_prediction
        -
        dl_prediction
    )

    average = (
        ml_prediction
        +
        dl_prediction
    ) / 2.0

    agreement = max(
        0.0,
        100.0
        *
        (
            1.0
            -
            difference
            /
            max(
                abs(average),
                1e-8
            )
        )
    )

    return jsonify(
        {
            "task":
                "house_price_regression",

            "target":
                TARGET_COLUMN,

            "best_machine_learning":
                {
                    "model":
                        BEST_ML_NAME,

                    "prediction":
                        ml_prediction
                },

            "deep_learning":
                {
                    "model":
                        DL_NAME,

                    "prediction":
                        dl_prediction
                },

            "comparison":
                {
                    "difference":
                        difference,

                    "average":
                        average,

                    "agreement":
                        agreement,

                    "note":
                        (
                            "Agreement is a consistency "
                            "indicator between the two "
                            "point estimates. It is not "
                            "a probability."
                        )
                }
        }
    )


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():

    return send_from_directory(
        str(WEB_DIR),
        "index.html"
    )


@app.get("/styles.css")
def styles():

    return send_from_directory(
        str(WEB_DIR),
        "styles.css"
    )


@app.get("/app.js")
def app_js():

    return send_from_directory(
        str(WEB_DIR),
        "app.js"
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("HOUSE PRICE AI WEB APPLICATION")
    print("=" * 70)

    print(
        f"PROJECT : {PROJECT_DIR}"
    )

    print(
        f"MODEL   : {MODEL_DIR}"
    )

    print(
        f"WEB     : {WEB_DIR}"
    )

    print(
        f"ML      : {BEST_ML_NAME}"
    )

    print(
        f"DL      : {DL_NAME}"
    )

    print(
        f"TARGET  : {TARGET_COLUMN}"
    )

    print(
        f"FEATURES: "
        f"{len(PREPROCESSING_INFO['feature_names'])}"
    )

    print(
        "APP     : http://127.0.0.1:5000"
    )

    print(
        "CSS     : http://127.0.0.1:5000/styles.css"
    )

    print(
        "JS      : http://127.0.0.1:5000/app.js"
    )

    print("=" * 70)
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )