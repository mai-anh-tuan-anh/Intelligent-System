from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
WEB_DIR = ROOT_DIR / "web"


# ============================================================
# ARTIFACT PATHS
# ============================================================

BEST_ML_PATH = MODEL_DIR / "diabetes_best_ml_bundle.joblib"

DL_WEIGHTS_PATH = MODEL_DIR / "diabetes_dl_weights.npz"
DL_CONFIG_PATH = MODEL_DIR / "diabetes_dl_config.json"
DL_PREPROCESSOR_PATH = MODEL_DIR / "diabetes_dl_preprocessor.joblib"
DL_SELECTOR_PATH = MODEL_DIR / "diabetes_dl_selector.joblib"

METADATA_PATH = MODEL_DIR / "diabetes_metadata.json"


REQUIRED_FILES = [
    BEST_ML_PATH,
    DL_WEIGHTS_PATH,
    DL_CONFIG_PATH,
    DL_PREPROCESSOR_PATH,
    DL_SELECTOR_PATH,
    METADATA_PATH,
]


for path in REQUIRED_FILES:
    if not path.exists():
        raise RuntimeError(
            f"Missing required artifact:\n{path}"
        )


# ============================================================
# LOAD BEST ML ARTIFACT
# ============================================================

best_ml_bundle = joblib.load(
    BEST_ML_PATH
)

BEST_ML_NAME = best_ml_bundle[
    "model_name"
]

ML_MODEL = best_ml_bundle[
    "model"
]

ML_PREPROCESSOR = best_ml_bundle[
    "preprocessor"
]

ML_SELECTOR = best_ml_bundle[
    "selector"
]

CLASS_ORDER = list(
    best_ml_bundle.get(
        "classes",
        ["Low", "Moderate", "High"]
    )
)


# ============================================================
# LOAD DEEP LEARNING ARTIFACTS
# ============================================================

dl_weights = np.load(
    DL_WEIGHTS_PATH
)

DL_CONFIG = json.loads(
    DL_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)

DL_PREPROCESSOR = joblib.load(
    DL_PREPROCESSOR_PATH
)

DL_SELECTOR = joblib.load(
    DL_SELECTOR_PATH
)

DEPLOYMENT_METADATA = json.loads(
    METADATA_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# LOAD DL PARAMETERS
# ============================================================

W1 = dl_weights["W1"]
b1 = dl_weights["b1"]

W2 = dl_weights["W2"]
b2 = dl_weights["b2"]

W3 = dl_weights["W3"]
b3 = dl_weights["b3"]


# ============================================================
# EXACT RAW FEATURES USED BY A3 NOTEBOOK
# ============================================================

RAW_DEPLOYMENT_FEATURES = [
    "Age",
    "BMI",
    "Waist_Circumference_cm",
    "Blood_Glucose",
    "HbA1c",
    "Fasting_Blood_Sugar",
    "Insulin_Level",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Total_Cholesterol",
    "HDL",
    "Family_History_Diabetes",
    "Hypertension",
    "Heart_Disease",
    "Diet_Quality",
    "Physical_Activity_Level",
]

ENGINEERED_FEATURES = [
    "Pulse_Pressure",
    "Chol_HDL_Ratio",
]


FINAL_FEATURES = [
    "Age",
    "BMI",
    "Waist_Circumference_cm",
    "Blood_Glucose",
    "HbA1c",
    "Fasting_Blood_Sugar",
    "Insulin_Level",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Total_Cholesterol",
    "HDL",
    "Family_History_Diabetes",
    "Hypertension",
    "Heart_Disease",
    "Diet_Quality",
    "Physical_Activity_Level",
    "Pulse_Pressure",
    "Chol_HDL_Ratio",
]


# ============================================================
# FEATURE ENGINEERING
# Same logic as the notebook
# ============================================================

def add_engineered_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    data = data.copy()

    if {
        "Blood_Pressure_Systolic",
        "Blood_Pressure_Diastolic",
    }.issubset(data.columns):

        data["Pulse_Pressure"] = (
            data["Blood_Pressure_Systolic"]
            - data["Blood_Pressure_Diastolic"]
        )

    if {
        "Total_Cholesterol",
        "HDL",
    }.issubset(data.columns):

        denominator = (
            data["HDL"]
            .replace(0, np.nan)
        )

        data["Chol_HDL_Ratio"] = (
            data["Total_Cholesterol"]
            / denominator
        )

    return data


# ============================================================
# NUMPY DEEP LEARNING
# Exact architecture:
#
# 17 -> 64 -> 32 -> 3
#
# ReLU
# ReLU
# Softmax
# ============================================================

def relu(
    x: np.ndarray,
) -> np.ndarray:

    return np.maximum(
        0,
        x,
    )


def softmax(
    x: np.ndarray,
) -> np.ndarray:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x - np.max(
        x,
        axis=1,
        keepdims=True,
    )

    exp_x = np.exp(
        x
    )

    return exp_x / np.sum(
        exp_x,
        axis=1,
        keepdims=True,
    )


def dl_forward(
    X: np.ndarray,
) -> np.ndarray:

    z1 = (
        X @ W1
        + b1
    )

    h1 = relu(
        z1
    )

    z2 = (
        h1 @ W2
        + b2
    )

    h2 = relu(
        z2
    )

    z3 = (
        h2 @ W3
        + b3
    )

    y_prob = softmax(
        z3
    )

    return y_prob


# ============================================================
# VALIDATE RAW INPUT
# ============================================================

def validate_raw_input(
    patient: dict[str, Any]
) -> None:

    missing = [
        feature
        for feature in RAW_DEPLOYMENT_FEATURES
        if feature not in patient
    ]

    if missing:

        raise HTTPException(
            status_code=422,
            detail={
                "message":
                    "Missing required patient features.",
                "missing":
                    missing,
            },
        )


# ============================================================
# PREPARE PATIENT
# ============================================================

def prepare_patient(
    patient: dict[str, Any],
) -> pd.DataFrame:

    validate_raw_input(
        patient
    )

    input_df = pd.DataFrame(
        [patient]
    )

    # Same feature engineering
    engineered = (
        add_engineered_features(
            input_df
        )
    )

    missing_final = [
        feature
        for feature in FINAL_FEATURES
        if feature not in engineered.columns
    ]

    if missing_final:

        raise HTTPException(
            status_code=422,
            detail={
                "message":
                    "Could not create final model features.",
                "missing":
                    missing_final,
            },
        )

    return engineered[
        FINAL_FEATURES
    ].copy()


# ============================================================
# BEST ML PREDICTION
# ============================================================

def predict_best_ml(
    patient_df: pd.DataFrame,
) -> dict[str, Any]:

    transformed = (
        ML_PREPROCESSOR.transform(
            patient_df
        )
    )

    selected = (
        ML_SELECTOR.transform(
            transformed
        )
    )

    probability = (
        ML_MODEL.predict_proba(
            selected
        )[0]
    )

    predicted_id = int(
        np.argmax(
            probability
        )
    )

    top_indices = np.argsort(
        probability
    )[::-1]


    return {

        "model":
            BEST_ML_NAME,

        "prediction":
            CLASS_ORDER[predicted_id],

        "class_id":
            predicted_id,

        "confidence":
            float(
                probability[
                    predicted_id
                ]
            ),

        "probabilities": {

            CLASS_ORDER[i]:
                float(
                    probability[i]
                )

            for i in range(
                len(CLASS_ORDER)
            )
        },

        "top3": [

            {
                "rank":
                    rank,

                "class":
                    CLASS_ORDER[
                        int(idx)
                    ],

                "probability":
                    float(
                        probability[
                            int(idx)
                        ]
                    ),
            }

            for rank, idx
            in enumerate(
                top_indices[:3],
                start=1,
            )

        ],
    }


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

def predict_deep_learning(
    patient_df: pd.DataFrame,
) -> dict[str, Any]:

    transformed = (
        DL_PREPROCESSOR.transform(
            patient_df
        )
    )

    selected = (
        DL_SELECTOR.transform(
            transformed
        )
    )

    X = np.asarray(
        selected,
        dtype=np.float64,
    )

    probability = (
        dl_forward(
            X
        )[0]
    )

    predicted_id = int(
        np.argmax(
            probability
        )
    )

    top_indices = np.argsort(
        probability
    )[::-1]


    return {

        "model":
            "Deep Learning - NumPy MLP",

        "prediction":
            CLASS_ORDER[predicted_id],

        "class_id":
            predicted_id,

        "confidence":
            float(
                probability[
                    predicted_id
                ]
            ),

        "probabilities": {

            CLASS_ORDER[i]:
                float(
                    probability[i]
                )

            for i in range(
                len(CLASS_ORDER)
            )
        },

        "top3": [

            {
                "rank":
                    rank,

                "class":
                    CLASS_ORDER[
                        int(idx)
                    ],

                "probability":
                    float(
                        probability[
                            int(idx)
                        ]
                    ),
            }

            for rank, idx
            in enumerate(
                top_indices[:3],
                start=1,
            )

        ],
    }


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Diabetes Risk Intelligence",
    description=(
        "Assignment 3 diabetes-risk deployment "
        "using the saved Best ML model and "
        "NumPy Deep Learning model."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PatientRequest(
    BaseModel
):

    Age: float = Field(
        ge=18,
        le=120,
    )

    BMI: float = Field(
        ge=0,
        le=100,
    )

    Waist_Circumference_cm: float = Field(
        ge=0,
        le=250,
    )

    Blood_Glucose: float = Field(
        ge=0,
        le=500,
    )

    HbA1c: float = Field(
        ge=0,
        le=30,
    )

    Fasting_Blood_Sugar: float = Field(
        ge=0,
        le=500,
    )

    Insulin_Level: float = Field(
        ge=0,
        le=1000,
    )

    Blood_Pressure_Systolic: float = Field(
        ge=50,
        le=300,
    )

    Blood_Pressure_Diastolic: float = Field(
        ge=30,
        le=200,
    )

    Total_Cholesterol: float = Field(
        ge=0,
        le=500,
    )

    HDL: float = Field(
        ge=1,
        le=200,
    )

    Family_History_Diabetes: str

    Hypertension: str

    Heart_Disease: str

    Diet_Quality: str

    Physical_Activity_Level: str


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
def health():

    return {

        "status":
            "ok",

        "best_ml":
            BEST_ML_NAME,

        "deep_learning":
            "NumPy MLP",

        "classes":
            CLASS_ORDER,

        "architecture":
            DEPLOYMENT_METADATA[
                "deep_learning"
            ][
                "architecture"
            ],
    }


# ============================================================
# METADATA
# ============================================================

@app.get(
    "/metadata"
)
def metadata():

    return {

        "dataset":
            DEPLOYMENT_METADATA[
                "dataset"
            ],

        "target":
            DEPLOYMENT_METADATA[
                "task"
            ][
                "target"
            ],

        "classes":
            CLASS_ORDER,

        "features":
            RAW_DEPLOYMENT_FEATURES,

        "engineered_features":
            ENGINEERED_FEATURES,

        "best_ml":
            DEPLOYMENT_METADATA[
                "best_machine_learning"
            ],

        "deep_learning":
            DEPLOYMENT_METADATA[
                "deep_learning"
            ],

        "representation":
            DEPLOYMENT_METADATA[
                "representation"
            ],

        "feature_engineering":
            DEPLOYMENT_METADATA[
                "feature_engineering"
            ],

        "comparison":
            DEPLOYMENT_METADATA[
                "four_model_comparison"
            ],
    }


# ============================================================
# PREDICT BOTH MODELS
# ============================================================

@app.post(
    "/predict"
)
def predict(
    request: PatientRequest,
):

    patient = request.model_dump()

    patient_df = prepare_patient(
        patient
    )

    ml_result = predict_best_ml(
        patient_df
    )

    dl_result = predict_deep_learning(
        patient_df
    )

    return {

        "input":
            patient,

        "engineered_features": {

            "Pulse_Pressure":
                float(
                    patient[
                        "Blood_Pressure_Systolic"
                    ]
                    -
                    patient[
                        "Blood_Pressure_Diastolic"
                    ]
                ),

            "Chol_HDL_Ratio":
                float(
                    patient[
                        "Total_Cholesterol"
                    ]
                    /
                    patient[
                        "HDL"
                    ]
                ),
        },

        "best_ml":
            ml_result,

        "deep_learning":
            dl_result,

        "agreement":
            (
                ml_result["prediction"]
                ==
                dl_result["prediction"]
            ),

        "model_metrics": {

            "best_ml":
                DEPLOYMENT_METADATA[
                    "best_machine_learning"
                ][
                    "metrics"
                ],

            "deep_learning":
                DEPLOYMENT_METADATA[
                    "deep_learning"
                ][
                    "metrics"
                ],
        },

    }


# ============================================================
# SERVE WEB
# ============================================================

if not WEB_DIR.exists():

    raise RuntimeError(
        f"Web directory not found: {WEB_DIR}"
    )


app.mount(
    "/web",
    StaticFiles(
        directory=str(WEB_DIR)
    ),
    name="web",
)


@app.get(
    "/",
    include_in_schema=False,
)
def home():

    return FileResponse(
        WEB_DIR / "index.html"
    )


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )