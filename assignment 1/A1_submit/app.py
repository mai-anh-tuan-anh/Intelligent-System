import os

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "diabetes.csv"
)

TARGET = "Outcome"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        "diabetes.csv was not found. "
        "Please put diabetes.csv in the same folder as app.py."
    )


df = pd.read_csv(
    DATA_PATH
)

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].astype(int).copy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",

            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),

            FEATURES
        )
    ],

    remainder="drop"
)


# ============================================================
# FINAL MODEL
# ============================================================

final_model = Pipeline(

    [
        (
            "preprocess",
            preprocessor
        ),

        (
            "model",

            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


cv_scores = cross_validate(

    final_model,

    X_train,

    y_train,

    cv=cv,

    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],

    n_jobs=-1
)


cv_metrics = {

    "accuracy":
        float(
            cv_scores[
                "test_accuracy"
            ].mean()
        ),

    "precision":
        float(
            cv_scores[
                "test_precision"
            ].mean()
        ),

    "recall":
        float(
            cv_scores[
                "test_recall"
            ].mean()
        ),

    "f1":
        float(
            cv_scores[
                "test_f1"
            ].mean()
        ),

    "roc_auc":
        float(
            cv_scores[
                "test_roc_auc"
            ].mean()
        )
}


# ============================================================
# FINAL TRAINING
# ============================================================

final_model.fit(
    X_train,
    y_train
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

y_test_pred = final_model.predict(
    X_test
)

y_test_prob = final_model.predict_proba(
    X_test
)[:, 1]


test_metrics = {

    "accuracy":
        float(
            accuracy_score(
                y_test,
                y_test_pred
            )
        ),

    "precision":
        float(
            precision_score(
                y_test,
                y_test_pred,
                zero_division=0
            )
        ),

    "recall":
        float(
            recall_score(
                y_test,
                y_test_pred,
                zero_division=0
            )
        ),

    "f1":
        float(
            f1_score(
                y_test,
                y_test_pred,
                zero_division=0
            )
        ),

    "roc_auc":
        float(
            roc_auc_score(
                y_test,
                y_test_prob
            )
        )
}


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(
    y_test,
    y_test_pred
)


# ============================================================
# DATASET INFORMATION
# ============================================================

class_counts = (

    y.value_counts()

    .sort_index()

)


class_distribution = {

    "No Diabetes":
        int(
            class_counts.get(
                0,
                0
            )
        ),

    "Diabetes":
        int(
            class_counts.get(
                1,
                0
            )
        )
}


feature_means = {

    feature:
        float(
            df[feature].mean()
        )

    for feature in FEATURES
}


# ============================================================
# DASHBOARD INFORMATION
# ============================================================

dashboard_data = {

    "rows":
        int(
            len(df)
        ),

    "features":
        int(
            len(FEATURES)
        ),

    "train_rows":
        int(
            len(X_train)
        ),

    "test_rows":
        int(
            len(X_test)
        ),

    "positive_rate":
        float(
            y.mean()
        ),

    "model":
        "Random Forest",

    "model_detail":
        "200 trees · max_depth=8 · min_samples_leaf=4",

    "cv_metrics":
        cv_metrics,

    "test_metrics":
        test_metrics,

    "confusion_matrix":
        confusion.tolist(),

    "class_distribution":
        class_distribution,

    "feature_means":
        feature_means
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# DASHBOARD API
# ============================================================

@app.route(
    "/api/dashboard",
    methods=["GET"]
)
def dashboard():

    return jsonify(
        dashboard_data
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify(
                {
                    "error":
                        "No input data received."
                }
            ), 400


        # ----------------------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------------------

        sample = pd.DataFrame(
            [
                {
                    feature:
                        float(
                            data[feature]
                        )

                    for feature in FEATURES
                }
            ]
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            final_model.predict(
                sample
            )[0]
        )


        probabilities = (
            final_model.predict_proba(
                sample
            )[0]
        )


        probability_negative = float(
            probabilities[0]
        )

        probability_positive = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # Human-readable result
        # ----------------------------------------------------

        if prediction == 1:

            label = (
                "Diabetes predicted"
            )

            status = "positive"

        else:

            label = (
                "No diabetes predicted"
            )

            status = "negative"


        return jsonify(

            {
                "prediction":
                    prediction,

                "label":
                    label,

                "status":
                    status,

                "probability_negative":
                    probability_negative,

                "probability_positive":
                    probability_positive
            }

        )


    except Exception as error:

        return jsonify(

            {
                "error":
                    str(error)
            }

        ), 400


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "Final model loaded successfully!"
    )

    print(
        "Model: Random Forest"
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    print(
        "Test F1:",
        round(
            test_metrics["f1"],
            4
        )
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )