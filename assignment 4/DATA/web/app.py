from pathlib import Path
import io
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, jsonify, render_template, request

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
FRAMEWORKS = ["Scratch", "PyTorch", "Keras"]


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def model_path(filename):
    final_dir = MODELS_DIR / "final"
    candidate = final_dir / filename
    if candidate.exists():
        return candidate
    return MODELS_DIR / filename


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_json_file(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# PyTorch model definitions
# =========================
class SimpleCNNPyTorch(nn.Module):
    def __init__(self, in_channels, height, width, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        flat_features = 16 * (height // 2) * (width // 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SimpleMLPPyTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# =========================
# Scratch CNN inference
# =========================
def im2col_nchw(x, kernel_size=3, padding=1, stride=1):
    x = np.asarray(x, dtype=np.float32)
    n, c, h, w = x.shape
    kh = kw = kernel_size

    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        x_pad, (kh, kw), axis=(2, 3)
    )
    windows = windows[:, :, ::stride, ::stride, :, :]

    h_out = windows.shape[2]
    w_out = windows.shape[3]

    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
        n * h_out * w_out, c * kh * kw
    )
    return cols, h_out, w_out


def scratch_conv_forward(x, W, b):
    cols, h_out, w_out = im2col_nchw(
        x,
        kernel_size=W.shape[2],
        padding=1,
        stride=1,
    )
    out = cols @ W.reshape(W.shape[0], -1).T
    out += b.reshape(1, -1)
    out = out.reshape(x.shape[0], h_out, w_out, W.shape[0])
    return out.transpose(0, 3, 1, 2)


def scratch_maxpool(x):
    windows = np.lib.stride_tricks.sliding_window_view(
        x, (2, 2), axis=(2, 3)
    )
    windows = windows[:, :, ::2, ::2, :, :]
    return windows.max(axis=(-1, -2))


def scratch_cnn_logits(x, params):
    x = scratch_conv_forward(x, params["W1"], params["b1"])
    x = np.maximum(0.0, x)
    x = scratch_maxpool(x)
    x = x.reshape(x.shape[0], -1)
    return x @ params["W2"] + params["b2"]


# =========================
# Model caches
# =========================
MODEL_CACHE = {}


def load_image_model(dataset, framework):
    key = (dataset, framework)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    height = 28 if dataset == "MNIST" else 32
    width = height
    channels = 1 if dataset == "MNIST" else 3
    num_classes = 10

    fw = framework.lower()

    if fw == "scratch":
        filename = (
            "mnist_cnn_scratch.joblib"
            if dataset == "MNIST"
            else "cifar10_cnn_scratch.joblib"
        )
        path = model_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")
        model = joblib.load(path)

    elif fw == "pytorch":
        filename = (
            "mnist_cnn_pytorch.pth"
            if dataset == "MNIST"
            else "cifar10_cnn_pytorch.pth"
        )
        path = model_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")

        model = SimpleCNNPyTorch(channels, height, width, num_classes)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

    elif fw == "keras":
        filename = (
            "mnist_cnn_keras.keras"
            if dataset == "MNIST"
            else "cifar10_cnn_keras.keras"
        )
        path = model_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")
        model = keras.models.load_model(path)

    else:
        raise ValueError(f"Unsupported framework: {framework}")

    MODEL_CACHE[key] = model
    return model


def diabetes_csv():
    preferred = DATA_DIR / "diabetes" / "diabetes_012_health_indicators_BRFSS2015.csv"
    if preferred.exists():
        return preferred
    csvs = sorted((DATA_DIR / "diabetes").glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No Diabetes CSV found in data/diabetes")
    return csvs[0]


@lru_cache(maxsize=1)
def diabetes_schema():
    path = diabetes_csv()
    df = pd.read_csv(path)

    candidates = ["Diabetes_012", "Diabetes_binary", "diabetes", "target", "Target"]
    target = next((c for c in candidates if c in df.columns), None)

    if target is None:
        possible = [c for c in df.columns if "diabetes" in c.lower()]
        if len(possible) == 1:
            target = possible[0]

    if target is None:
        raise ValueError("Could not identify Diabetes target column")

    features = [
        c for c in df.drop(columns=[target]).columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    return {
        "target": target,
        "features": features,
        "classes": sorted(df[target].astype(int).unique().tolist()),
        "medians": df[features].median().astype(float).to_dict(),
        "minimums": df[features].min().astype(float).to_dict(),
        "maximums": df[features].max().astype(float).to_dict(),
    }


def load_diabetes_model(framework):
    key = ("Diabetes", framework)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    schema = diabetes_schema()
    fw = framework.lower()

    if fw == "scratch":
        path = model_path("diabetes_mlp_scratch.joblib")
        model = joblib.load(path)
    elif fw == "pytorch":
        path = model_path("diabetes_mlp_pytorch.pth")
        model = SimpleMLPPyTorch(len(schema["features"]), len(schema["classes"]))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
    elif fw == "keras":
        path = model_path("diabetes_mlp_keras.keras")
        model = keras.models.load_model(path)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

    MODEL_CACHE[key] = model
    return model


def predict_diabetes_raw(model, X, framework):
    fw = framework.lower()

    if fw == "scratch":
        p = model
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(0.0, z2)
        return a2 @ p["W3"] + p["b3"]

    if fw == "pytorch":
        with torch.no_grad():
            return model(torch.from_numpy(X)).numpy()

    return model.predict(X, verbose=0)


# =========================
# Metrics and confusion data
# =========================
def metrics_rows():
    rows = []
    for dataset in ["MNIST", "CIFAR-10", "Diabetes"]:
        for fw in FRAMEWORKS:
            filename = {
                ("MNIST", "Scratch"): "mnist_scratch_metrics.json",
                ("MNIST", "PyTorch"): "mnist_pytorch_metrics.json",
                ("MNIST", "Keras"): "mnist_keras_metrics.json",
                ("CIFAR-10", "Scratch"): "cifar10_scratch_metrics.json",
                ("CIFAR-10", "PyTorch"): "cifar10_pytorch_metrics.json",
                ("CIFAR-10", "Keras"): "cifar10_keras_metrics.json",
                ("Diabetes", "Scratch"): "diabetes_scratch_metrics.json",
                ("Diabetes", "PyTorch"): "diabetes_pytorch_metrics.json",
                ("Diabetes", "Keras"): "diabetes_keras_metrics.json",
            }[(dataset, fw)]
            data = load_json_file(METRICS_DIR / filename)
            rows.append({"dataset": dataset, "framework": fw, **data})
    return rows


def prediction_columns(dataset, framework):
    prefix = framework.lower()
    return "true_label", f"{prefix}_prediction"


def confusion_matrix_data(dataset, framework):
    filename = {
        "MNIST": "mnist_predictions.csv",
        "CIFAR-10": "cifar10_predictions.csv",
        "Diabetes": "diabetes_predictions.csv",
    }[dataset]
    path = PREDICTIONS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")

    df = pd.read_csv(path)
    true_col, pred_col = prediction_columns(dataset, framework)

    if pred_col not in df.columns:
        raise ValueError(f"Missing column: {pred_col}")

    if dataset == "MNIST":
        labels = list(range(10))
    elif dataset == "CIFAR-10":
        labels = list(range(10))
    else:
        labels = diabetes_schema()["classes"]

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(df[true_col], df[pred_col]):
        true = int(true)
        pred = int(pred)
        if dataset == "Diabetes":
            if true not in labels or pred not in labels:
                continue
            r = labels.index(true)
            c = labels.index(pred)
        else:
            r, c = true, pred
        matrix[r, c] += 1

    display_labels = (
        MNIST_CLASSES if dataset == "MNIST"
        else CIFAR_CLASSES if dataset == "CIFAR-10"
        else [str(x) for x in labels]
    )
    return {"labels": display_labels, "matrix": matrix.tolist()}


# =========================
# Prediction preprocessing
# =========================
def image_to_model_input(image, dataset):
    if dataset == "MNIST":
        image = image.convert("L").resize((28, 28))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = arr[None, None, :, :]
        return arr

    image = image.convert("RGB").resize((32, 32))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr[None, :, :, :]


def predict_image(dataset, framework, image):
    x = image_to_model_input(image, dataset)
    model = load_image_model(dataset, framework)

    fw = framework.lower()
    if fw == "scratch":
        logits = scratch_cnn_logits(x, model)
    elif fw == "pytorch":
        with torch.no_grad():
            logits = model(torch.from_numpy(x)).numpy()
    else:
        logits = model.predict(np.transpose(x, (0, 2, 3, 1)), verbose=0)

    probs = softmax(logits)[0]
    classes = MNIST_CLASSES if dataset == "MNIST" else CIFAR_CLASSES
    idx = int(np.argmax(probs))

    return {
        "dataset": dataset,
        "framework": framework,
        "prediction": classes[idx],
        "prediction_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": [
            {"class": classes[i], "probability": float(probs[i])}
            for i in range(len(classes))
        ],
    }


# =========================
# Routes
# =========================
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    required = [
        model_path("mnist_cnn_scratch.joblib"),
        model_path("mnist_cnn_pytorch.pth"),
        model_path("mnist_cnn_keras.keras"),
        model_path("cifar10_cnn_scratch.joblib"),
        model_path("cifar10_cnn_pytorch.pth"),
        model_path("cifar10_cnn_keras.keras"),
    ]
    return jsonify({
        "status": "ok",
        "models_ready": all(p.exists() for p in required),
    })


@app.get("/api/metrics")
def api_metrics():
    return jsonify({"rows": metrics_rows()})


@app.get("/api/confusion/<dataset>/<framework>")
def api_confusion(dataset, framework):
    try:
        if dataset not in {"MNIST", "CIFAR-10", "Diabetes"}:
            raise ValueError("Unsupported dataset")
        if framework not in FRAMEWORKS:
            raise ValueError("Unsupported framework")
        return jsonify(confusion_matrix_data(dataset, framework))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/diabetes/schema")
def api_diabetes_schema():
    return jsonify(diabetes_schema())


@app.post("/api/predict/image")
def api_predict_image():
    dataset = request.form.get("dataset", "MNIST")
    framework = request.form.get("framework", "PyTorch")
    file = request.files.get("image")

    if not file:
        return jsonify({"error": "Please upload an image."}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        return jsonify(predict_image(dataset, framework, image))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/predict/diabetes")
def api_predict_diabetes():
    try:
        body = request.get_json(force=True)
        framework = body.get("framework", "PyTorch")
        payload = body.get("features", {})

        schema = diabetes_schema()
        values = []
        for feature in schema["features"]:
            raw = payload.get(feature, schema["medians"][feature])
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float(schema["medians"][feature])
            values.append(value)

        X = np.asarray(values, dtype=np.float32).reshape(1, -1)
        scaler_path = model_path("diabetes_standard_scaler.joblib")
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X).astype(np.float32)

        model = load_diabetes_model(framework)
        logits = predict_diabetes_raw(model, X, framework)
        probs = softmax(logits)[0]
        idx = int(np.argmax(probs))
        classes = schema["classes"]

        return jsonify({
            "dataset": "Diabetes",
            "framework": framework,
            "prediction": int(classes[idx]),
            "prediction_index": idx,
            "confidence": float(probs[idx]),
            "probabilities": [
                {"class": int(classes[i]), "probability": float(probs[i])}
                for i in range(len(classes))
            ],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
