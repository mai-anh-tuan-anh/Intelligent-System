from __future__ import annotations

import json
import re
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
from scipy.sparse import csr_matrix, hstack

# Folder structure expected:
# comments/
# ├── model/                <- this file + requirements.txt
# ├── model_comments/       <- artifacts exported by the final notebook
# ├── web/                  <- index.html, app.js, styles.css
# ├── mobile/               <- Android Studio project
# ├── model/
# └── data/

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
WEB_DIR = ROOT_DIR / "web"

ARTIFACTS = {
    "ml_bundle": MODEL_DIR / "comments_best_ml_bundle.joblib",
    "dl_weights": MODEL_DIR / "comments_dl_weights.npz",
    "dl_config": MODEL_DIR / "comments_dl_config.json",
    "dl_vocab": MODEL_DIR / "comments_dl_vocab.json",
    "metrics": MODEL_DIR / "comments_deployment_metrics.json",
    "comparison": MODEL_DIR / "comments_four_model_comparison.csv",
}

for name, path in ARTIFACTS.items():
    if not path.exists():
        raise RuntimeError(
            f"Missing artifact '{name}': {path}\n"
            "Copy the files exported by the final A3 notebook into comments/model_comments/."
        )

# -----------------------------------------------------------------------------
# Load trained artifacts - no retraining during deployment
# -----------------------------------------------------------------------------
ml_bundle = joblib.load(ARTIFACTS["ml_bundle"])
dl_weights = np.load(ARTIFACTS["dl_weights"])
dl_config = json.loads(ARTIFACTS["dl_config"].read_text(encoding="utf-8"))
dl_vocab = json.loads(ARTIFACTS["dl_vocab"].read_text(encoding="utf-8"))
deployment_metrics = json.loads(ARTIFACTS["metrics"].read_text(encoding="utf-8"))
comparison_df = pd.read_csv(ARTIFACTS["comparison"])

BEST_ML_NAME = str(ml_bundle["model_name"])
ML_MODEL = ml_bundle["model"]
VECTORIZER = ml_bundle["vectorizer"]
SELECTOR = ml_bundle["selector"]
CLASS_ORDER = list(ml_bundle["class_order"])
CLASS_TO_ID = {str(k): int(v) for k, v in ml_bundle["class_to_id"].items()}
ID_TO_CLASS = {i: c for i, c in enumerate(CLASS_ORDER)}

ARCH = dl_config["architecture"]
TOKEN_TO_ID = {str(k): int(v) for k, v in dl_vocab["token_to_id"].items()}
MAX_LEN = int(ARCH["max_len"])
PAD_ID = int(ARCH["padding_id"])
UNK_ID = int(ARCH["unknown_id"])

EMBEDDING = dl_weights["embedding"]
W1 = dl_weights["W1"]
B1 = dl_weights["b1"]
W2 = dl_weights["W2"]
B2 = dl_weights["b2"]
W3 = dl_weights["W3"]
B3 = dl_weights["b3"]

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def normalize_review_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"read full review.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_text(title: Any, content: Any) -> str:
    title_clean = normalize_review_text(title)
    content_clean = normalize_review_text(content)
    if title_clean and content_clean:
        return f"{title_clean} [sep] {content_clean}"
    return title_clean or content_clean


def softmax_1d(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    denom = np.sum(exp_scores)
    return exp_scores / max(denom, 1e-12)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def encode_text(text: str) -> np.ndarray:
    tokens = tokenize(text)
    ids = [TOKEN_TO_ID.get(tok, UNK_ID) for tok in tokens[:MAX_LEN]]
    if len(ids) < MAX_LEN:
        ids.extend([PAD_ID] * (MAX_LEN - len(ids)))
    return np.asarray(ids, dtype=np.int64)


def dl_forward_single(token_ids: np.ndarray) -> np.ndarray:
    emb = EMBEDDING[token_ids]
    mask = token_ids != PAD_ID
    length = max(int(mask.sum()), 1)

    mean_pool = (emb * mask[:, None]).sum(axis=0) / length
    masked_emb = np.where(mask[:, None], emb, -1e9)
    max_pool = masked_emb.max(axis=0)
    pooled = np.concatenate([mean_pool, max_pool], axis=0)

    z1 = pooled @ W1 + B1.reshape(-1)
    h1 = np.maximum(0.0, z1)
    z2 = h1 @ W2 + B2.reshape(-1)
    h2 = np.maximum(0.0, z2)
    z3 = h2 @ W3 + B3.reshape(-1)
    return softmax_1d(z3)


def transform_ml(title: str, content: str, rating: int):
    combined = [combine_text(title, content)]
    tfidf = VECTORIZER.transform(combined)
    selected = SELECTOR.transform(tfidf)
    rating_aux = np.asarray([(rating - 1.0) / 4.0], dtype=np.float64).reshape(1, 1)
    return hstack([selected, csr_matrix(rating_aux)], format="csr")


def predict_ml(title: str, content: str, rating: int) -> dict[str, Any]:
    x = transform_ml(title, content, rating)
    pred_id = int(ML_MODEL.predict(x)[0])

    if BEST_ML_NAME == "Linear SVM":
        raw_scores = np.asarray(ML_MODEL.decision_function(x)[0], dtype=np.float64)
        display_scores = softmax_1d(raw_scores)
        score_note = "Normalized SVM decision score — not a calibrated probability."
    else:
        display_scores = np.asarray(ML_MODEL.predict_proba(x)[0], dtype=np.float64)
        score_note = "Model probability output."

    top_ids = np.argsort(display_scores)[::-1][:5]
    return {
        "model": BEST_ML_NAME,
        "prediction": ID_TO_CLASS[pred_id],
        "class_id": pred_id,
        "top_score": float(display_scores[pred_id]),
        "score_note": score_note,
        "top5": [
            {
                "rank": rank,
                "category": ID_TO_CLASS[int(idx)],
                "score": float(display_scores[int(idx)]),
            }
            for rank, idx in enumerate(top_ids, start=1)
        ],
    }


def predict_dl(title: str, content: str) -> dict[str, Any]:
    ids = encode_text(combine_text(title, content))
    probs = dl_forward_single(ids)
    pred_id = int(np.argmax(probs))
    top_ids = np.argsort(probs)[::-1][:5]
    return {
        "model": "Deep Learning - NumPy",
        "prediction": ID_TO_CLASS[pred_id],
        "class_id": pred_id,
        "top_score": float(probs[pred_id]),
        "score_note": "Softmax probability from the NumPy neural network.",
        "top5": [
            {
                "rank": rank,
                "category": ID_TO_CLASS[int(idx)],
                "probability": float(probs[int(idx)]),
            }
            for rank, idx in enumerate(top_ids, start=1)
        ],
    }


def comparison_payload() -> list[dict[str, Any]]:
    return comparison_df.replace({np.nan: None}).to_dict(orient="records")


app = FastAPI(
    title="Comments Interest Intelligence API",
    version="2.0.0",
    description="A3 e-commerce text-representation deployment: Best ML vs Deep Learning from Scratch.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    review_title: str = Field(default="", max_length=500)
    review_content: str = Field(default="", max_length=10000)
    rating: int = Field(default=5, ge=1, le=5)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "best_ml_model": BEST_ML_NAME,
        "deep_learning": True,
        "classes": len(CLASS_ORDER),
        "task": deployment_metrics.get("task"),
    }


@app.get("/metadata")
def metadata():
    return {
        "task": deployment_metrics.get("task"),
        "classes": CLASS_ORDER,
        "best_ml_model": BEST_ML_NAME,
        "best_ml_metrics": deployment_metrics.get("best_ml_metrics", {}),
        "deep_learning_metrics": deployment_metrics.get("deep_learning_metrics", {}),
        "comparison": comparison_payload(),
        "artifacts": {
            "ml_bundle": ARTIFACTS["ml_bundle"].name,
            "dl_weights": ARTIFACTS["dl_weights"].name,
            "dl_config": ARTIFACTS["dl_config"].name,
            "dl_vocab": ARTIFACTS["dl_vocab"].name,
        },
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.review_title.strip() and not request.review_content.strip():
        raise HTTPException(status_code=422, detail="Please enter a review title or review content.")

    ml_result = predict_ml(request.review_title, request.review_content, request.rating)
    dl_result = predict_dl(request.review_title, request.review_content)
    agreement = ml_result["prediction"] == dl_result["prediction"]

    return {
        "input": {
            "review_title": request.review_title,
            "review_content": request.review_content,
            "rating": request.rating,
        },
        "best_ml": ml_result,
        "deep_learning": dl_result,
        "agreement": agreement,
        "model_metrics": {
            "best_ml": deployment_metrics.get("best_ml_metrics", {}),
            "deep_learning": deployment_metrics.get("deep_learning_metrics", {}),
        },
    }


# Serve the web UI from the same process so the project remains simple to demo.
if not WEB_DIR.exists():
    raise RuntimeError(f"Web directory not found: {WEB_DIR}")
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(str(WEB_DIR / "index.html"))
# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )