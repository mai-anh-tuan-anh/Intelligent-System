from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request


# Purpose: Create the Flask application and resolve paths relative to the web folder.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "wikitext103"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

MODEL_PATH = MODEL_DIR / "rnn_next_word_best.pt"
TRAIN_FILE = DATA_DIR / "wikitext-103" / "wiki.train.tokens"
VALID_FILE = DATA_DIR / "wikitext-103" / "wiki.valid.tokens"
TEST_FILE = DATA_DIR / "wikitext-103" / "wiki.test.tokens"

VOCAB_SIZE_DEFAULT = 30_000
MAX_TRAIN_TOKENS_DEFAULT = 2_000_000
SEQ_LEN_DEFAULT = 32
DEVICE = torch.device("cpu")

app = Flask(__name__)


# Purpose: Tokenize seed text exactly as the notebook does for WikiText word-level inference.
def tokenize_seed(text: str) -> list[str]:
    """Split a seed string into whitespace-delimited word-level tokens."""
    return text.strip().split()


# Purpose: Read WikiText tokens while matching the notebook's <eos> handling and CPU token limit.
def read_wikitext_tokens(path: Path, max_tokens: int | None = None) -> list[str]:
    """Read a WikiText file into a word-level token stream with line-level <eos> markers."""
    tokens: list[str] = []
    if not path.exists():
        return tokens

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tokens.extend(line.split())
            tokens.append("<eos>")
            if max_tokens is not None and len(tokens) >= max_tokens:
                return tokens[:max_tokens]
    return tokens


# Purpose: Reconstruct the exact vocabulary ordering used by the supplied notebook checkpoint.
def build_vocabulary(tokens: list[str], vocab_size: int) -> tuple[dict[str, int], dict[int, str], Counter[str]]:
    """Build token-to-id and id-to-token mappings from the training split."""
    counts = Counter(tokens)
    special_tokens = ["<unk>", "<eos>"]
    regular = [
        token
        for token, _ in counts.most_common()
        if token not in special_tokens
    ][: max(vocab_size - len(special_tokens), 0)]
    vocab = special_tokens + regular
    token_to_id = {token: index for index, token in enumerate(vocab)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    return token_to_id, id_to_token, counts


# Purpose: Define the same RNN architecture used by the training notebook so checkpoints can be loaded.
class RNNLanguageModel(nn.Module):
    """PyTorch RNN language model for word-level next-token prediction."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # Embedding converts vocabulary IDs into dense vectors.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # RNN recurrently updates hidden state across sequence positions.
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        # Dropout is active only during training and is automatically disabled in eval mode.
        self.dropout = nn.Dropout(dropout)
        # Final linear layer maps hidden representation to vocabulary logits.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Step 1: convert token IDs to embedding vectors.
        x = self.embedding(x)
        # Step 2: process the sequence recurrently.
        output, hidden = self.rnn(x, hidden)
        # Step 3: apply regularization layer; disabled automatically during inference.
        output = self.dropout(output)
        # Step 4: convert hidden states to vocabulary logits.
        logits = self.fc(output)
        return logits, hidden


def infer_model_shapes(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Infer vocabulary size, embedding dimension, hidden size, and recurrent layer count from checkpoint tensors."""
    embed_weight = state_dict["embedding.weight"]
    fc_weight = state_dict["fc.weight"]
    recurrent_layers = sorted(
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"rnn\.weight_ih_l(\d+)$", key))
    )
    return {
        "vocab_size": int(embed_weight.shape[0]),
        "embed_dim": int(embed_weight.shape[1]),
        "hidden_size": int(embed_weight.shape[1] if False else fc_weight.shape[1]),
        "num_layers": (max(recurrent_layers) + 1) if recurrent_layers else 1,
    }


# Purpose: Load the trained checkpoint and return both model metadata and raw checkpoint details.
def load_model_and_checkpoint() -> tuple[RNNLanguageModel, dict[str, Any]]:
    """Load the best checkpoint on CPU and reconstruct the architecture from tensor shapes."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model checkpoint: {MODEL_PATH}. Hãy train notebook trước."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    shapes = infer_model_shapes(state_dict)

    model = RNNLanguageModel(
        vocab_size=shapes["vocab_size"],
        embed_dim=shapes["embed_dim"],
        hidden_size=shapes["hidden_size"],
        num_layers=shapes["num_layers"],
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "val_loss": checkpoint.get("val_loss") if isinstance(checkpoint, dict) else None,
        **shapes,
    }
    return model, metadata


# Purpose: Load the same limited training stream used by the notebook and cache its vocabulary.
def build_runtime_resources() -> dict[str, Any]:
    """Build vocabulary and dataset statistics needed by the web application."""
    train_limit = int(os.getenv("RNN_TRAIN_TOKEN_LIMIT", str(MAX_TRAIN_TOKENS_DEFAULT)))
    vocab_limit = int(os.getenv("RNN_VOCAB_SIZE", str(VOCAB_SIZE_DEFAULT)))

    train_tokens = read_wikitext_tokens(TRAIN_FILE, train_limit)
    if not train_tokens:
        raise FileNotFoundError(
            f"Không tìm thấy WikiText training file: {TRAIN_FILE}"
        )

    token_to_id, id_to_token, counts = build_vocabulary(train_tokens, vocab_limit)

    split_paths = {"train": TRAIN_FILE, "validation": VALID_FILE, "test": TEST_FILE}
    split_stats: dict[str, dict[str, Any]] = {}
    for split, path in split_paths.items():
        max_tokens = train_limit if split == "train" else 200_000
        tokens = read_wikitext_tokens(path, max_tokens)
        split_stats[split] = {
            "tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "size_mb": round((path.stat().st_size / (1024**2)) if path.exists() else 0.0, 2),
        }

    unknown_id = token_to_id["<unk>"]
    split_stats["train"]["unk_rate"] = float(np.mean([token_to_id.get(t, unknown_id) == unknown_id for t in train_tokens]))

    # Validation and test are read once for dashboard statistics; no full inference is performed at startup.
    for split in ("validation", "test"):
        tokens = read_wikitext_tokens(split_paths[split], 200_000)
        if tokens:
            split_stats[split]["unk_rate"] = float(
                np.mean([token_to_id.get(t, unknown_id) == unknown_id for t in tokens])
            )

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "train_counts": counts,
        "split_stats": split_stats,
        "train_limit": train_limit,
    }


# Purpose: Convert cross-entropy to perplexity using a numerically safe exponential.
def loss_to_perplexity(loss: float) -> float:
    """Return exp(loss), capped for numerical safety."""
    return float(math.exp(min(float(loss), 20.0)))


# Purpose: Compute parameter counts for major model components for the dashboard.
def parameter_breakdown(model: RNNLanguageModel) -> dict[str, int]:
    """Count learnable parameters by embedding, recurrent layer, and classifier."""
    return {
        "embedding": int(sum(p.numel() for p in model.embedding.parameters())),
        "rnn": int(sum(p.numel() for p in model.rnn.parameters())),
        "classifier": int(sum(p.numel() for p in model.fc.parameters())),
    }


# Purpose: Compute language-model uncertainty statistics from a probability distribution.
def probability_statistics(probs: torch.Tensor) -> dict[str, float]:
    """Return entropy, top-1 probability, top-5 cumulative probability, and margin."""
    values, _ = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    margin = values[:, 0] - values[:, 1] if values.shape[-1] >= 2 else values[:, 0]
    return {
        "entropy": float(entropy.mean().item()),
        "top1_probability": float(values[:, 0].mean().item()),
        "top5_cumulative_probability": float(values.sum(dim=-1).mean().item()),
        "top1_top2_margin": float(margin.mean().item()),
    }


# Purpose: Predict top-k next words and expose sufficient information for the visual dashboard.
def predict_next_words(model: RNNLanguageModel, resources: dict[str, Any], text: str, top_k: int, temperature: float) -> dict[str, Any]:
    """Return top-k next-token candidates, probabilities, model uncertainty, and context tokens."""
    tokens = tokenize_seed(text)
    if not tokens:
        raise ValueError("Seed text không được rỗng.")

    top_k = max(1, min(int(top_k), 50))
    temperature = max(float(temperature), 0.05)
    context = tokens[-SEQ_LEN_DEFAULT:]
    token_to_id = resources["token_to_id"]
    id_to_token = resources["id_to_token"]

    ids = [token_to_id.get(token, token_to_id["<unk>"]) for token in context]
    input_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        logits, hidden = model(input_tensor)
        raw_logits = logits[:, -1, :]
        probs = torch.softmax(raw_logits / temperature, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    candidates = []
    cumulative = 0.0
    for rank, (prob, token_id) in enumerate(zip(top_probs[0], top_ids[0]), start=1):
        probability = float(prob.item())
        cumulative += probability
        candidates.append(
            {
                "rank": rank,
                "word": id_to_token[int(token_id.item())],
                "probability": probability,
                "percent": probability * 100.0,
                "log_probability": float(math.log(max(probability, 1e-12))),
                "cumulative_probability": cumulative,
            }
        )

    stats = probability_statistics(probs)
    return {
        "input_text": text,
        "context_tokens": context,
        "context_length": len(context),
        "unknown_context_tokens": sum(token not in token_to_id for token in context),
        "candidates": candidates,
        "entropy": stats["entropy"],
        "top1_probability": stats["top1_probability"],
        "top5_cumulative_probability": stats["top5_cumulative_probability"],
        "top1_top2_margin": stats["top1_top2_margin"],
        "temperature": temperature,
        "inference_ms": elapsed_ms,
        "hidden_state_norm": float(hidden.norm().item()),
    }


# Purpose: Generate a continuation one word at a time using autoregressive sampling.
def generate_text(
    model: RNNLanguageModel,
    resources: dict[str, Any],
    seed_text: str,
    num_words: int,
    temperature: float,
    top_k: int,
    mode: str,
) -> dict[str, Any]:
    """Generate continuation and return token-level generation statistics."""
    generated = tokenize_seed(seed_text)
    if not generated:
        raise ValueError("Seed text không được rỗng.")

    num_words = max(1, min(int(num_words), 100))
    temperature = max(float(temperature), 0.05)
    top_k = max(1, min(int(top_k), 100))
    token_to_id = resources["token_to_id"]
    id_to_token = resources["id_to_token"]
    step_stats: list[dict[str, Any]] = []

    start = time.perf_counter()
    for step in range(num_words):
        context = generated[-SEQ_LEN_DEFAULT:]
        ids = [token_to_id.get(token, token_to_id["<unk>"]) for token in context]
        input_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            logits, hidden = model(input_tensor)
            next_logits = logits[:, -1, :] / temperature

            if mode == "greedy":
                probs = torch.softmax(next_logits, dim=-1)
                next_id = int(torch.argmax(probs, dim=-1).item())
            else:
                # Restrict sampling to top-k vocabulary items for a more controllable demo.
                k = min(top_k, next_logits.shape[-1])
                values, indices = torch.topk(next_logits, k=k, dim=-1)
                filtered_probs = torch.softmax(values, dim=-1)
                selected = torch.multinomial(filtered_probs, num_samples=1)
                next_id = int(indices[0, selected.item()].item())
                probs = torch.softmax(next_logits, dim=-1)

            next_word = id_to_token[next_id]
            next_prob = float(probs[0, next_id].item())
            entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())

        generated.append(next_word)
        step_stats.append(
            {
                "step": step + 1,
                "word": next_word,
                "probability": next_prob,
                "entropy": entropy,
                "context_length": len(context),
                "hidden_state_norm": float(hidden.norm().item()),
            }
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "text": " ".join(generated),
        "generated_words": num_words,
        "temperature": temperature,
        "top_k": top_k,
        "mode": mode,
        "generation_ms": elapsed_ms,
        "words_per_second": (num_words / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
        "steps": step_stats,
    }


# Purpose: Load optional metrics exported from the notebook without inventing unavailable training results.
def load_optional_metrics() -> dict[str, Any]:
    """Load training history and final test metrics when corresponding JSON files exist."""
    result: dict[str, Any] = {}
    mapping = {
        "history": METRICS_DIR / "training_history.json",
        "test": METRICS_DIR / "test_metrics.json",
        "metadata": METRICS_DIR / "model_metadata.json",
    }
    for key, path in mapping.items():
        if path.exists():
            try:
                result[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result[key] = None
        else:
            result[key] = None
    return result


# Purpose: Build all dashboard data once at application startup and reuse it for fast page rendering.
def create_dashboard_payload(model: RNNLanguageModel, model_meta: dict[str, Any], resources: dict[str, Any] | None) -> dict[str, Any]:
    """Construct model metadata even when dataset resources are unavailable."""
    optional = load_optional_metrics()
    params = parameter_breakdown(model)
    total_params = sum(params.values())

    metadata = {
        "checkpoint": MODEL_PATH.name,
        "checkpoint_epoch": model_meta.get("epoch"),
        "best_validation_loss": model_meta.get("val_loss"),
        "best_validation_perplexity": (
            loss_to_perplexity(model_meta["val_loss"])
            if model_meta.get("val_loss") is not None else None
        ),
        "vocab_size": model_meta["vocab_size"],
        "embedding_dim": model_meta["embed_dim"],
        "hidden_size": model_meta["hidden_size"],
        "num_layers": model_meta["num_layers"],
        "sequence_length": SEQ_LEN_DEFAULT,
        "device": str(DEVICE),
        "total_parameters": total_params,
        "parameter_breakdown": params,
        "dataset": resources["split_stats"] if resources else {},
        "train_token_limit": resources["train_limit"] if resources else None,
        "top_tokens": ([
            {"token": token, "count": int(count)}
            for token, count in resources["train_counts"].most_common(20)
        ] if resources else []),
        "optional_metrics": optional,
    }
    return metadata


# Purpose: Initialize the model/vocabulary state used by inference endpoints.
MODEL = None
MODEL_METADATA = {}
RESOURCES = None
DASHBOARD = {}
STARTUP_ERROR = None
RESOURCE_ERROR = None

# Load the checkpoint independently from the dataset so model architecture information
# can still be displayed when the local WikiText files have not been prepared yet.
try:
    MODEL, MODEL_METADATA = load_model_and_checkpoint()
except Exception as exc:  # pragma: no cover
    STARTUP_ERROR = str(exc)

# Load vocabulary and dataset statistics separately because inference needs them,
# while the rest of the dashboard can operate without them.
if MODEL is not None:
    try:
        RESOURCES = build_runtime_resources()
    except Exception as exc:  # pragma: no cover
        RESOURCE_ERROR = str(exc)

if MODEL is not None:
    DASHBOARD = create_dashboard_payload(MODEL, MODEL_METADATA, RESOURCES)
else:
    DASHBOARD = {}

DASHBOARD["paths"] = {
    "model": str(MODEL_PATH),
    "model_exists": MODEL_PATH.exists(),
    "train_file": str(TRAIN_FILE),
    "train_exists": TRAIN_FILE.exists(),
    "valid_file": str(VALID_FILE),
    "valid_exists": VALID_FILE.exists(),
    "test_file": str(TEST_FILE),
    "test_exists": TEST_FILE.exists(),
}
DASHBOARD["resource_error"] = RESOURCE_ERROR


# Purpose: Render the interactive dashboard shell.
@app.route("/")
def index():
    """Render the single-page RNN demo interface."""
    return render_template("index.html", dashboard=DASHBOARD, startup_error=STARTUP_ERROR)


# Purpose: Provide all static dashboard analytics through a JSON API.
@app.get("/api/dashboard")
def api_dashboard():
    """Return model, dataset, parameter, and optional training statistics."""
    if STARTUP_ERROR:
        return jsonify({"error": STARTUP_ERROR}), 500
    return jsonify(DASHBOARD)


# Purpose: Predict the top-k next words from user-provided context.
@app.post("/api/predict")
def api_predict():
    """Handle next-word inference and return visualization-ready prediction data."""
    if STARTUP_ERROR:
        return jsonify({"error": STARTUP_ERROR}), 500
    if RESOURCES is None:
        return jsonify({"error": RESOURCE_ERROR or "Dataset resources are not available."}), 503
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_next_words(
            MODEL,
            RESOURCES,
            payload.get("text", ""),
            payload.get("top_k", 10),
            payload.get("temperature", 1.0),
        )
        return jsonify(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Inference failed: {exc}"}), 500


# Purpose: Generate autoregressive text for the presentation demo.
@app.post("/api/generate")
def api_generate():
    """Generate continuation and return per-step uncertainty statistics."""
    if STARTUP_ERROR:
        return jsonify({"error": STARTUP_ERROR}), 500
    if RESOURCES is None:
        return jsonify({"error": RESOURCE_ERROR or "Dataset resources are not available."}), 503
    payload = request.get_json(silent=True) or {}
    try:
        result = generate_text(
            MODEL,
            RESOURCES,
            payload.get("seed_text", ""),
            payload.get("num_words", 30),
            payload.get("temperature", 0.8),
            payload.get("top_k", 20),
            payload.get("mode", "sampling"),
        )
        return jsonify(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Generation failed: {exc}"}), 500


# Purpose: Expose a lightweight diagnostic endpoint for the web UI and troubleshooting.
@app.get("/api/health")
def api_health():
    """Return model/data availability and resolved paths without running inference."""
    return jsonify({
        "ok": STARTUP_ERROR is None,
        "startup_error": STARTUP_ERROR,
        "model_exists": MODEL_PATH.exists(),
        "train_exists": TRAIN_FILE.exists(),
        "valid_exists": VALID_FILE.exists(),
        "test_exists": TEST_FILE.exists(),
        "model_path": str(MODEL_PATH),
        "train_file": str(TRAIN_FILE),
        "device": str(DEVICE),
    })


# Purpose: Provide model metadata and parameter breakdown for the model-inspection panel.
@app.get("/api/model")
def api_model():
    """Return model architecture, parameter counts, and checkpoint information."""
    if STARTUP_ERROR:
        return jsonify({"error": STARTUP_ERROR}), 500
    return jsonify({
        "architecture": MODEL_METADATA,
        "parameters": DASHBOARD["parameter_breakdown"],
        "total_parameters": DASHBOARD["total_parameters"],
    })


# Purpose: Start the Flask development server when app.py is executed directly.
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
