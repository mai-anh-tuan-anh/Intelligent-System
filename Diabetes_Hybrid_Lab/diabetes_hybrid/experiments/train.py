"""Reproduce the held-out experiment on the local Assignment 4 dataset."""
import hashlib
import json
import platform
from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

from core.model import ARTIFACTS, FEATURES, ROOT, make_model

DATA_FILE = "diabetes_012_health_indicators_BRFSS2015.csv"
DATA = ROOT / "data" / DATA_FILE
if not DATA.is_file():
    DATA = ROOT.parent / "assignment 4/DATA/data/diabetes" / DATA_FILE
SEED = 42
CLASSES = [0, 1, 2]


def evaluate(y, scores):
    prediction = scores.argmax(axis=1)
    targets = np.eye(len(CLASSES))[y]
    matrix = confusion_matrix(y, prediction, labels=CLASSES)
    report = classification_report(y, prediction, labels=CLASSES, output_dict=True, zero_division=0)
    for cls in CLASSES:
        negatives = matrix.sum() - matrix[cls, :].sum()
        false_positives = matrix[:, cls].sum() - matrix[cls, cls]
        report[str(cls)]["specificity_ovr"] = float(1 - false_positives / negatives)
        report[str(cls)]["average_precision_ovr"] = average_precision_score(targets[:, cls], scores[:, cls])
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro"),
        "roc_auc_ovr_macro": roc_auc_score(y, scores, multi_class="ovr", average="macro"),
        "average_precision_macro": average_precision_score(targets, scores, average="macro"),
        "log_loss": log_loss(y, scores, labels=CLASSES),
        "brier_multiclass": float(np.mean(np.sum((scores - targets) ** 2, axis=1))),
        "confusion_matrix": matrix.tolist(),
        "per_class": report,
    }


def downsample(first, second):
    keep = np.unique(np.linspace(0, len(first) - 1, 150, dtype=int))
    return {"x": first[keep].tolist(), "y": second[keep].tolist()}


def diagnostics(network, x, y, predictions):
    curves = {}
    positive = y == 2
    for name, scores in predictions.items():
        fpr, tpr, _ = roc_curve(positive, scores[:, 2])
        precision, recall, _ = precision_recall_curve(positive, scores[:, 2])
        observed, mean = calibration_curve(positive, scores[:, 2], n_bins=10)
        counts, edges = np.histogram(scores[:, 2], bins=20, range=(0, 1))
        curves[name] = {"roc": downsample(fpr, tpr), "pr": downsample(recall, precision),
                        "calibration": {"x": mean.tolist(), "y": observed.tolist()},
                        "score_histogram": {"edges": edges.tolist(), "counts": counts.tolist()}}
    rng = np.random.default_rng(SEED)
    subset = rng.choice(len(y), size=6000, replace=False)
    base = f1_score(y[subset], predictions["MLP"][subset].argmax(axis=1), average="macro")
    importance = []
    with torch.inference_mode():
        for col, feature in enumerate(FEATURES):
            drops = []
            for _ in range(3):
                shuffled = x[subset].copy()
                shuffled[:, col] = rng.permutation(shuffled[:, col])
                prediction = network(torch.from_numpy(shuffled)).argmax(dim=1).numpy()
                drops.append(base - f1_score(y[subset], prediction, average="macro"))
            importance.append({"feature": feature, "mean_drop": float(np.mean(drops)),
                               "std_drop": float(np.std(drops))})
    return {"positive_class": 2, "positive_prevalence": float(positive.mean()), "curves": curves,
            "permutation": importance, "permutation_rows": len(subset), "permutation_repeats": 3}


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    df = pd.read_csv(DATA)
    x, y = df[list(FEATURES)], df["Diabetes_012"].astype(int)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED).split(x, y, groups))
    train_rel, val_rel = next(GroupShuffleSplit(n_splits=1, test_size=3/17, random_state=SEED).split(
        x.iloc[train_val], y.iloc[train_val], groups[train_val]))
    train, val = train_val[train_rel], train_val[val_rel]
    assert not set(groups[train]) & set(groups[test])
    assert not set(groups[train]) & set(groups[val])
    assert not set(groups[val]) & set(groups[test])
    for feature, (lower, upper, _) in FEATURES.items():
        assert x[feature].between(lower, upper).all()
    assert set(y.unique()) == set(CLASSES)
    scaler = StandardScaler().fit(x.iloc[train])
    x_train, x_val, x_test = [scaler.transform(x.iloc[idx]).astype("float32") for idx in (train, val, test)]
    y_train, y_val, y_test = [y.iloc[idx].to_numpy() for idx in (train, val, test)]

    network = make_model()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.0001)
    counts = y.iloc[train].value_counts().sort_index().to_numpy()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (len(CLASSES) * counts), dtype=torch.float32))
    xt, yt, xv, yv = map(torch.from_numpy, (x_train, y_train, x_val, y_val))
    history = []
    best_f1, best_weights = -1, None
    patience = 8
    for epoch in range(1, 17):
        network.train()
        for idx in torch.randperm(len(xt)).split(1024):
            optimizer.zero_grad()
            loss = loss_fn(network(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            train_logits, val_logits = network(xt), network(xv)
            score = f1_score(y_val, val_logits.argmax(dim=1).numpy(), average="macro")
            history.append({"epoch": epoch, "train_weighted_loss": loss_fn(train_logits, yt).item(),
                            "val_weighted_loss": loss_fn(val_logits, yv).item(), "val_macro_f1": score,
                            "train_accuracy": (train_logits.argmax(dim=1) == yt).float().mean().item(),
                            "val_accuracy": (val_logits.argmax(dim=1) == yv).float().mean().item()})
        print(f"Epoch {epoch:02d}  validation macro-F1={score:.4f}", flush=True)
        if score > best_f1:
            best_f1, best_epoch, best_weights = score, epoch, deepcopy(network.state_dict())
        if epoch - best_epoch >= patience:
            break
    network.load_state_dict(best_weights)
    with torch.inference_mode():
        mlp_scores = network(torch.from_numpy(x_test)).softmax(dim=1).numpy()
    logreg = LogisticRegression(class_weight="balanced", max_iter=500, random_state=SEED).fit(x_train, y_train)
    dummy = DummyClassifier(strategy="prior").fit(x_train, y_train)
    predictions = {"Majority": dummy.predict_proba(x_test),
                   "Logistic Regression": logreg.predict_proba(x_test), "MLP": mlp_scores}
    results = {
        "dataset": DATA.name, "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "seed": SEED, "rows": len(df), "unique_feature_profiles": len(set(groups)),
        "features": list(FEATURES), "class_counts": {str(c): int((y == c).sum()) for c in CLASSES},
        "splits": {name: {"rows": len(idx), "feature_profiles": len(set(groups[idx])),
                         "class_counts": {str(c): int((y.iloc[idx] == c).sum()) for c in CLASSES}}
                   for name, idx in (("train", train), ("validation", val), ("test", test))},
        "best_epoch": best_epoch, "history": history,
        "test": {name: evaluate(y_test, scores) for name, scores in predictions.items()},
        "training_config": {"model_version": "brfss-mlp-v2", "layer_widths": [21, 64, 32, 3],
                            "batch_size": 1024, "dropout": 0.15, "learning_rate": 0.001,
                            "weight_decay": 0.0001, "early_stopping_patience": patience,
                            "maximum_epochs": 16, "decision_rule": "argmax"},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({"weights": best_weights, "features": list(FEATURES), "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(), "version": results["training_config"]["model_version"]},
               ARTIFACTS / "mlp.pt")
    (ARTIFACTS / "experiment.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    extra = diagnostics(network, x_test, y_test, predictions)
    (ARTIFACTS / "brfss_diagnostics.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

