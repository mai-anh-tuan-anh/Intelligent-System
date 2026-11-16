"""Package the locally supplied Assignment 4 weights and scaler without training."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from core.model import ARTIFACTS, FEATURES, ROOT, Assignment4MLP, Predictor

DATA_FILE = "diabetes_012_health_indicators_BRFSS2015.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(assignment):
    models = assignment / "models"
    weights_file = models / "diabetes_mlp_pytorch.pth"
    scaler_file = models / "diabetes_standard_scaler.joblib"
    metrics_file = assignment / "outputs/metrics/diabetes_pytorch_metrics.json"
    with (assignment / "data/diabetes" / DATA_FILE).open(encoding="utf-8-sig", newline="") as handle:
        features = [name for name in next(csv.reader(handle)) if name != "Diabetes_012"]
    if features != list(FEATURES):
        raise ValueError("Assignment 4 feature order differs from the current schema")
    weights = torch.load(weights_file, map_location="cpu", weights_only=True)
    # The scaler is the trusted, locally generated Assignment 4 artifact.
    scaler = joblib.load(scaler_file)
    if list(getattr(scaler, "feature_names_in_", features)) != features:
        raise ValueError("The saved scaler has a different feature order")
    original = Assignment4MLP()
    original.load_state_dict(weights)
    original.eval()
    version = "assignment4-original"
    destination = ARTIFACTS / "assignment4_original.pt"
    torch.save({"weights": weights, "features": features, "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(), "version": version, "architecture": "assignment4"},
               destination)
    predictor = Predictor(destination)
    first = [bounds[2] for bounds in FEATURES.values()]
    second = [bounds[1] for bounds in FEATURES.values()]
    for values in (first, second):
        inputs = scaler.transform(np.asarray([values], dtype=np.float32))
        with torch.inference_mode():
            expected = original(torch.from_numpy(inputs))[0].softmax(dim=0).numpy()
        np.testing.assert_allclose(predictor.predict(values)["scores"], expected, rtol=0, atol=1e-6)
    (ARTIFACTS / "assignment4_original.json").write_text(json.dumps({
        "version": version, "feature_order": features, "architecture": [21, 64, 32, 3],
        "source_weights_sha256": sha256(weights_file), "source_scaler_sha256": sha256(scaler_file),
        "saved_weights_match_original": all(torch.equal(weights[key], predictor.model.state_dict()[key])
                                            for key in weights),
        "inference_parity_profiles": 2,
        "inference_parity_atol": 1e-6,
        "archived_assignment_metrics": json.loads(metrics_file.read_text(encoding="utf-8")),
        "note": "Assignment 4 results used its own evaluation protocol, separate from the grouped holdout.",
    }, indent=2) + "\n", encoding="utf-8")
    print("Original checkpoint and scaler loaded; both parity checks passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        workspace = next((parent for parent in ROOT.parents if (parent / "assignment 4/DATA").is_dir()), None)
        if workspace is None:
            raise SystemExit("Supply the path to Assignment 4's DATA directory.")
        main(workspace / "assignment 4/DATA")

