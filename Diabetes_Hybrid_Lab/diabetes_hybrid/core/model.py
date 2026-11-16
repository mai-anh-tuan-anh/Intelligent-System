"""Three-class BRFSS model shared by training and inference."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FEATURES = {
    "HighBP": (0, 1, 0), "HighChol": (0, 1, 0), "CholCheck": (0, 1, 1),
    "BMI": (12, 98, 27), "Smoker": (0, 1, 0), "Stroke": (0, 1, 0),
    "HeartDiseaseorAttack": (0, 1, 0), "PhysActivity": (0, 1, 1),
    "Fruits": (0, 1, 1), "Veggies": (0, 1, 1), "HvyAlcoholConsump": (0, 1, 0),
    "AnyHealthcare": (0, 1, 1), "NoDocbcCost": (0, 1, 0), "GenHlth": (1, 5, 2),
    "MentHlth": (0, 30, 0), "PhysHlth": (0, 30, 0), "DiffWalk": (0, 1, 0),
    "Sex": (0, 1, 0), "Age": (1, 13, 8), "Education": (1, 6, 5), "Income": (1, 8, 6),
}
LABELS = ("Không báo cáo tiểu đường", "Báo cáo tiền tiểu đường", "Báo cáo tiểu đường")


def make_model():
    return nn.Sequential(
        nn.Linear(len(FEATURES), 64), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(LABELS)),
    )


class Assignment4MLP(nn.Module):
    """The original saved Assignment 4 architecture and state-dict layout."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURES), 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(LABELS)),
        )

    def forward(self, values):
        return self.network(values)


class Predictor:
    def __init__(self, checkpoint=ARTIFACTS / "mlp.pt"):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["features"] != list(FEATURES):
            raise ValueError("Sai thứ tự biến đầu vào của mô hình")
        self.original = saved.get("architecture") == "assignment4"
        self.model = Assignment4MLP() if self.original else make_model()
        self.model.load_state_dict(saved["weights"])
        self.model.eval()
        self.version = saved.get("version", "brfss-mlp-v1")
        dtype = np.float64 if self.original else np.float32
        self.mean = np.array(saved["mean"], dtype=dtype)
        self.scale = np.array(saved["scale"], dtype=dtype)

    @torch.inference_mode()
    def predict(self, values):
        inputs = np.array(values, dtype=np.float32)
        inputs -= self.mean
        inputs /= self.scale
        scores = self.model(torch.from_numpy(inputs)).softmax(dim=0).tolist()
        return {"class_id": int(np.argmax(scores)), "scores": scores, "model_version": self.version}

