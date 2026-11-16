"""Two independent tabular MLP benchmarks. Neither model serves web requests."""
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from core.model import ARTIFACTS, ROOT

SEED = 42
A2 = ROOT / "data/assignment2_diabetes.csv"
if not A2.is_file():
    A2 = ROOT.parent / "assignment 2/DATA/diabetes/data/diabetes_dataset.csv"
UCI = ROOT / "data/uci-diabetes-hospitals.zip"
if not UCI.is_file():
    UCI = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "DiabetesHybridRuntime/uci-diabetes-hospitals.zip"
A2_NUM = ["age", "physical_activity_minutes_per_week", "bmi", "family_history_diabetes",
          "hypertension_history", "cardiovascular_history", "sleep_hours_per_day", "waist_to_hip_ratio"]
A2_CAT = ["gender", "smoking_status"]
UCI_NUM = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
UCI_CAT = ["age", "admission_type_id", "admission_source_id"]


def metrics(y, scores):
    prediction = (scores >= .5).astype(int)
    return {"accuracy": accuracy_score(y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "macro_f1": f1_score(y, prediction, average="macro"),
            "positive_recall": recall_score(y, prediction),
            "roc_auc": roc_auc_score(y, scores),
            "average_precision": average_precision_score(y, scores),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def run(name, source, path, x, y, groups, num, cat):
    torch.manual_seed(SEED)
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED).split(x, y, groups))
    train_rel, val_rel = next(GroupShuffleSplit(n_splits=1, test_size=3/17, random_state=SEED).split(
        x.iloc[train_val], y[train_val], groups[train_val]))
    train, val = train_val[train_rel], train_val[val_rel]
    assert not set(groups[train]) & set(groups[val])
    assert not set(groups[train]) & set(groups[test])
    assert not set(groups[val]) & set(groups[test])

    transformer = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
    ]).fit(x.iloc[train])
    xt, xv, xe = [torch.from_numpy(transformer.transform(x.iloc[i]).astype("float32")) for i in (train, val, test)]
    yt = torch.from_numpy(y[train])
    network = nn.Sequential(nn.Linear(xt.shape[1], 32), nn.ReLU(), nn.Dropout(.15),
                            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001)
    weights = len(train) / (2 * np.bincount(y[train]))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    best = -1
    for epoch in range(1, 13):
        network.train()
        for idx in torch.randperm(len(train)).split(512):
            optimizer.zero_grad()
            loss_fn(network(xt[idx]), yt[idx]).backward()
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            score = f1_score(y[val], network(xv).argmax(dim=1).numpy(), average="macro")
        if score > best:
            best, best_epoch, checkpoint = score, epoch, deepcopy(network.state_dict())
    network.load_state_dict(checkpoint)
    with torch.inference_mode():
        scores = network(xe).softmax(dim=1)[:, 1].numpy()
    logistic = LogisticRegression(class_weight="balanced", max_iter=400, random_state=SEED).fit(xt.numpy(), y[train])
    out = {"dataset": name, "source": source, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "rows": len(x), "features": num + cat, "encoded_dimension": xt.shape[1], "best_epoch": best_epoch,
           "validation_macro_f1": best, "seed": SEED,
           "splits": {part: {"rows": len(idx), "positive": int(y[idx].sum())}
                      for part, idx in (("train", train), ("validation", val), ("test", test))},
           "test": {"MLP": metrics(y[test], scores),
                    "Logistic Regression": metrics(y[test], logistic.predict_proba(xe.numpy())[:, 1]),
                    "Majority": metrics(y[test], np.full(len(test), y[train].mean()))}}
    print(name, out["test"], flush=True)
    return out


def main():
    torch.set_num_threads(2)
    a2 = pd.read_csv(A2)
    a2_x = a2[A2_NUM + A2_CAT]
    a2_result = run("Diabetes A2", "https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset",
                    A2, a2_x, a2["diagnosed_diabetes"].astype("int64").to_numpy(),
                    pd.util.hash_pandas_object(a2_x, index=False).to_numpy(), A2_NUM, A2_CAT)
    with ZipFile(UCI) as archive:
        with archive.open("diabetic_data.csv") as csv:
            hospital = pd.read_csv(csv, usecols=UCI_NUM + UCI_CAT + ["readmitted", "patient_nbr"])
    hospital_result = run("UCI 130-US Hospitals", "https://doi.org/10.24432/C5230J", UCI,
                          hospital[UCI_NUM + UCI_CAT], (hospital["readmitted"] == "<30").astype("int64").to_numpy(),
                          hospital["patient_nbr"].to_numpy(), UCI_NUM, UCI_CAT)
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "additional_experiments.json").write_text(
        json.dumps([a2_result, hospital_result], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

