"""Import the saved test results from the prior assignments without retraining."""
import json

from core.model import ARTIFACTS, ROOT

A4 = ROOT.parent / "assignment 4/DATA/outputs/metrics"
PIMA = ROOT.parent / "diabetes_multi_agent/results/pima-3db3f0cb3e6d/metrics.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    runs = []
    for slug, name, rows, source in (
        ("mnist", "MNIST", 70000, "https://www.tensorflow.org/datasets/catalog/mnist"),
        ("cifar10", "CIFAR-10", 60000, "https://cave.cs.toronto.edu/kriz/cifar.html"),
    ):
        for framework in ("scratch", "pytorch", "keras"):
            path = A4 / f"{slug}_{framework}_metrics.json"
            result = read(path)
            runs.append({"dataset": name, "rows": rows, "test": 10000,
                         "model": "CNN · " + {"scratch": "NumPy", "pytorch": "PyTorch", "keras": "Keras"}[framework],
                         "accuracy": result["accuracy"], "metric_name": "Macro-F1",
                         "metric": result["f1_macro"], "origin": path.relative_to(ROOT.parent).as_posix(),
                         "source": source})
    for name, result in read(PIMA)["models"].items():
        runs.append({"dataset": "Pima", "rows": 768, "test": sum(map(sum, result["confusion_matrix"])),
                     "model": name, "accuracy": result["accuracy"], "metric_name": "ROC-AUC",
                     "metric": result["roc_auc"], "origin": PIMA.relative_to(ROOT.parent).as_posix()})
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "assignment_benchmarks.json").write_text(
        json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

