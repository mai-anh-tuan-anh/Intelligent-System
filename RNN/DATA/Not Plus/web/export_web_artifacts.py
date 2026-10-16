from __future__ import annotations

import json
from pathlib import Path


# Purpose: Export notebook variables into JSON files that the Flask dashboard can read.
def export_training_history(history: dict, output_path: Path) -> None:
    """Save training history in JSON format for the web dashboard."""
    clean = {key: value for key, value in history.items() if isinstance(value, (list, int, float))}
    output_path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


# Purpose: Save final test metrics so the dashboard can display objective evaluation results.
def export_test_metrics(test_metrics: dict, output_path: Path) -> None:
    """Save final language-model test metrics as JSON."""
    output_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")


# Purpose: Provide the exact notebook cell to paste after training/evaluation.
def print_notebook_cell() -> None:
    """Print the notebook-side export cell used to connect training results to the web demo."""
    print(r'''
# Mục đích: Export training history và test metrics để Flask web demo hiển thị đầy đủ biểu đồ.
from pathlib import Path
import json

WEB_METRICS_DIR = Path(METRICS_DIR)
WEB_METRICS_DIR.mkdir(parents=True, exist_ok=True)

with open(WEB_METRICS_DIR / "training_history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

with open(WEB_METRICS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2)

print("Web demo artifacts exported to:", WEB_METRICS_DIR)
''')


# Purpose: Run a self-contained CLI helper that prints the integration cell for the notebook.
if __name__ == "__main__":
    print_notebook_cell()
