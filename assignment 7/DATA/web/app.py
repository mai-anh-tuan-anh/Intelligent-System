"""A7 local dashboard. Start with: python app.py

This app reads artifacts exported by A7_RNN_LSTM_5Layers.ipynb. It never trains models.
Place app.py and the static/ directory beside outputs/, models/, and data/ in the A7 project root.
"""
from __future__ import annotations
import csv, importlib.util, io, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

APP_DIR = Path(__file__).resolve().parent
ROOT = next((folder for folder in [APP_DIR, *APP_DIR.parents] if (folder / "outputs").is_dir() and (folder / "models").is_dir()), APP_DIR)
STATIC = APP_DIR / "static"
HOST, PORT = "127.0.0.1", 8000
_LOCK = threading.RLock()
_BUNDLES = {}
_INFERENCE_MODULE = None

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

def project_state():
    """Read the notebook's latest-run pointer and exported JSON manifests."""
    latest = read_json(ROOT / "outputs" / "latest_run.json")
    if not latest:
        return {"available": False, "message": "No notebook export found yet.", "models": [], "dashboard": None}
    dash_path = ROOT / latest.get("dashboard", "") if latest.get("dashboard") else None
    if dash_path is None:
        dash_path = ROOT / "outputs" / latest.get("run_id", "") / "dashboard.json"
    dashboard = read_json(dash_path) if dash_path else None
    models_path = ROOT / latest.get("models", "") if latest.get("models") else ROOT / "models" / latest.get("run_id", "")
    registry = read_json(models_path / "model_registry.json") if models_path else None
    if not dashboard:
        return {"available": False, "message": "The latest run has no dashboard.json yet. Finish notebook export cell G1.", "models": [], "dashboard": None}
    models = [{"key": key, **entry} for key, entry in (registry or {}).items()]
    return {"available": True, "message": "Notebook results loaded.",
        "run_id": latest.get("run_id", dashboard.get("run_id", "unknown")),
        "dashboard": dashboard, "dashboard_path": str(dash_path), "model_dir": str(models_path),
        "registry": registry or {}, "models": models,
        "status": dashboard.get("status", latest.get("status", "unknown"))}

def inference_module(model_dir: Path):
    """Import the notebook-exported preprocessing and inference functions once."""
    global _INFERENCE_MODULE
    if _INFERENCE_MODULE is not None:
        return _INFERENCE_MODULE
    module_path = model_dir / "a7_inference.py"
    if not module_path.exists():
        raise FileNotFoundError("a7_inference.py is missing. Rerun notebook export cell G1.")
    spec = importlib.util.spec_from_file_location("a7_inference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the exported inference module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _INFERENCE_MODULE = module
    return module

def load_bundle(state, key):
    """Load only the requested checkpoint; do not load all eight models into memory."""
    if key not in state["registry"]:
        raise ValueError("Unknown model. Refresh the page and select a model from this run.")
    with _LOCK:
        if key not in _BUNDLES:
            module = inference_module(Path(state["model_dir"]))
            _BUNDLES[key] = module.load_bundle(Path(state["model_dir"]), key)
        return _BUNDLES[key]

class Handler(BaseHTTPRequestHandler):
    server_version = "A7Demo/1.0"
    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))
    def send_bytes(self, payload: bytes, content_type="application/json; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)
    def send_json(self, data, status=200):
        self.send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"), status=status)
    def read_body(self):
        size = int(self.headers.get("Content-Length", "0"))
        if size > 15_000_000:
            raise ValueError("Request too large (limit: 15 MB).")
        raw = self.rfile.read(size)
        return json.loads(raw.decode("utf-8")) if raw else {}
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            path = STATIC / "index.html"
            if not path.exists():
                return self.send_bytes(b"Missing static/index.html", "text/plain", 500)
            return self.send_bytes(path.read_bytes(), "text/html; charset=utf-8")
        if parsed.path == "/api/state":
            state = project_state()
            safe = {k: state[k] for k in ["available", "message", "run_id", "status", "models"] if k in state}
            if state.get("available"):
                safe["dashboard"] = state["dashboard"]
            return self.send_json(safe)
        if parsed.path == "/api/detail":
            return self.detail(parse_qs(parsed.query).get("key", [""])[0])
        if parsed.path == "/api/health":
            return self.send_json({"ok": True, "app": "A7 RNN/LSTM demo"})
        return self.send_json({"error": "Not found"}, 404)
    def detail(self, key):
        state = project_state()
        if not state.get("available"):
            return self.send_json({"error": state.get("message", "Notebook export is unavailable.")}, 409)
        if key not in state["registry"]:
            return self.send_json({"error": "Unknown model key."}, 400)
        record = next((r for r in state["dashboard"].get("experiments", []) if r.get("key") == key), None)
        history = state["dashboard"].get("histories", {}).get(key, [])
        result = {"record": record, "history": history, "trace": state["dashboard"].get("traces", {}).get(key)}
        if record and record.get("task") == "text":
            csv_path = ROOT / record.get("predictions_csv", "")
            rows = []
            if csv_path.exists():
                with csv_path.open(encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        rows.append({"true_label": row.get("true_label"), "pred_label": row.get("pred_label"),
                            "confidence": float(row.get("confidence", 0) or 0),
                            "p_negative": float(row.get("p_negative", 0) or 0),
                            "p_neutral": float(row.get("p_neutral", 0) or 0),
                            "p_positive": float(row.get("p_positive", 0) or 0)})
            result["text_test_rows"] = rows
        return self.send_json(result)
    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self.read_body()
            state = project_state()
            if not state.get("available"):
                return self.send_json({"error": state.get("message", "Run the notebook and export its artifacts first.")}, 409)
            if parsed.path == "/api/predict/review":
                key = body.get("key", "")
                if state["registry"].get(key, {}).get("task") != "text":
                    return self.send_json({"error": "Choose a sentiment model."}, 400)
                module = inference_module(Path(state["model_dir"]))
                prediction = module.predict_review(load_bundle(state, key), str(body.get("text", "")))
                return self.send_json(prediction)
            if parsed.path == "/api/predict/stock":
                key = body.get("key", "")
                if state["registry"].get(key, {}).get("task") != "stock":
                    return self.send_json({"error": "Choose a stock model."}, 400)
                csv_text = str(body.get("csv", ""))
                if not csv_text.strip():
                    return self.send_json({"error": "Choose a CSV file with Date, Open, High, Low, Close, and Volume columns."}, 400)
                import pandas as pd
                frame = pd.read_csv(io.StringIO(csv_text))
                module = inference_module(Path(state["model_dir"]))
                prediction = module.predict_next_close(load_bundle(state, key), frame)
                return self.send_json(prediction)
            return self.send_json({"error": "Not found"}, 404)
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

def main():
    state = project_state()
    print("\nA7 RNN/LSTM demo")
    print(f"Project folder: {ROOT}")
    if state.get("available"):
        print(f"Loaded run: {state.get('run_id')} | models: {len(state.get('models', []))}")
    else:
        print("Notebook exports not found yet. The dashboard will show setup instructions.")
    print(f"Open http://{HOST}:{PORT}  (Ctrl+C to stop)\n")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping A7 demo...")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
