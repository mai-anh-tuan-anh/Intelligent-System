"""Run the local browser demo with python app.py."""
import json
import os

from flask import Blueprint, Flask, abort, current_app, jsonify, render_template, request
from neo4j.exceptions import DriverError, Neo4jError

from core.agents import DEFAULT_MODEL, MODEL_OPTIONS, Coordinator, PRESETS
from core.localization import CLIENT_MESSAGES, translate
from core.model import ARTIFACTS, FEATURES, LABELS, Predictor
from core.rag import GraphRAG, connect

routes = Blueprint("demo", __name__)
try:
    graph = GraphRAG(connect())
except (ValueError, OSError):
    graph = None


def load_predictor(checkpoint):
    try:
        return Predictor(ARTIFACTS / checkpoint)
    except (OSError, RuntimeError, ValueError):
        return None


coordinator = Coordinator(load_predictor("mlp.pt"), graph,
                          load_predictor("assignment4_original.pt"))
RESULT_PATH = ARTIFACTS / "experiment.json"
experiment = json.loads(RESULT_PATH.read_text(encoding="utf-8")) if RESULT_PATH.exists() else None
BENCHMARK_PATH = ARTIFACTS / "assignment_benchmarks.json"
benchmarks = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8")) if BENCHMARK_PATH.exists() else []
ADDITIONAL_PATH = ARTIFACTS / "additional_experiments.json"
additional = json.loads(ADDITIONAL_PATH.read_text(encoding="utf-8")) if ADDITIONAL_PATH.exists() else []
DIAGNOSTICS_PATH = ARTIFACTS / "brfss_diagnostics.json"
diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8")) if DIAGNOSTICS_PATH.exists() else None
VERIFICATION_PATH = ARTIFACTS / "demo_verification.json"
verification = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8")) if VERIFICATION_PATH.exists() else None
ORIGINAL_PATH = ARTIFACTS / "assignment4_original.json"
original = json.loads(ORIGINAL_PATH.read_text(encoding="utf-8")) if ORIGINAL_PATH.exists() else None
SAMPLES = ("nguy cơ?", "Chẩn đoán tiểu đường type 1 bằng xét nghiệm nào?",
           "HbA1c phản ánh điều gì?", "Ngưỡng HbA1c được hiểu thế nào?",
           "Sàng lọc tiểu đường thai kỳ khi nào?", "Xét nghiệm HbA1c có cần nhịn ăn không?",
           "Dấu hiệu hạ đường huyết là gì?", "Theo dõi gì ngoài đường huyết?",
           "BRFSS 2015 và nhãn 2 có ý nghĩa gì?")


def graph_status():
    try:
        return graph.status() if graph else {"ready": False}
    except (DriverError, Neo4jError, OSError):
        return {"ready": False}


@routes.route("/", methods=["GET", "POST"])
def home():
    language = current_app.config["DEMO_LANGUAGE"]

    def t(message, **values):
        return translate(message, language, **values)

    def format_count(value):
        formatted = f"{value:,}"
        return formatted.replace(",", ".") if language == "vi" else formatted

    question = request.form.get("question", "")[:501]
    result = None
    model_choice = request.form.get("model", DEFAULT_MODEL)
    custom_profile = request.form.get("custom_profile") == "yes"
    raw_profile = {name: request.form.get("feature_" + name, "" if custom_profile else str(spec[2]))
                   for name, spec in FEATURES.items()}
    if request.method == "POST":
        profile = None
        if custom_profile:
            profile = {}
            for name, raw in raw_profile.items():
                try:
                    profile[name] = int(raw)
                except ValueError:
                    profile[name] = raw
        result = coordinator.run(request.form.get("mode"), request.form.get("preset"), question, profile,
                                 model=model_choice, language=language)
    return render_template("index.html", language=language, t=t, format_count=format_count,
                           client_messages={message: t(message) for message in CLIENT_MESSAGES} if language == "en" else {},
                           result=result, question=question, samples=[t(s) for s in SAMPLES],
                           presets={key: (t(info[0]), info[1]) for key, info in PRESETS.items()},
                           preset=request.form.get("preset", "sample_a"), labels=[t(label) for label in LABELS],
                           experiment=experiment, benchmarks=benchmarks, additional=additional, kg=graph_status(),
                           features=FEATURES, raw_profile=raw_profile, custom_profile=custom_profile,
                           diagnostics=diagnostics, verification=verification, original=original,
                           model_options={key: t(label) for key, label in MODEL_OPTIONS.items()}, model_choice=model_choice,
                           browser_url=os.getenv("NEO4J_BROWSER_URL", "http://127.0.0.1:7476/browser/"))


@routes.post("/api/run")
def api_run():
    language = current_app.config["DEMO_LANGUAGE"]
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error=translate("Yêu cầu cần là một đối tượng JSON.", language)), 400
    profile = payload.get("features")
    if profile is not None and payload.get("synthetic") is not True:
        return jsonify(error=translate("Chỉ chấp nhận hồ sơ giả lập được xác nhận.", language)), 400
    result = coordinator.run(payload.get("mode"), payload.get("preset"), payload.get("question"),
                             profile, context=payload.get("context"), model=payload.get("model", DEFAULT_MODEL),
                             language=language)
    return jsonify(result), 400 if result["error"] else 200


@routes.get("/api/graph")
def api_graph():
    status = graph_status()
    return jsonify(status), 200 if status["ready"] else 503


@routes.get("/api/experiment")
def api_experiment():
    if experiment is None:
        abort(503)
    return jsonify(brfss=experiment, additional=additional, assignment_benchmarks=benchmarks,
                   original_assignment4=original)


def create_app(language="vi"):
    demo = Flask(__name__)
    demo.config.update(MAX_CONTENT_LENGTH=16 * 1024, DEMO_LANGUAGE=language)
    demo.register_blueprint(routes)
    return demo


app = create_app()


def run(demo):
    if demo.config["DEMO_LANGUAGE"] == "en":
        port = os.getenv("DIABETES_EN_PORT", "8013")
    else:
        port = os.getenv("DIABETES_PORT", "8011")
    demo.run(host="127.0.0.1", port=int(port), debug=False)


if __name__ == "__main__":
    run(app)

