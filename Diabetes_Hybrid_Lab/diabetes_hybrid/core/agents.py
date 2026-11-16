"""Deterministic agent boundaries for the educational web demo."""
import re
from dataclasses import asdict, dataclass
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

from neo4j.exceptions import DriverError, Neo4jError

from core.localization import localize_result, translate
from core.model import FEATURES, LABELS
from core.rag import CONTEXT_TOPICS, normalize_question, phrase_in

VERSION = "hybrid-3"
DEFAULT_MODEL = "assignment4"
MODEL_OPTIONS = {"assignment4": "MLP gốc từ Assignment 4", "grouped": "MLP kiểm tra theo nhóm hồ sơ"}
SCORE_MARGIN = .10
SCOPE = "Thông tin học thuật. Các điểm số biểu diễn nhãn khảo sát, không dùng để chẩn đoán hoặc điều trị."
MEDICATION = ("liều dùng", "liều thuốc", "liều insulin", "kê đơn", "đơn thuốc", "tăng thuốc",
              "giảm thuốc", "ngừng thuốc", "ngưng thuốc", "bao nhiêu đơn vị", "prescribe", "dosage",
              "how much insulin", "how many units", "insulin dose", "change my dose", "stop taking")
BLOCKED = (
    "tiêm insulin", "uống thuốc", "phác đồ", "metformin",
    "tôi có", "tôi bị", "tôi mắc", "tôi nên", "của tôi", "em bị", "em có",
    "mình bị", "mình có", "của mình", "của em", "mẹ tôi", "bố tôi", "con tôi", "mẹ em", "bố em", "con em",
    "bệnh nhân này", "trường hợp này", "system prompt", "bỏ qua hướng dẫn",
    "my glucose", "do i have", "my a1c", "chẩn đoán giúp", "xem kết quả giúp",
    "có bị tiểu đường không", "có mắc tiểu đường không", "có bị đái tháo đường không",
    "tôi đang mang thai", "em đang mang thai", "mình đang mang thai",
    "could i have", "am i diabetic", "diagnose me", "my blood sugar", "my hba1c", "am i pregnant",
)
URGENT = ("bất tỉnh", "hôn mê", "co giật", "mất ý thức", "khó thở", "thở gấp",
          "thở nhanh sâu", "nôn liên tục", "không giữ được nước", "không đánh thức được",
          "unconscious", "seizure", "trouble breathing", "cannot keep liquids down")
OFF_TOPIC = ("thời tiết", "bóng đá", "giá vàng", "bitcoin", "giá nhà", "chứng khoán",
             "weather", "football", "stock market", "house prices")
PUBLISHERS = {"www.niddk.nih.gov", "www.cdc.gov", "www.vinmec.com", "www.kaggle.com",
              "www.who.int", "nhathuoclongchau.com.vn"}
ROUTING_GUIDANCE = {
    "urgent": ("Nếu dấu hiệu này đang xảy ra, hãy gọi cấp cứu địa phương hoặc đến khoa cấp cứu ngay.",
               "Ưu tiên cấp cứu. Đừng chờ chatbot đánh giá thêm."),
    "medication": ("Việc bắt đầu, ngừng hoặc đổi liều thuốc cần dựa trên đơn và tình trạng của từng người.",
                   "Liên hệ bác sĩ kê đơn hoặc dược sĩ phụ trách. Chuẩn bị tên thuốc và đơn hiện tại để trao đổi."),
    "blocked": ("Triệu chứng và một kết quả xét nghiệm riêng lẻ chưa đủ để xác định bạn có mắc đái tháo đường hay không.",
                "Trao đổi với cơ sở y tế. Mang theo phiếu xét nghiệm, thời điểm lấy mẫu, triệu chứng và danh sách thuốc nếu có."),
}
PRESETS = {
    "sample_a": ("Hồ sơ A", {}),
    "sample_b": ("Hồ sơ B", {"HighBP": 1, "BMI": 33, "Age": 10, "GenHlth": 4, "PhysActivity": 0}),
}


@dataclass(frozen=True)
class Message:
    request_id: str
    agent: str
    status: str
    payload: dict
    evidence: tuple[str, ...] = ()
    version: str = VERSION


class MonitoringAgent:
    """Publish summaries and selected metadata, separate from agent payloads."""

    def __init__(self):
        self.id = uuid4().hex[:12]
        self.events = []

    def record(self, message, started, summary, **public_fields):
        public = Message(message.request_id, message.agent, message.status,
                         {"summary": summary, **public_fields}, message.evidence, message.version)
        self.events.append({**asdict(public), "elapsed_ms": round((perf_counter() - started) * 1000, 2)})


class DataAgent:
    @classmethod
    def run(cls, request_id, profile, language="vi"):
        try:
            return Message(request_id, "Data Agent", "ok", {"features": cls.validate(profile, language)})
        except ValueError as exc:
            return Message(request_id, "Data Agent", "invalid", {"error": str(exc)})

    @staticmethod
    def validate(profile, language="vi"):
        if not isinstance(profile, dict) or profile.keys() != FEATURES.keys():
            raise ValueError("Cần đúng 21 biến khảo sát.")
        values = []
        for name, (lower, upper, _) in FEATURES.items():
            value = profile[name]
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(translate("{name} phải là số nguyên từ {lower} đến {upper}.", language,
                                           name=name, lower=lower, upper=upper))
            values.append(value)
        return values


class RuleAgent:
    @staticmethod
    def check(question):
        if not isinstance(question, str) or len(question) > 500 or not question.strip():
            return "invalid", "Hãy nhập câu hỏi từ 1 đến 500 ký tự."
        text = normalize_question(question)
        if not text:
            return "invalid", "Hãy nhập câu hỏi bằng chữ."
        if any(phrase_in(text, word) for word in URGENT):
            return "urgent", ROUTING_GUIDANCE["urgent"][0]
        without_topics = re.sub(
            r"\b(?:(?:tip|type|loai) [12]|(?:dai thao duong|tieu duong) [12]|"
            r"(?:brfss )?2015|diabetes_012|diabete3|(?:nhan|label|class) [012]|(?:hb)?a1c)\b", "", text)
        if re.search(r"\bliều\b", question.casefold()) or any(phrase_in(text, phrase) for phrase in MEDICATION):
            return "medication", ROUTING_GUIDANCE["medication"][0]
        if (any(phrase_in(text, phrase) for phrase in BLOCKED)
                or re.search(r"\b(?:toi|minh|em) (?:dang |hay )?(?:khat|tieu nhieu|sut can|non|run tay|met)\b", text)
                or any(char.isdigit() for char in without_topics)):
            return "blocked", ROUTING_GUIDANCE["blocked"][0]
        if any(phrase_in(text, phrase) for phrase in OFF_TOPIC):
            return "insufficient", "Câu hỏi nằm ngoài chủ đề của kho tri thức."
        return "ok", ""

    @classmethod
    def assess(cls, request_id, question):
        status, message = cls.check(question)
        payload = {"code": status, "message": message}
        if status in ROUTING_GUIDANCE:
            payload["next_step"] = ROUTING_GUIDANCE[status][1]
        if status == "urgent":
            payload["support_url"] = "https://www.cdc.gov/diabetes/about/diabetic-ketoacidosis.html"
        return Message(request_id, "Rule Agent", status, payload)

    @staticmethod
    def score(prediction):
        top, second = sorted(prediction.payload["scores"], reverse=True)[:2]
        margin = top - second
        status = "review_required" if margin < SCORE_MARGIN else "ok"
        return Message(prediction.request_id, "Rule Agent", status, {
            "code": "close_scores" if status == "review_required" else "score_margin_passed",
            "margin": margin, "threshold": SCORE_MARGIN,
            "message": "Hai điểm cao nhất chênh dưới 0,10 theo quy tắc demo." if status == "review_required" else "",
        })


class PredictionAgent:
    @staticmethod
    def run(data, predictor):
        if predictor is None:
            return Message(data.request_id, "DL Prediction Agent", "unavailable",
                           {"error": "Chưa có trọng số cho mô hình được chọn."})
        result = predictor.predict(data.payload["features"])
        return Message(data.request_id, "DL Prediction Agent", "ok",
                       {**result, "label": LABELS[result["class_id"]]}, version=result["model_version"])


class KnowledgeAgent:
    @staticmethod
    def run(question_rule, graph, question, context, language="vi"):
        try:
            prior = context if isinstance(context, str) and context in CONTEXT_TOPICS else None
            result = graph.answer(question, context=prior, language=language) if graph else {"status": "unavailable"}
        except (DriverError, Neo4jError, OSError):
            result = {"status": "unavailable"}
        if result["status"] == "unavailable":
            result["message"] = "Chưa kết nối được Neo4j. Hãy kiểm tra dịch vụ và cấu hình kết nối."
        ids = tuple(item["id"] for item in result.get("evidence", ()))
        return Message(question_rule.request_id, "KG/RAG Agent", result["status"], result, ids)


def evidence_conflicts(evidence):
    """Compare only curated, structured assertions with the same scope key."""
    claims = {}
    for item in evidence:
        for claim in item.get("claims", ()):
            scope, separator, value = claim.partition("=")
            if separator and scope and value:
                claims.setdefault(scope, []).append((value, item["id"], item["source"]["url"]))
    conflicts = []
    for scope, assertions in claims.items():
        opposing = {left_id for left_value, left_id, left_url in assertions
                    for right_value, _, right_url in assertions
                    if left_value != right_value and left_url != right_url}
        if opposing:
            conflicts.append({"scope": scope, "passage_ids": sorted(opposing)})
    return conflicts


class ExplanationAgent:
    @staticmethod
    def compose(result):
        if result["status"] != "ok":
            return result
        evidence = []
        for item in result["evidence"]:
            url = urlsplit(item["source"]["url"])
            if url.scheme == "https" and url.hostname in PUBLISHERS and item["text"].strip():
                evidence.append(item)
        if not evidence:
            return {"status": "insufficient", "message": "Chưa có nguồn đạt điều kiện để hiển thị."}
        conflicts = evidence_conflicts(evidence)
        if conflicts:
            ids = {item for conflict in conflicts for item in conflict["passage_ids"]}
            return {"status": "conflicting_evidence", "conflicts": conflicts,
                    "evidence": [item for item in evidence if item["id"] in ids],
                    "message": "Các nguồn đã truy xuất có nhận định khác nhau trong cùng phạm vi.",
                    "next_step": "Hãy nhờ người có chuyên môn đối chiếu các nguồn trước khi áp dụng."}
        return {**result, "answer": evidence[0]["text"], "evidence": evidence}

    @classmethod
    def report(cls, request_id, prediction, score_rule, question_rule, retrieval, error):
        model = None
        if prediction and prediction.status == "ok":
            model = {**prediction.payload, "flag": score_rule.payload["message"]}
        rules = {
            "question": ({"status": question_rule.status, **question_rule.payload} if question_rule
                         else {"status": "not_requested"}),
            "prediction": ({"status": score_rule.status, **score_rule.payload} if score_rule
                           else {"status": "not_available"}),
        }
        knowledge = None
        if retrieval:
            knowledge = cls.compose(retrieval.payload)
        elif question_rule and question_rule.status != "ok":
            knowledge = {"status": question_rule.status, **question_rule.payload}

        reasons = []
        if knowledge and knowledge["status"] in ("urgent", "blocked", "medication", "conflicting_evidence"):
            reasons.append({"code": knowledge["status"], "reason": knowledge["message"],
                            "next_step": knowledge["next_step"]})
        if score_rule and score_rule.status == "review_required":
            reasons.append({"code": "close_scores", "reason": score_rule.payload["message"],
                            "next_step": "Nếu đây là mối lo sức khỏe thật, hãy trao đổi với nhân viên y tế."})
        level = "urgent" if reasons and reasons[0]["code"] == "urgent" else "required" if reasons else "none"
        status = ("review_required" if reasons else "partial" if error and knowledge else
                  "error" if error else "ok")
        payload = {
            "status": status, "request_id": request_id, "error": error,
            "prediction": model, "rules": rules, "knowledge": knowledge,
            "review": {"status": level, "reasons": reasons},
            "limitations": [SCOPE, "Điểm softmax chưa hiệu chuẩn. Tài liệu trích dẫn chỉ dùng để tham khảo."],
        }
        ids = tuple(item["id"] for item in (knowledge or {}).get("evidence", ()))
        return Message(request_id, "Explanation Agent", status, payload, ids)


class Coordinator:
    def __init__(self, predictor, graph, original_predictor=None):
        self.predictors = {"grouped": predictor, "assignment4": original_predictor}
        self.graph = graph

    def run(self, mode, preset=None, question=None, custom_profile=None, context=None, model="grouped", language="vi"):
        monitor = MonitoringAgent()
        request_id = monitor.id
        started = perf_counter()
        monitor.record(Message(request_id, "Coordinator", "started", {}), started,
                       "Nhận yêu cầu và xác định nhánh thực thi.")
        error, prediction, score_rule, question_rule, retrieval = None, None, None, None, None
        has_model = mode in ("predict", "combined")
        if mode not in ("qa", "predict", "combined"):
            error = "Chọn một chế độ hợp lệ."
        elif has_model:
            stage = perf_counter()
            profile = custom_profile
            if profile is None and isinstance(preset, str) and preset in PRESETS:
                profile = {name: bounds[2] for name, bounds in FEATURES.items()} | PRESETS[preset][1]
            data = DataAgent.run(request_id, profile, language)
            error = data.payload.get("error")
            if error is None:
                monitor.record(data, stage, "Đã xác thực 21 biến của hồ sơ giả lập.",
                               feature_count=len(data.payload["features"]))
            else:
                monitor.record(data, stage, "Dữ liệu đầu vào không hợp lệ.")
        if error is None:
            if has_model:
                stage = perf_counter()
                prediction = PredictionAgent.run(data, self.predictors.get(model) if isinstance(model, str) else None)
                error = prediction.payload.get("error")
                monitor.record(prediction, stage, error or "MLP đã tính ba điểm cho nhãn BRFSS.")
                if prediction.status == "ok":
                    stage = perf_counter()
                    score_rule = RuleAgent.score(prediction)
                    monitor.record(score_rule, stage, "Kiểm tra khoảng cách giữa hai điểm cao nhất.",
                                   rule=score_rule.payload["code"])
            provided = question is not None and (not isinstance(question, str) or bool(question.strip()))
            if mode == "qa" or (mode == "combined" and provided):
                stage = perf_counter()
                question_rule = RuleAgent.assess(request_id, question)
                monitor.record(question_rule, stage, "Kiểm tra phạm vi của câu hỏi.", rule=question_rule.status)
                if question_rule.status == "ok":
                    stage = perf_counter()
                    retrieval = KnowledgeAgent.run(question_rule, self.graph, question, context, language)
                    monitor.record(retrieval, stage, "Tra khái niệm, đoạn tri thức và nguồn trong Neo4j.")
        stage = perf_counter()
        report = ExplanationAgent.report(request_id, prediction, score_rule, question_rule, retrieval, error)
        monitor.record(report, stage, "Tổng hợp dự đoán, quy tắc, bằng chứng và giới hạn.",
                       review_status=report.payload["review"]["status"])
        monitor.record(Message(request_id, "Monitoring Agent", "ok", {}), started,
                       "Đo thời gian và phiên bản của các bước thực thi.")
        return localize_result({
            "request_id": request_id, "version": VERSION, "scope": SCOPE, "error": error,
            "prediction": report.payload["prediction"], "knowledge": report.payload["knowledge"],
            "report": report.payload, "trace": monitor.events,
        }, language)

