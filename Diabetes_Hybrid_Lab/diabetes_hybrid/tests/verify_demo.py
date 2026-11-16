"""Integration checks against the running local Neo4j and Flask test client."""
import json
import math
from statistics import median
from time import perf_counter
from unittest.mock import patch

from core.agents import Coordinator, ExplanationAgent, RuleAgent
from app import app, graph
from core.model import ARTIFACTS
from knowledge_base.build_kb import build

CASES = [
    ("nguy cơ?", "ok", "p05"),
    ("nguy co", "ok", "p05"),
    ("Ai dễ mắc đái tháo đường típ 2?", "ok", "p05"),
    ("Tiền sử gia đình có liên quan đến type 2?", "ok", "p05"),
    ("Làm sao giảm nguy cơ mắc típ 2?", "ok", "p06"),
    ("Phòng bệnh type 2 thế nào?", "ok", "p06"),
    ("HBA1C phản ánh điều gì?", "ok", "p07"),
    ("Đường huyết trung bình là xét nghiệm gì?", "ok", "p07"),
    ("Xét nghiệm glucose huyết tương lúc đói", "ok", "p08"),
    ("Chẩn đoán tiểu đường type 1 bằng xét nghiệm nào?", "ok", "p02"),
    ("Tự kháng thể dùng làm gì trong tiểu đường típ 1?", "ok", "p03"),
    ("Tiểu đường type 1 là gì?", "ok", "p01"),
    ("Tiểu đường type 2 là gì?", "ok", "p04"),
    ("Phân biệt tiểu đường type 1 và 2", "ok", "p04"),
    ("Khát nước có phải dấu hiệu chung của tiểu đường?", "ok", "p09"),
    ("Tiểu đường có biến chứng gì?", "ok", "p10"),
    ("Tiền đái tháo đường là gì?", "ok", "p11"),
    ("BRFSS 2015 đo gì?", "ok", "p12"),
    ("Tài liệu CDC về BRFSS", "ok", "p12"),
    ("Sàng lọc tiểu đường thai kỳ khi nào?", "ok", "p18"),
    ("Tiểu đường thai kỳ là gì?", "ok", "p27"),
    ("Tiểu đường thai kỳ có dấu hiệu rõ không?", "ok", "p19"),
    ("Xét nghiệm HbA1c có cần nhịn ăn không?", "ok", "p20"),
    ("Ngưỡng HbA1c được hiểu thế nào?", "ok", "p24"),
    ("Hạ đường huyết có dấu hiệu gì?", "ok", "p21"),
    ("Nhiễm toan ceton là gì?", "ok", "p22"),
    ("Biến chứng bàn chân và loét chân", "ok", "p23"),
    ("Theo dõi ngoài đường huyết gồm những gì?", "ok", "p25"),
    ("Nhãn 2 trong Diabetes_012 là gì?", "ok", "p13"),
    ("Tiểu đường là bệnh gì?", "ok", "p14"),
    ("Chiều nay mấy giờ mưa?", "insufficient", None),
    ("Bitcoin và đái tháo đường liên quan gì?", "insufficient", None),
    ("Phòng ngừa tiểu đường type 1 thế nào?", "insufficient", None),
    ("Tôi có HbA1c 8.1 thì có mắc tiểu đường?", "blocked", None),
    ("Gần đây khát nhiều và sụt cân. Có bị tiểu đường không?", "blocked", None),
    ("Em đang mang thai, có cần làm xét nghiệm không?", "blocked", None),
    ("Insulin nên tiêm bao nhiêu đơn vị?", "medication", None),
    ("Mẹ em đang nôn liên tục và thở gấp", "urgent", None),
    ("", "invalid", None),
]


def architecture_probe(coordinator):
    scenarios = [CASES[index] for index in (0, 8, -7, -5, -3, -2)]
    scenarios = [(q, expected) for q, expected, _ in scenarios]
    methods = {
        "Direct retrieval": lambda q: ExplanationAgent.compose(graph.answer(q, language="vi")),
        "Multi-agent with rules": lambda q: coordinator.run("qa", question=q)["knowledge"],
    }

    def offline_control(q):
        with patch.object(RuleAgent, "check", return_value=("ok", "")):
            return coordinator.run("qa", question=q)["knowledge"]

    methods["Multi-agent without question rules (offline)"] = offline_control
    results = []
    for name, run in methods.items():
        durations, correct, cited, answers = [], 0, 0, 0
        for question, expected in scenarios * 3:
            start = perf_counter()
            result = run(question)
            durations.append((perf_counter() - start) * 1000)
            correct += result["status"] == expected
            if result["status"] == "ok":
                answers += 1
                cited += all(item["source"]["url"] for item in result["evidence"])
        results.append({"method": name, "requests": len(durations), "expected_status_matches": correct,
                        "answers_with_sources": cited, "answers": answers,
                        "median_ms": round(median(durations), 2)})
    return results


def verify_report(client, combined):
    result = combined.json
    report = result["report"]
    assert report["request_id"] == result["request_id"]
    assert report["prediction"] == result["prediction"]
    assert report["knowledge"] == result["knowledge"]
    assert report["limitations"]
    assert all(isinstance(event["payload"], dict) and event["payload"]["summary"] for event in result["trace"])
    assert all("features" not in event["payload"] and "question" not in event["payload"] for event in result["trace"])
    assert next(event["version"] for event in result["trace"] if event["agent"] == "DL Prediction Agent") == "assignment4-original"
    assert result["prediction"]["model_version"] == "assignment4-original"
    reused = json.loads((ARTIFACTS / "assignment4_original.json").read_text(encoding="utf-8"))
    assert reused["saved_weights_match_original"] and reused["inference_parity_profiles"] == 2

    optional = client.post("/api/run", json={"mode": "combined", "preset": "sample_b", "question": " "})
    assert optional.status_code == 200
    assert optional.json["prediction"] and optional.json["knowledge"] is None
    assert optional.json["report"]["rules"]["question"]["status"] == "not_requested"
    grouped = client.post("/api/run", json={"mode": "predict", "preset": "sample_b", "model": "grouped"})
    assert grouped.status_code == 200 and grouped.json["prediction"]["model_version"] == "brfss-mlp-v2"
    assert client.post("/api/run", json={"mode": "predict", "preset": "sample_b", "model": []}).status_code == 400

    evidence = graph.answer("Xét nghiệm HbA1c có cần nhịn ăn không?", language="vi")["evidence"]
    assert sum("a1c.standard_test.fasting_required=no" in item["claims"] for item in evidence) >= 2

    class ClosePredictor:
        def predict(self, values):
            return {"class_id": 0, "scores": [.40, .37, .23], "model_version": "synthetic-check"}

    def sample_passage(identifier, domain, claim):
        return {"id": identifier, "text": "Synthetic test fixture", "claims": [claim],
                "source": {"url": f"https://{domain}/synthetic-test"}}

    first = sample_passage("fixture-a", "www.who.int", "example.standard.test=alpha")
    second = sample_passage("fixture-b", "www.cdc.gov", "example.standard.test=beta")
    scoped = sample_passage("fixture-c", "www.cdc.gov", "example.other.test=beta")
    assert ExplanationAgent.compose({"status": "ok", "evidence": [first, scoped]})["status"] == "ok"
    same_source = sample_passage("fixture-d", "www.who.int", "example.standard.test=beta")
    assert ExplanationAgent.compose({"status": "ok", "evidence": [first, same_source]})["status"] == "ok"
    unapproved = sample_passage("fixture-e", "example.org", "example.standard.test=beta")
    assert ExplanationAgent.compose({"status": "ok", "evidence": [first, unapproved]})["status"] == "ok"

    class ConflictingGraph:
        def answer(self, question, context=None, language="vi"):
            return {"status": "ok", "evidence": [first, second]}

    conflict = Coordinator(ClosePredictor(), ConflictingGraph()).run("combined", "sample_a", "HbA1c")
    assert conflict["report"]["review"]["status"] == "required"
    assert {reason["code"] for reason in conflict["report"]["review"]["reasons"]} == {
        "conflicting_evidence", "close_scores"}
    assert conflict["knowledge"]["status"] == "conflicting_evidence"
    assert "answer" not in conflict["knowledge"]
    assert conflict["knowledge"]["conflicts"][0]["passage_ids"] == ["fixture-a", "fixture-b"]
    assert conflict["prediction"]["scores"] == [.40, .37, .23]
    urgent = client.post("/api/run", json={"mode": "qa", "question": "Mẹ em đang nôn liên tục và thở gấp"})
    assert urgent.json["report"]["review"]["status"] == "urgent"
    return {"optional_question": "passed", "structured_report": "passed", "assignment4_reuse": "passed",
            "separate_grouped_checkpoint": "passed", "synthetic_close_scores": "passed",
            "synthetic_source_conflict": "passed", "different_scope_and_unapproved_sources": "passed",
            "live_consistent_claims": "passed", "urgent_route": "passed"}


def main():
    expected = build(check=True)
    client = app.test_client()
    graph_response = client.get("/api/graph")
    assert graph_response.status_code == 200
    for label, key, fields in (
            ("Concept", "concepts", ("aliases", "name_vi")),
            ("Passage", "passages", ("title_vi", "text_vi")),
            ("Source", "sources", ("title_vi",))):
        actual = {row["id"]: row["content"]
                  for row in graph.query(f"MATCH (n:{label}) RETURN n.id AS id, properties(n) AS content")}
        for item in expected[key]:
            for field in fields:
                assert actual[item["id"]].get(field) == item.get(field), (label, item["id"], field)
    graph_info = graph_response.json
    assert {row["label"]: row["total"] for row in graph_info["nodes"]} == {
        "Concept": 21, "Passage": 27, "Source": 22}
    assert sum(edge["total"] for edge in graph_info["edges"]) == 112

    results = []
    for question, expected, passage in CASES:
        start = perf_counter()
        response = client.post("/api/run", json={"mode": "qa", "question": question})
        elapsed = round((perf_counter() - start) * 1000, 2)
        assert response.status_code == 200, (question, response.status_code)
        data = response.json
        actual = data["knowledge"]
        ids = [item["id"] for item in actual.get("evidence", [])]
        assert actual["status"] == expected, (question, actual)
        assert data["prediction"] is None
        assert not passage or passage in ids, (question, ids)
        assert expected == "ok" or not ids
        for item in actual.get("evidence", ()):
            assert item["source"]["url"].startswith("https://")
        if expected in ("blocked", "medication", "urgent", "invalid"):
            assert "KG/RAG Agent" not in {event["agent"] for event in data["trace"]}
        if expected == "ok":
            assert actual["answer"] == actual["evidence"][0]["text"]
        results.append({"question": question, "expected": expected, "expected_passage": passage,
                        "actual": actual["status"], "retrieved": ids, "elapsed_ms": elapsed})

    combined = client.post("/api/run", json={"mode": "combined", "preset": "sample_b",
                                            "question": "Chẩn đoán tiểu đường type 1 bằng xét nghiệm nào?"})
    assert combined.status_code == 200
    assert abs(sum(combined.json["prediction"]["scores"]) - 1) < 1e-5
    assert any(item["source"]["publisher"] == "Vinmec" for item in combined.json["knowledge"]["evidence"])
    assert {event["agent"] for event in combined.json["trace"]} == {
        "Coordinator", "Data Agent", "Rule Agent", "DL Prediction Agent",
        "KG/RAG Agent", "Explanation Agent", "Monitoring Agent"}
    assert len({event["request_id"] for event in combined.json["trace"]}) == 1
    report_checks = verify_report(client, combined)

    starter = client.post("/api/run", json={"mode": "qa", "question": "Tiểu đường thai kỳ là gì?"}).json
    assert starter["knowledge"]["context"] == "gestational"
    follow_up = client.post("/api/run", json={"mode": "qa", "question": "Còn xét nghiệm thì sao?",
                                             "context": "gestational"}).json
    assert follow_up["knowledge"]["status"] == "ok"
    assert {"p18", "p19"} <= {item["id"] for item in follow_up["knowledge"]["evidence"]}
    assert graph.answer("What is diabetic ketoacidosis?")["evidence"][0]["id"] == "p22"

    invalid = client.post("/api/run", json={"mode": "combined", "synthetic": True,
                                           "features": {"BMI": 999}, "question": "HbA1c"})
    assert invalid.status_code == 400 and invalid.json["prediction"] is None
    assert "DL Prediction Agent" not in {event["agent"] for event in invalid.json["trace"]}
    assert "KG/RAG Agent" not in {event["agent"] for event in invalid.json["trace"]}
    assert client.post("/api/run", json={"mode": "predict", "features": {"BMI": 27}}).status_code == 400
    assert client.post("/api/run", json=["qa"]).status_code == 400
    report = client.get("/api/experiment").json
    assert len(report["additional"]) == 2
    assert len(report["assignment_benchmarks"]) >= 6
    assert client.get("/").status_code == 200
    assert graph.query("MATCH p=(:Concept)<-[:HAS_RISK_FACTOR]-(:Concept)-[:HAS_PREVENTION_TOPIC]->(:Concept) RETURN count(p) AS total")[0]["total"] == 2
    offline = Coordinator(None, None).run("qa", question="HbA1c")
    assert offline["knowledge"]["status"] == "unavailable"

    retrieval = [row for row in results if row["expected_passage"]]
    output = {"qa_cases": len(results), "retrieval_cases": len(retrieval),
              "hits_at_1": sum(row["retrieved"][0] == row["expected_passage"] for row in retrieval),
              "hits_at_3": sum(row["expected_passage"] in row["retrieved"] for row in retrieval),
              "median_ms": median(row["elapsed_ms"] for row in results),
              "p95_ms": sorted(row["elapsed_ms"] for row in results)[math.ceil(.95 * len(results)) - 1],
              "reference_precision_at_3": sum(row["expected_passage"] in row["retrieved"] for row in retrieval)
              / (3 * len(retrieval)),
              "cases": results, "graph": graph_info, "report_checks": report_checks,
              "architecture_probe": architecture_probe(Coordinator(None, graph)),
              "combined_trace_ms": [{"agent": event["agent"], "elapsed_ms": event["elapsed_ms"]}
                                    for event in combined.json["trace"]]}
    (ARTIFACTS / "demo_verification.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print({key: value for key, value in output.items() if key not in ("cases", "graph")})


if __name__ == "__main__":
    main()

