"""Local extractive graph RAG. All answer text and citations come from Neo4j."""
import json
import os
import re
import unicodedata
from pathlib import Path

from neo4j import GraphDatabase, READ_ACCESS, Query
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RETRIEVE = (ROOT / "knowledge_base/cypher/retrieve.cypher").read_text(encoding="utf-8")
LOCAL_RUNTIME = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "DiabetesHybridRuntime"
DESKTOP_SETTINGS = LOCAL_RUNTIME / "neo4j-desktop.json"
CONTEXT_TOPICS = {"type1", "type2", "gestational"}
FOLLOW_UP = ("còn", "vậy", "thế", "what about", "and", "how about", "how is it")
SPECIAL_TOPICS = {"fasting", "threshold", "foot", "hypoglycemia", "dka", "followup"}
RETRIEVAL_MESSAGES = {
    "vi": ("Neo4j chưa có dữ liệu. Hãy chạy lệnh nạp kho tri thức.",
           "Bạn muốn hỏi về triệu chứng, xét nghiệm, phòng ngừa hay biến chứng?",
           "Kho tri thức chưa có đoạn trực tiếp cho tổ hợp chủ đề và típ này."),
    "en": ("The graph is empty. Run the knowledge seed.",
           "Which topic do you mean: symptoms, testing, prevention, or complications?",
           "The knowledge base has no direct evidence for this topic and diabetes type."),
}


def normalize(text):
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9_]+", text))


def phrase_in(text, phrase):
    return f" {normalize(phrase)} " in f" {text} "


def normalize_question(text):
    return re.sub(r"\b(tip|type|loai) ([12]) (va|voi|hay) ([12])\b",
                  r"\1 \2 \3 \1 \4", normalize(text))


def connect():
    if DESKTOP_SETTINGS.is_file():
        settings = json.loads(DESKTOP_SETTINGS.read_text(encoding="utf-8-sig"))
        # Keep a manually selected connection independent of the saved Desktop credentials.
        if os.getenv("NEO4J_URI") in (None, settings["NEO4J_URI"]):
            for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE",
                        "NEO4J_BROWSER_URL", "DIABETES_PORT"):
                os.environ.setdefault(key, settings[key])
    credential = LOCAL_RUNTIME / "neo4j-password.txt"
    password = os.getenv("NEO4J_PASSWORD")
    uri = os.getenv("NEO4J_URI")
    if password is None and uri is None and credential.is_file():
        password = credential.read_text(encoding="utf-8").strip()
    if not password:
        raise ValueError("Neo4j password is missing")
    return GraphDatabase.driver(uri or "bolt://127.0.0.1:7689",
                                auth=(os.getenv("NEO4J_USER", "neo4j"), password),
                                connection_timeout=3.0, connection_acquisition_timeout=5.0,
                                max_transaction_retry_time=0.0)


class GraphRAG:
    def __init__(self, driver):
        self.driver = driver

    def query(self, statement, **params):
        with self.driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"),
                                 default_access_mode=READ_ACCESS) as session:
            return session.run(Query(statement, timeout=4.0), **params).data()

    def status(self):
        nodes = self.query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS total")
        edges = self.query("MATCH ()-[r]->() RETURN type(r) AS relation, count(r) AS total")
        return {"ready": any(row["label"] == "Passage" for row in nodes),
                "nodes": nodes, "edges": edges}

    def answer(self, question, context=None, language="en"):
        messages = RETRIEVAL_MESSAGES[language]
        concepts = self.query("MATCH (c:Concept) RETURN c.id AS id, c.aliases AS aliases ORDER BY c.id")
        cleaned = normalize_question(question)
        if not concepts:
            return {"status": "empty_graph", "message": messages[0]}
        anchors = [c["id"] for c in concepts if any(phrase_in(cleaned, a) for a in c["aliases"])]
        if context in CONTEXT_TOPICS and anchors and not (CONTEXT_TOPICS & set(anchors)):
            if any(cleaned.startswith(normalize(p) + " ") for p in FOLLOW_UP):
                anchors.append(context)
        if "prevention" in anchors and any(phrase_in(cleaned, p) for p in ("giảm nguy cơ", "phòng ngừa", "ngăn ngừa", "prevent", "prevention")):
            anchors = [anchor for anchor in anchors if anchor != "risk"]
        # Suppress the generic diabetes alias when there is a specific topic.
        if "diabetes" in anchors and len(anchors) > 1:
            anchors.remove("diabetes")
        if not anchors or (anchors == ["diabetes"] and any(phrase_in(cleaned, p) for p in ("thế nào", "có sao không", "nguy hiểm không", "is it serious"))):
            return {"status": "insufficient", "message": messages[1]}
        rows = self.query(RETRIEVE, anchors=anchors, language=language)
        if not rows:
            return {"status": "insufficient", "message": messages[2]}

        # Fit sparse character TF-IDF on this small candidate set. Alias
        # grounding and graph distance are independent of lexical similarity.
        corpus = [normalize(f"{p['title']} {p['text']}") for p in rows]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        vectors = vectorizer.fit_transform(corpus)
        lexical = (vectors @ vectorizer.transform([cleaned]).T).toarray().ravel()
        specific = set(anchors) & CONTEXT_TOPICS
        topic_anchors = set(anchors) - CONTEXT_TOPICS - {"diabetes"}
        priority = topic_anchors & SPECIAL_TOPICS
        if priority:
            topic_anchors = priority
        for row, similarity in zip(rows, lexical, strict=True):
            direct = {path["id"] for path in row["paths"] if path["distance"] == 0}
            focused = set(row["focus"]) & (topic_anchors or set(anchors))
            row["score"] = round(float(similarity) + 0.25 * len(direct) +
                                 0.30 * len(focused) +
                                 (0.03 if row["source"]["tier"] == "Public health" else 0), 4)
        ordered = sorted(rows, key=lambda row: (-row["score"], row["id"]))
        if specific == {"type1", "type2"} and not topic_anchors:
            focused = [row for row in ordered if set(row["focus"]) & specific]
            ordered = focused if focused else ordered
            # Give each subtype representation before filling the last citation.
            per_type = [next((row for row in ordered if row["about"] == [subtype]), None)
                        for subtype in ("type1", "type2")]
            selected = [row for row in per_type if row is not None]
            ordered = selected + [row for row in ordered if row not in selected]
        if len(specific) == 1:
            grounded = [row for row in ordered
                        if any(p["id"] in specific and p["distance"] == 0 for p in row["paths"])]
            if topic_anchors:
                grounded = [row for row in grounded if any(
                    p["id"] in topic_anchors and p["distance"] == 0 for p in row["paths"])]
            if not grounded:
                return {"status": "insufficient", "anchors": anchors,
                        "message": messages[2]}
            ordered = grounded
            if not topic_anchors:
                ordered = sorted(ordered, key=lambda row: not bool(set(row["focus"]) & specific))
        elif topic_anchors:
            focused = [row for row in ordered if set(row["focus"]) & topic_anchors]
            if focused:
                ordered = focused
        distinct = {}
        for row in ordered:
            distinct.setdefault(row["source"]["url"], row)
        evidence = list(distinct.values())[:3]
        if not topic_anchors and len(specific) == 1 and (
                cleaned.startswith("what is ") or phrase_in(cleaned, "là gì")):
            definition = next((row for row in distinct.values() if set(row["about"]) == specific), None)
            if definition:
                evidence = ([definition] + [row for row in evidence if row["id"] != definition["id"]])[:3]
        if "fasting" in priority:
            evidence.sort(key=lambda row: row["source"]["tier"] != "Public health")
        return {"status": "ok", "anchors": anchors, "evidence": evidence,
                "context": next(iter(specific)) if len(specific) == 1 else None}

