from fastapi.testclient import TestClient

from ragkit import chunk, retrieve
from ragkit.api import create_app


def test_ask_returns_answer_and_marked_sources():
    app = create_app(retrieve.BM25(chunk.load("section")), llm=lambda system, user: "It costs $65 [1].")
    r = TestClient(app).post("/ask", json={"question": "replacement drill price", "k": 3}).json()
    assert r["answer"] == "It costs $65 [1]." and not r["refused"] and not r["uncited"]
    assert r["sources"][0]["section"] == "kits#drill-replacement" and r["sources"][0]["cited"]
    assert [s["cited"] for s in r["sources"][1:]] == [False, False]


def test_validation():
    c = TestClient(create_app(retrieve.BM25(chunk.load("section")), llm=lambda s, u: ""))
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={"question": "x", "k": 50}).status_code == 422
