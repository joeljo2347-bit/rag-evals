"""Chunking, retrieval, citations and scoring, with no model: HashEmbedder and a stub LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit import chunk, metrics, retrieve
from ragkit.answer import REFUSAL, answer, parse
from ragkit.embed import HashEmbedder

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.jsonl"


@pytest.fixture(scope="module")
def sections():
    return chunk.load("section")


def test_sections_follow_the_headings(sections):
    ids = {c.id for c in sections}
    assert "exchanges#exchange-windows" in ids and "kits#loaner-kits" in ids
    window = next(c for c in sections if c.id == "exchanges#exchange-windows")
    assert window.title == "Exchanges > Exchange windows"
    assert window.text.startswith("Unopened products can be exchanged")


def test_windows_overlap_and_credit_a_section():
    chunks = chunk.windows(chunk.CORPUS / "shipping.md", size=20, overlap=5)
    assert len(chunks) > 3
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]
    assert all(c.section.startswith("shipping#") for c in chunks)


def test_every_eval_section_exists(sections):
    ids = {c.section for c in sections}
    for case in map(json.loads, DATASET.read_text().splitlines()):
        assert set(case["sections"]) <= ids, case["id"]
        assert case["answerable"] == bool(case["facts"]), case["id"]


def test_stemming_joins_word_forms():
    assert retrieve.tokens("Shipping ships shipped") == ["ship", "ship", "ship"]
    assert retrieve.stem("cancelled") == "cancel" and retrieve.stem("glass") == "glass"


def test_bm25_finds_exact_terms(sections):
    hits = retrieve.BM25(sections).search("loaner kit deposit", 3)
    assert hits[0].chunk.id == "kits#loaner-kits"


def test_dense_and_hybrid_rank(sections):
    dense = retrieve.Dense(sections, HashEmbedder())
    assert dense.search("borrow a loaner kit refundable deposit returned complete", 1)[0].chunk.id == "kits#loaner-kits"
    hybrid = retrieve.Hybrid(retrieve.BM25(sections), dense)
    hits = hybrid.search("drills replaced after uses single replacement drills", 4)
    assert hits[0].chunk.id == "kits#drill-replacement"
    assert len({h.chunk.id for h in hits}) == 4


def test_rrf_rewards_agreement():
    a, b, c = (chunk.Chunk(x, x, x, x) for x in "abc")

    class Fixed:
        def __init__(self, order):
            self.order = order

        def search(self, q, k):
            return [retrieve.Hit(x, 1.0) for x in self.order][:k]

    fused = retrieve.Hybrid(Fixed([a, b, c]), Fixed([b, c, a])).search("q", 3)
    assert [h.chunk.id for h in fused] == ["b", "a", "c"]


def test_citations_are_parsed_and_checked(sections):
    hits = retrieve.BM25(sections).search("restocking fee returns credit", 3)
    result = parse("Labels cost 15% [1], free for members [1, 3]. See also [9].", hits)
    assert result.citations == [1, 3]           # [9] wasn't shown, so it doesn't count
    assert not result.uncited and not result.refused
    assert parse("Labels cost 15%.", hits).uncited
    assert parse(REFUSAL, hits).refused


def test_answer_sends_numbered_sources(sections):
    seen = {}

    def llm(system, user):
        seen["user"] = user
        return "It costs $65 [1]."

    result = answer("replacement drill price", retrieve.BM25(sections), llm, k=2)
    assert "[1] Surgical kits > Drill replacement" in seen["user"]
    assert result.cited_sections == ["kits#drill-replacement"]


def test_fact_matching_normalizes():
    assert metrics.has_fact("Order before 1 p.m. Pacific", "1 p.m.|1 pm")
    assert metrics.has_fact("a two‑year warranty", "2-year|two-year")
    assert not metrics.has_fact("30 days", "60")


def test_scores(sections):
    hits = retrieve.BM25(sections).search("restocking fee returns credit", 4)
    case = {"answerable": True, "sections": ["returns#returns-for-credit"], "facts": ["15%"]}
    rank = metrics.rank_of_first_relevant(hits, case["sections"])
    assert metrics.retrieval(case, hits)["mrr"] == pytest.approx(1 / rank)
    good = parse(f"A 15% restocking fee applies [{rank}].", hits)
    assert metrics.generation(case, good) == {"correct": 1.0, "false_refusal": 0.0,
                                              "cites_right_source": 1.0, "uncited": 0.0}
    assert metrics.generation(case, parse(REFUSAL, hits))["false_refusal"] == 1.0
    uncovered = {"answerable": False, "sections": [], "facts": []}
    assert metrics.generation(uncovered, parse(REFUSAL, hits)) == {"refused_correctly": 1.0}


def test_full_width_citations_count(sections):
    hits = retrieve.BM25(sections).search("restocking fee returns credit", 3)
    assert parse("Labels cost 15%【1】.", hits).citations == [1]
    assert parse("I don’t know based on the help center.", hits).refused


def test_percent_spacing():
    assert metrics.has_fact("a 20 % discount", "20%")


def test_stemmer_keeps_word_forms_together():
    for a, b in [("prices", "price"), ("sizes", "size"), ("exchanges", "exchange"), ("policies", "policy"),
                 ("billing", "bill"), ("recalled", "recall"), ("boxes", "box"), ("early", "early")]:
        assert retrieve.stem(a) == retrieve.stem(b), (a, b)


def test_facts_match_whole_words():
    assert not metrics.has_fact("I don't know", "not|no")
    assert not metrics.has_fact("how often", "10|ten")
    assert not metrics.has_fact("in 2026", "20")
    assert metrics.has_fact("No, it isn't.", "not|no") and metrics.has_fact("over $1,000.", "$1,000")


def test_window_labels_are_stable():
    first = [c.section for c in chunk.load("window")]
    assert first == [c.section for c in chunk.load("window")]
    with pytest.raises(ValueError):
        chunk.windows(chunk.CORPUS / "shipping.md", size=10, overlap=10)


def test_dagger_citations():
    hits = retrieve.BM25(chunk.load("section")).search("loaner kit deposit", 3)
    assert parse("A $500 deposit【1†L3-L5】.", hits).citations == [1]


def test_answer_one_records_what_the_grader_needs(sections):
    from evals.run import answer_one
    case = {"id": "x", "question": "replacement drill price", "answerable": True,
            "sections": ["kits#drill-replacement"], "facts": ["$65"]}
    row = answer_one(case, retrieve.BM25(sections), lambda s, u: "Single drills cost $65 [1].", None, 2)
    assert row["score"]["correct"] == 1.0 and row["sources"].startswith("[1] Surgical kits > Drill replacement")


def test_retrieval_only_leaves_published_results_alone(monkeypatch, capsys):
    import sys

    from evals import run
    monkeypatch.setattr(run, "OllamaEmbedder", HashEmbedder)
    monkeypatch.setattr(run, "write_report", lambda *a, **kw: pytest.fail("results.md was overwritten"))
    monkeypatch.setattr(sys, "argv", ["run", "--retrieval-only"])
    run.main()
    assert "bm25   section" in capsys.readouterr().out


def test_cli_answers_from_rag_corpus(tmp_path, monkeypatch, capsys):
    import sys

    from ragkit import cli
    (tmp_path / "office.md").write_text("# Office\n\n## Parking\nVisitors park in lot B.\n")
    monkeypatch.setenv("RAG_CORPUS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["cli", "--search-only", "--retriever", "bm25", "where to park"])
    cli.main()
    assert "Office > Parking" in capsys.readouterr().out
