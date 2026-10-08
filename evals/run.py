"""Run the evals and write evals/results.md.

    python -m evals.run                                  # retrieval + answers with gpt-oss:20b
    python -m evals.run --models gpt-oss:20b,qwen3:8b    # compare answer models
    python -m evals.run --retrieval-only                 # fast: prints retrieval only, results.md untouched

1. Retrieval: every retriever x chunking strategy on the answerable questions (hit@1, hit@3, MRR).
2. Answers: the chosen retriever feeds each model; scored on facts, citations, refusals and,
   through an LLM judge, whether every claim is supported by the sources it was shown.
Per-question results go to evals/runs/ so failures can be read, not just counted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from ragkit import chunk, metrics, retrieve
from ragkit.answer import OllamaChat, answer, sources_block
from ragkit.embed import OllamaEmbedder

HERE = Path(__file__).resolve().parent
JUDGE_PROMPT = (
    "You check answers for a help center. Given the sources and an answer, decide whether every "
    "factual claim in the answer is stated in the sources. Citations like [1] are not claims. "
    'Reply with JSON only: {"supported": true or false, "claim": "the first unsupported claim, or empty"}'
)


def load_cases() -> List[Dict]:
    return [json.loads(line) for line in (HERE / "dataset.jsonl").read_text().splitlines() if line]


def judge(llm, hits, text: str) -> Dict:
    raw = llm(JUDGE_PROMPT, f"Sources:\n\n{sources_block(hits)}\n\nAnswer:\n{text}")
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        verdict = json.loads(raw[start:end])
        supported = verdict.get("supported")
        if isinstance(supported, str):
            supported = supported.strip().lower() == "true"
        return {"supported": supported is True, "claim": verdict.get("claim", "")}
    except ValueError:
        return {"supported": False, "claim": f"(judge reply unreadable: {raw[:80]})"}


def run_retrieval(cases, embedder, k: int) -> List[Dict]:
    rows = []
    answerable = [c for c in cases if c["answerable"]]
    for chunking in ("section", "window"):
        chunks = chunk.load(chunking)
        for kind in ("bm25", "dense", "hybrid"):
            r = retrieve.build(kind, chunks, embedder)
            scores = [metrics.retrieval(c, r.search(c["question"], k)) for c in answerable]
            rows.append({"retriever": kind, "chunking": chunking,
                         **{m: metrics.mean(scores, m) for m in ("hit@1", "hit@3", "mrr")}})
            print(f"  {kind:6} {chunking:7}  hit@1 {rows[-1]['hit@1']:.2f}  hit@3 {rows[-1]['hit@3']:.2f}  mrr {rows[-1]['mrr']:.2f}")
    return rows


def answer_one(case: Dict, retriever, llm, judge_llm, k: int) -> Dict:
    """Answer one question, score it, and ask the judge about supported claims."""
    t0 = time.time()
    result = answer(case["question"], retriever, llm, k)
    seconds = time.time() - t0
    score, verdict = metrics.generation(case, result), None
    if case["answerable"] and not result.refused and judge_llm is not None:
        verdict = judge(judge_llm, result.hits, result.text)
        score["faithful"] = float(verdict["supported"])
    flag = "ok " if score.get("correct", score.get("refused_correctly")) else "MISS"
    print(f"  {flag} {case['id']} {seconds:5.1f}s  {result.text[:70]!r}")
    return {"id": case["id"], "question": case["question"], "answer": result.text,
            "retrieved": [h.chunk.section for h in result.hits], "sources": sources_block(result.hits),
            "cited": result.cited_sections, "score": score, "judge": verdict, "seconds": round(seconds, 1)}


def run_answers(cases, retriever, model: str, judge_llm, k: int) -> Dict:
    llm = OllamaChat(model)
    details = [answer_one(c, retriever, llm, judge_llm, k) for c in cases]
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"{model.replace(':', '_').replace('/', '_')}.jsonl").write_text(
        "\n".join(json.dumps(d) for d in details) + "\n")
    scores = [d["score"] for d in details]
    keys = ("correct", "cites_right_source", "uncited", "false_refusal", "refused_correctly", "faithful")
    summary = {"model": model, **{key: metrics.mean(scores, key) for key in keys},
               "seconds": sum(d["seconds"] for d in details) / len(details)}
    summary["misses"] = [d for d in details
                         if not d["score"].get("correct", d["score"].get("refused_correctly"))
                         or d["score"].get("faithful") == 0.0]
    return summary


def pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:.0f}%"


def write_report(retrieval_rows, answer_rows, retriever_name: str, judge_model: str, n_a: int, n_u: int,
                 k: int = 4) -> None:
    lines = ["# Eval results", "",
             f"{n_a} answerable questions and {n_u} the help center doesn't cover. "
             "Generated by `python -m evals.run`.", "",
             f"## Retrieval (answerable questions, k={k})", "",
             "| Retriever | Chunking | hit@1 | hit@3 | MRR |", "|---|---|---|---|---|"]
    for r in retrieval_rows:
        lines.append(f"| {r['retriever']} | {r['chunking']} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {r['mrr']:.2f} |")
    if answer_rows:
        lines += ["", f"## Answers ({retriever_name}, k={k}; judge: {judge_model})", "",
                  "| Model | Correct | Cites right source | Faithful (judge) | Refuses uncovered | False refusals | Uncited | s/question |",
                  "|---|---|---|---|---|---|---|---|"]
        for a in answer_rows:
            lines.append(f"| {a['model']} | {pct(a['correct'])} | {pct(a['cites_right_source'])} | "
                         f"{pct(a['faithful'])} | {pct(a['refused_correctly'])} | {pct(a['false_refusal'])} | "
                         f"{pct(a['uncited'])} | {a['seconds']:.1f} |")
        for a in answer_rows:
            if a["misses"]:
                lines += ["", f"### Misses: {a['model']}", ""]
                for m in a["misses"]:
                    why = f" Judge: {m['judge']['claim']}" if m["judge"] and not m["judge"]["supported"] else ""
                    lines.append(f"- **{m['id']}** {m['question']} → {m['answer'][:160]!r}.{why}")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="gpt-oss:20b")
    p.add_argument("--judge", default="gpt-oss:20b")
    p.add_argument("--retriever", default="hybrid")
    p.add_argument("--chunking", default="section")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--retrieval-only", action="store_true")
    a = p.parse_args()

    cases, embedder = load_cases(), OllamaEmbedder()
    print("Retrieval")
    retrieval_rows = run_retrieval(cases, embedder, a.k)
    if a.retrieval_only:  # print only: a partial run must not overwrite the published results
        return
    r = retrieve.build(a.retriever, chunk.load(a.chunking), embedder)
    judge_llm = OllamaChat(a.judge) if a.judge else None
    answer_rows = []
    for model in a.models.split(","):
        print(f"Answers: {model}")
        answer_rows.append(run_answers(cases, r, model, judge_llm, a.k))
    write_report(retrieval_rows, answer_rows, f"{a.retriever} retriever, {a.chunking} chunks",
                 a.judge or "none", sum(c["answerable"] for c in cases),
                 sum(not c["answerable"] for c in cases), a.k)
    print(f"Wrote {HERE / 'results.md'}")


if __name__ == "__main__":
    main()
