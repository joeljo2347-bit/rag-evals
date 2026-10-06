"""Blind grading: hand the answers to a grader that knows nothing about this project.

    python -m evals.blind packet    # evals/runs/*.jsonl -> evals/blind/packet.jsonl (+ key.json)
    python -m evals.blind score     # evals/blind/grades.jsonl + key.json -> evals/blind/results.md

The packet holds only what a careful outside reader needs: the help center's full text, each
question, the numbered sources the answering model was shown, and its answer. Item ids are
opaque hashes (no model, no case id) and the items are shuffled; the key that maps them back stays in key.json,
which the grader never sees. No expected answers are included: the grader decides from the help
center itself.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ragkit.chunk import CORPUS

HERE = Path(__file__).resolve().parent
OUT = HERE / "blind"

RUBRIC = """You are grading answers written by an assistant for a dental implant supplier's help center.
You know nothing else about the assistant. Grade strictly, using only the help center text below.

For each item you get: a question, the numbered sources the assistant was shown, and its answer.
Decide, for each item:

- "correct": true only if the answer gives the right answer to the question according to the help
  center, with every figure exact, and contains NO statement that is wrong or not supported by the
  help center (an invented detail, a guess, an unhedged inference, advice the text doesn't give).
  If the help center does not answer the question, "correct" is true only if the assistant says it
  doesn't know (or equivalent) without inventing anything.
- "refused": true if the assistant said it doesn't know / can't answer.
- "citations_ok": true if every bracketed citation number points at a shown source that actually
  supports the sentence it's attached to, and the answer cites at least one source; for a refusal,
  true if it cites nothing.
- "note": one short sentence on anything wrong; empty if nothing is.

Be strict: when in doubt, mark it false. Judge each item on its own.
Reply with one JSON object per line, in this exact form, for every item id:
{"id": "...", "correct": true, "refused": false, "citations_ok": true, "note": ""}
"""


def packet() -> None:
    items, key = [], {}
    runs = sorted((HERE / "runs").glob("*.jsonl"))
    for run in runs:
        for line in run.read_text().splitlines():
            row = json.loads(line)
            item_id = hashlib.sha256(f"{run.stem}|{row['id']}".encode()).hexdigest()[:10]
            items.append({"id": item_id, "question": row["question"], "sources": row["sources"],
                          "answer": row["answer"]})
            key[item_id] = {"model": run.stem, "case": row["id"]}
    random.Random(7).shuffle(items)
    corpus = "\n\n".join(p.read_text() for p in sorted(CORPUS.glob("*.md")))
    OUT.mkdir(exist_ok=True)
    (OUT / "packet.md").write_text(RUBRIC + "\n# Help center (complete)\n\n" + corpus + "\n")
    (OUT / "packet.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(items)} items -> {OUT / 'packet.md'} + packet.jsonl (key kept in key.json)")


def score() -> None:
    key = json.loads((OUT / "key.json").read_text())
    grades = [json.loads(l) for l in (OUT / "grades.jsonl").read_text().splitlines() if l.strip()]
    missing = set(key) - {g["id"] for g in grades}
    if missing:
        sys.exit(f"The grader skipped {len(missing)} items: {sorted(missing)[:5]}")
    cases = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "dataset.jsonl").read_text().splitlines()}
    by_model = defaultdict(list)
    for g in grades:
        k = key[g["id"]]
        by_model[k["model"]].append({**g, "case": k["case"], "answerable": cases[k["case"]]["answerable"]})
    lines = ["# Blind grading", "",
             "Graded by a separate agent that saw only the help center, the questions, the sources each",
             "answer was given and the answers: no expected answers, no model names, items shuffled.", "",
             "| Model | Correct | Answerable correct | Uncovered refused | Citations OK |", "|---|---|---|---|---|"]
    for model, rows in sorted(by_model.items()):
        ans = [r for r in rows if r["answerable"]]
        unc = [r for r in rows if not r["answerable"]]
        pct = lambda xs, f: f"{sum(f(x) for x in xs)}/{len(xs)}"
        lines.append(f"| {model.replace('_', ':')} | {pct(rows, lambda r: r['correct'])} | "
                     f"{pct(ans, lambda r: r['correct'])} | {pct(unc, lambda r: r['refused'])} | "
                     f"{pct(rows, lambda r: r['citations_ok'])} |")
    for model, rows in sorted(by_model.items()):
        bad = [r for r in rows if not (r["correct"] and r["citations_ok"])]
        if bad:
            lines += ["", f"### Marked down: {model.replace('_', ':')}", ""]
            lines += [f"- **{r['case']}** {r['note']}" for r in sorted(bad, key=lambda r: r["case"])]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"packet": packet, "score": score}[sys.argv[1]]()
