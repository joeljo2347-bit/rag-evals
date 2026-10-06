"""Scoring. Each function takes one eval case and one result, so it is easy to test and to read.

Facts are checked by string match after normalizing case and dashes. Each fact may list
alternatives separated by "|" ("1 hour|one hour"); every fact in a case must match.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from ragkit.answer import Answer
from ragkit.retrieve import Hit

_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
                         "’": "'", " ": " ", " ": " "})


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.translate(_DASHES).lower())
    return re.sub(r"(\d) %", r"\1%", text)  # "20 %" -> "20%"


def has_fact(text: str, fact: str) -> bool:
    """Whole-word match, so "no" doesn't match "know" and "20" doesn't match "2026"."""
    t = normalize(text)
    return any(re.search(rf"(?<!\w){re.escape(normalize(alt))}(?!\w)", t) for alt in fact.split("|"))


def rank_of_first_relevant(hits: Sequence[Hit], sections: List[str]) -> int:
    """1-based rank of the first hit from an expected section; 0 if none."""
    for rank, hit in enumerate(hits, 1):
        if hit.chunk.section in sections:
            return rank
    return 0


def retrieval(case: Dict, hits: Sequence[Hit]) -> Dict[str, float]:
    rank = rank_of_first_relevant(hits, case["sections"])
    return {"hit@1": float(rank == 1), "hit@3": float(0 < rank <= 3), "mrr": 1.0 / rank if rank else 0.0}


def generation(case: Dict, result: Answer) -> Dict[str, float]:
    if not case["answerable"]:
        return {"refused_correctly": float(result.refused)}
    correct = not result.refused and all(has_fact(result.text, f) for f in case["facts"])
    return {
        "correct": float(correct),
        "false_refusal": float(result.refused),
        "cites_right_source": float(any(s in case["sections"] for s in result.cited_sections)),
        "uncited": float(result.uncited),
    }


def mean(rows: List[Dict[str, float]], key: str) -> float:
    values = [r[key] for r in rows if key in r]
    return sum(values) / len(values) if values else float("nan")
