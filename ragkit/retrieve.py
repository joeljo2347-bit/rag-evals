"""Three retrievers behind one interface: search(query, k) -> [Hit].

- BM25:   keyword match, strong on exact terms ("loaner kit", "$395").
- Dense:  embedding similarity, strong on paraphrase ("send it back" ~ "return").
- Hybrid: reciprocal rank fusion of the two, which needs no score calibration.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ragkit.chunk import Chunk

_STOP = set("a an and are as at be by can do does for from how i if in is it its me my of on or "
            "the to what when where which who will with you your".split())


def stem(word: str) -> str:
    """A light suffix stemmer: "shipping", "ships", "shipped" -> "ship"; "policies" -> "policy";
    "prices" -> "price"; "billing" -> "bill". Short words are left alone."""
    if len(word) <= 4 or word.endswith("ss"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("ing") and len(word) >= 6:
        base = word[:-3]
    elif word.endswith("ed") and len(word) >= 5:
        base = word[:-2]
    elif word.endswith("s") and not word.endswith(("us", "is")):
        return word[:-1]
    else:
        return word
    if base[-1] == base[-2] and (base[-1] not in "lsz" or len(base) >= 7):
        base = base[:-1]  # shipp -> ship, cancell -> cancel; bill and recall stay
    return base


def tokens(text: str) -> List[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9$]+(?:\.\d+)?", text.lower()) if w not in _STOP]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = list(chunks), k1, b
        self.docs = [Counter(tokens(f"{c.title} {c.text}")) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> List[Hit]:
        terms = tokens(query)
        scores = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            scores.append(s)
        order = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


class Dense:
    def __init__(self, chunks: Sequence[Chunk], embedder):
        self.chunks, self.embedder = list(chunks), embedder
        self.matrix = embedder.documents([f"{c.title}\n{c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 4) -> List[Hit]:
        with np.errstate(all="ignore"):  # numpy 2.0 + macOS Accelerate warns spuriously on matmul
            scores = self.matrix @ self.embedder.query(query)
        order = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


class Hybrid:
    def __init__(self, *retrievers, rrf_k: int = 60, depth: int = 20):
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth

    def search(self, query: str, k: int = 4) -> List[Hit]:
        fused: Dict[str, float] = {}
        by_id: Dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, self.depth)):
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[hit.chunk.id] = hit.chunk
        best = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        return [Hit(by_id[i], fused[i]) for i in best]


def build(kind: str, chunks: Sequence[Chunk], embedder=None):
    if kind == "bm25":
        return BM25(chunks)
    if kind == "dense":
        return Dense(chunks, embedder)
    if kind == "hybrid":
        return Hybrid(BM25(chunks), Dense(chunks, embedder))
    raise ValueError(f"Unknown retriever {kind!r}")
