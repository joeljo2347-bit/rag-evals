"""Split the help-center markdown into chunks the retrievers index.

Two strategies, so the evals can compare them:
- "section": one chunk per `##` section (the help center's own structure).
- "window":  fixed windows of `size` words with `overlap`, ignoring structure.
Every chunk remembers the section it came from, which is what the evals score against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


@dataclass(frozen=True)
class Chunk:
    id: str        # unique: "exchanges#exchange-windows" or "exchanges#w3"
    section: str   # the section it came from: "exchanges#exchange-windows"
    title: str     # "Exchanges > Exchange windows"
    text: str


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def sections(path: Path) -> List[Chunk]:
    doc = path.stem
    title, current = doc, None
    body: List[str] = []
    out: List[Chunk] = []

    def flush():
        if current and body:
            text = " ".join(" ".join(body).split())
            sid = f"{doc}#{slug(current)}"
            out.append(Chunk(sid, sid, f"{title} > {current}", text))

    for line in path.read_text().splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            current, body = line[3:].strip(), []
        elif line.strip():
            body.append(line.strip())
    flush()
    return out


def windows(path: Path, size: int = 40, overlap: int = 10) -> List[Chunk]:
    """Fixed word windows. A window is credited to the section holding most of its words
    (the first such section on a tie, so labels are the same on every run)."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    words, owners = [], []
    for chunk in sections(path):
        for word in chunk.text.split():
            words.append(word)
            owners.append(chunk)
    out, start, n = [], 0, 0
    while start < len(words):
        span = owners[start:start + size]
        owner = max(dict.fromkeys(span), key=span.count)
        out.append(Chunk(f"{path.stem}#w{n}", owner.section, owner.title,
                         " ".join(words[start:start + size])))
        if start + size >= len(words):
            break
        start, n = start + size - overlap, n + 1
    return out


def load(strategy: str = "section", corpus: Path = CORPUS) -> List[Chunk]:
    paths = sorted(corpus.glob("*.md"))
    if strategy == "section":
        return [c for p in paths for c in sections(p)]
    if strategy == "window":
        return [c for p in paths for c in windows(p)]
    raise ValueError(f"Unknown chunking strategy {strategy!r}")
