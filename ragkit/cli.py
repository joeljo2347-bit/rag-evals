"""Ask the help center a question from the terminal.

    python -m ragkit.cli "Can I exchange an implant whose box I opened?"
    python -m ragkit.cli --search-only --retriever bm25 "loaner kit deposit"
"""

from __future__ import annotations

import argparse

from ragkit import chunk, retrieve
from ragkit.answer import OllamaChat, answer
from ragkit.embed import OllamaEmbedder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--chunking", default="section", choices=["section", "window"])
    p.add_argument("--model", default="gpt-oss:20b")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--search-only", action="store_true")
    a = p.parse_args()

    r = retrieve.build(a.retriever, chunk.load(a.chunking), OllamaEmbedder())
    hits = r.search(a.question, a.k)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.score:.3f}  {h.chunk.title}")
    if a.search_only:
        return
    result = answer(a.question, r, OllamaChat(a.model), hits=hits)
    print(f"\n{result.text}")
    if result.uncited:
        print("\n(warning: no valid citation in this answer)")


if __name__ == "__main__":
    main()
