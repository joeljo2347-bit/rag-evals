"""Answer a question from retrieved sources, with citations, or say it isn't covered.

The model sees numbered sources and must cite them as [1], [2]. Afterwards the code, not the
model, decides what counts: citations to sources that weren't shown are dropped, and an answer
with no valid citation that isn't a refusal is flagged as uncited.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

from ragkit.retrieve import Hit

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
REFUSAL = "I don't know based on the help center."

_SYSTEM = (
    "You answer questions from dentists and practice staff using only the numbered help-center sources you are given.\n"
    "- Cite the source of every fact in square brackets, like [2]. Cite only numbers you were given.\n"
    "- Give exact figures (prices, days, hours) as the sources state them.\n"
    f"- If the sources do not answer the question, reply exactly: {REFUSAL}\n"
    "- Two or three sentences at most. No preamble."
)


class OllamaChat:
    def __init__(self, model: str = "gpt-oss:20b"):
        self.model = model

    def __call__(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(f"{OLLAMA}/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            content = json.load(resp)["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


@dataclass
class Answer:
    text: str
    hits: List[Hit]
    citations: List[int] = field(default_factory=list)  # 1-based source numbers, valid ones only
    refused: bool = False
    uncited: bool = False

    @property
    def cited_sections(self) -> List[str]:
        return [self.hits[n - 1].chunk.section for n in self.citations]


def sources_block(hits: List[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk.title}\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def parse(text: str, hits: List[Hit]) -> Answer:
    refused = REFUSAL.lower().rstrip(".").replace("'", "") in text.lower().replace("\u2019", "'").replace("'", "")
    cited = []
    # Some models cite with full-width brackets, 【1】 or 【1†L3-L5】, as well as [1].
    for group in re.findall(r"[\[\u3010](\d+(?:\s*,\s*\d+)*)(?:\u2020[^\]\u3011]*)?[\]\u3011]", text):
        for n in (int(x) for x in group.split(",")):
            if 1 <= n <= len(hits) and n not in cited:
                cited.append(n)
    return Answer(text, hits, cited, refused, uncited=not refused and not cited)


def answer(question: str, retriever, llm, k: int = 4, hits: Optional[List[Hit]] = None) -> Answer:
    hits = hits if hits is not None else retriever.search(question, k)
    reply = llm(_SYSTEM, f"Sources:\n\n{sources_block(hits)}\n\nQuestion: {question}")
    return parse(reply, hits)
