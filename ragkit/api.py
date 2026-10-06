"""HTTP API: POST /ask returns the answer, its citations and the sources it was given.

    uvicorn ragkit.api:app --port 8000

The index is built once at startup. Swap the corpus folder (RAG_CORPUS) to point it at your own
markdown; every `##` section becomes a chunk.
"""

from __future__ import annotations

import os
import urllib.error
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragkit import chunk, retrieve
from ragkit.answer import OllamaChat, answer
from ragkit.embed import OLLAMA, OllamaEmbedder


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(4, ge=1, le=10)


class Source(BaseModel):
    n: int
    section: str
    title: str
    score: float
    cited: bool


class AskOut(BaseModel):
    answer: str
    refused: bool
    uncited: bool
    sources: List[Source]


def create_app(retriever=None, llm=None) -> FastAPI:
    if retriever is None:
        corpus = Path(os.environ.get("RAG_CORPUS", chunk.CORPUS))
        try:
            retriever = retrieve.build(os.environ.get("RAG_RETRIEVER", "hybrid"),
                                       chunk.load("section", corpus), OllamaEmbedder())
        except (urllib.error.URLError, OSError) as exc:
            raise SystemExit(f"Can't reach Ollama at {OLLAMA} to embed the documents ({exc}). "
                             "Start Ollama, or set OLLAMA_HOST.") from None
    llm = llm or OllamaChat(os.environ.get("RAG_MODEL", "gpt-oss:20b"))
    app = FastAPI(title="Help-center RAG", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskOut)
    def ask(body: AskIn) -> AskOut:
        try:
            result = answer(body.question, retriever, llm, body.k)
        except (urllib.error.URLError, OSError, KeyError) as exc:
            raise HTTPException(503, f"The model is unavailable: {exc}") from None
        return AskOut(answer=result.text, refused=result.refused, uncited=result.uncited, sources=[
            Source(n=i, section=h.chunk.section, title=h.chunk.title, score=round(h.score, 4),
                   cited=i in result.citations)
            for i, h in enumerate(result.hits, 1)])

    return app


_app = None


def __getattr__(name: str):  # `uvicorn ragkit.api:app` builds the index once, on first use
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
