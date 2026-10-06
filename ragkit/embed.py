"""Embeddings: Ollama's nomic-embed-text, cached on disk; and a hashing stand-in for tests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "embeddings.json"


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


class OllamaEmbedder:
    """nomic-embed-text wants task prefixes: documents and queries are embedded differently."""

    def __init__(self, model: str = "nomic-embed-text", cache: Path = CACHE):
        self.model, self.cache_path, self.lock = model, cache, threading.Lock()
        self.cache: Dict[str, List[float]] = json.loads(cache.read_text()) if cache.exists() else {}

    def _embed(self, texts: List[str]) -> np.ndarray:
        keys = [hashlib.sha1(f"{self.model}\n{t}".encode()).hexdigest() for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self.cache]
        if missing:
            body = json.dumps({"model": self.model, "input": missing}).encode()
            req = urllib.request.Request(f"{OLLAMA}/api/embed", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                vectors = json.load(resp)["embeddings"]
            with self.lock:  # the API answers from a thread pool
                for text, vector in zip(missing, vectors):
                    self.cache[hashlib.sha1(f"{self.model}\n{text}".encode()).hexdigest()] = vector
                self.cache_path.parent.mkdir(exist_ok=True)
                tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(self.cache))
                os.replace(tmp, self.cache_path)  # never leave a half-written cache
        return _normalize(np.array([self.cache[k] for k in keys], dtype=np.float32))

    def documents(self, texts: List[str]) -> np.ndarray:
        return self._embed([f"search_document: {t}" for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._embed([f"search_query: {text}"])[0]


class HashEmbedder:
    """Bag of hashed words. Deterministic and offline: good enough to test the plumbing."""

    def __init__(self, dims: int = 512):
        self.dims = dims

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dims, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1
        return v

    def documents(self, texts: List[str]) -> np.ndarray:
        return _normalize(np.array([self._one(t) for t in texts]))

    def query(self, text: str) -> np.ndarray:
        return _normalize(self._one(text)[None])[0]
