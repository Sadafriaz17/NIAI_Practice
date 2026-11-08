"""Step 3 - EMBED: text -> vectors (L2-normalised, so a dot product is the cosine similarity)."""
import hashlib
import re
from typing import List

import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype("float32")


class FastEmbedEmbedder:
    """Real semantic embeddings, run locally with ONNX (no GPU, no API key). Model is downloaded on first use."""
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # recommended for BGE models

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir=None):
        from fastembed import TextEmbedding
        self.name = model_name
        self._use_prefix = "bge" in model_name.lower()
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts, batch_size=batch_size)), dtype="float32")
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        q = (self.QUERY_PREFIX if self._use_prefix else "") + text
        return _normalize(np.array(list(self._model.embed([q])), dtype="float32"))[0]


class HashingEmbedder:
    """Tiny keyword-based stand-in (no download). Only for tests / offline demos, NOT semantic."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]


def get_embedder(settings):
    if settings.embedder == "hash":
        return HashingEmbedder()
    return FastEmbedEmbedder(settings.embed_model, settings.model_cache_dir)
