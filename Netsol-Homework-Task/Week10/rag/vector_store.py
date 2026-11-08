"""Step 4 - STORE: a simple vector database (numpy matrix + JSON metadata) with cosine search and persistence."""
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .models import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, embedder_name: str = ""):
        self.embedder_name = embedder_name
        self.embeddings: Optional[np.ndarray] = None      # shape (n_chunks, dim)
        self.chunks: List[Chunk] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- write ---------------------------------------------------------
    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        with self._lock:
            self.embeddings = embeddings if self.embeddings is None else np.vstack([self.embeddings, embeddings])
            self.chunks.extend(chunks)

    def delete_source(self, source: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self.chunks) if c.source != source]
            removed = len(self.chunks) - len(keep)
            if removed:
                self.chunks = [self.chunks[i] for i in keep]
                self.embeddings = self.embeddings[keep] if keep else None
            return removed

    def clear(self) -> None:
        with self._lock:
            self.chunks, self.embeddings = [], None

    # ---- read ----------------------------------------------------------
    def search(self, query_vec: np.ndarray, k: int = 4, min_score: float = 0.0) -> List[RetrievedChunk]:
        with self._lock:
            if self.embeddings is None or not self.chunks:
                return []
            scores = self.embeddings @ query_vec                  # cosine similarity (vectors are normalised)
            top = np.argsort(-scores)[:k]
            return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]

    def sources(self) -> Dict[str, int]:
        with self._lock:
            return dict(Counter(c.source for c in self.chunks))

    # ---- persistence ---------------------------------------------------
    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self.embeddings is None:
                for name in ("embeddings.npy", "chunks.json"):
                    (d / name).unlink(missing_ok=True)
                return
            tmp_npy = d / "embeddings.tmp.npy"
            np.save(tmp_npy, self.embeddings)
            os.replace(tmp_npy, d / "embeddings.npy")
            tmp_json = d / "chunks.tmp.json"
            tmp_json.write_text(json.dumps({"embedder": self.embedder_name,
                                            "chunks": [c.to_dict() for c in self.chunks]}, ensure_ascii=False),
                                encoding="utf-8")
            os.replace(tmp_json, d / "chunks.json")

    @classmethod
    def load(cls, directory, embedder_name: str = "") -> "VectorStore":
        d = Path(directory)
        store = cls(embedder_name)
        if not (d / "embeddings.npy").exists() or not (d / "chunks.json").exists():
            return store
        meta = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        if meta.get("embedder") and embedder_name and meta["embedder"] != embedder_name:
            raise RuntimeError(
                f"The saved index was built with '{meta['embedder']}' but the current embedder is '{embedder_name}'. "
                f"Delete the folder '{d}' (or use the old embedder) and index your documents again.")
        store.embeddings = np.load(d / "embeddings.npy")
        store.chunks = [Chunk.from_dict(c) for c in meta["chunks"]]
        return store
