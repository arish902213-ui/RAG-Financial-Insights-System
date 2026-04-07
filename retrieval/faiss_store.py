from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np


class FAISSStore:
    def __init__(self, storage_dir: str | Path = "data/index") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_dir / "index.faiss"
        self.records_path = self.storage_dir / "records.json"

        self.index: faiss.IndexFlatIP | None = None
        self.records: List[Dict] = []
        self.dimension: int | None = None

        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.index_path.exists() and self.records_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.records_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
            self.dimension = self.index.d

    def _init_index(self, dim: int) -> None:
        self.dimension = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray, docs: List[Dict]) -> None:
        if embeddings.shape[0] != len(docs):
            raise ValueError("Number of embeddings must match number of docs.")

        if embeddings.shape[0] == 0:
            return

        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]
        if self.index is None:
            self._init_index(dim)

        if self.dimension != dim:
            raise ValueError("Embedding dimension mismatch with existing index.")

        self.index.add(embeddings)
        self.records.extend(docs)
        self.persist()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query = query_embedding.astype("float32")
        scores, indices = self.index.search(query, k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            results.append(
                {
                    "score": float(score),
                    "text": rec["text"],
                    "metadata": rec["metadata"],
                }
            )
        return results

    def persist(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.records_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def count(self) -> int:
        if self.index is None:
            return 0
        return int(self.index.ntotal)

    def clear(self) -> None:
        self.index = None
        self.records = []
        self.dimension = None
        if self.index_path.exists():
            self.index_path.unlink()
        if self.records_path.exists():
            self.records_path.unlink()
