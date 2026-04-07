from __future__ import annotations

from typing import Dict, List, Sequence

from retrieval.embedder import Embedder
from retrieval.faiss_store import FAISSStore


class Retriever:
    def __init__(self, embedder: Embedder, store: FAISSStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        k: int = 5,
        source_types: Sequence[str] | None = None,
    ) -> List[Dict]:
        q = self.embedder.encode_query(query)
        results = self.store.search(q, k=max(k * 2, k))

        if source_types:
            source_set = set(source_types)
            results = [r for r in results if r["metadata"].get("source_type") in source_set]

        return results[:k]
