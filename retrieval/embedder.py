from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.dimension = 384
        self.backend = "sentence_transformers"
        self._model = None
        self._fallback_vectorizer = None

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception:
            self.backend = "hashing_fallback"
            self._fallback_vectorizer = HashingVectorizer(
                n_features=self.dimension,
                alternate_sign=False,
                norm=None,
            )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors.astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype("float32")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        if self._model is not None:
            vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vectors.astype("float32")

        sparse = self._fallback_vectorizer.transform(texts)
        dense = sparse.toarray().astype("float32")
        return self._normalize(dense)

    def encode_query(self, text: str) -> np.ndarray:
        if self._model is not None:
            vector = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            return vector.astype("float32")

        sparse = self._fallback_vectorizer.transform([text])
        dense = sparse.toarray().astype("float32")
        return self._normalize(dense)
