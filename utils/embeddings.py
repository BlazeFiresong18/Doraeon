"""SentenceTransformer embedding generation for chunks and queries."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Lazy-loaded embedding model shared across the app session."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized embeddings suitable for cosine similarity via inner product."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
