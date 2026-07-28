"""FAISS-backed vector store for semantic chunk retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from utils.chunker import TextChunk


@dataclass
class SearchResult:
    """One retrieved chunk with similarity score."""

    chunk: TextChunk
    score: float


class FaissVectorStore:
    """
    In-memory FAISS index using inner product on normalized vectors (= cosine similarity).
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[TextChunk] = []

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add(self, embeddings: np.ndarray, chunks: list[TextChunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks length must match")
        if len(chunks) == 0:
            return
        vectors = np.ascontiguousarray(embeddings.astype(np.float32))
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        subject_filter: str | None = None,
        unit_filter: str | None = None,
    ) -> list[SearchResult]:
        if self.size == 0:
            return []

        q = np.ascontiguousarray(query_embedding.reshape(1, -1).astype(np.float32))
        # Retrieve extra candidates when filtering so top_k still fills after filter
        fetch_k = min(self.size, max(top_k * 4, top_k))
        scores, indices = self.index.search(q, fetch_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if subject_filter and chunk.subject.lower() != subject_filter.lower():
                continue
            if unit_filter and unit_filter.lower() not in chunk.unit.lower():
                continue
            results.append(SearchResult(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
