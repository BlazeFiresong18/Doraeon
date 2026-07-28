"""
Tests the hallucination guardrail in RAGPipeline.generate_answer() in isolation,
using a fake embedder (duck-typed, no real SentenceTransformer) so these run
fast and fully offline -- no model download, no OpenAI network calls.
"""

import numpy as np

from utils.chunker import TextChunk
from utils.rag_pipeline import RAGPipeline
from utils.vector_store import FaissVectorStore


class FakeEmbedder:
    """Returns a fixed, caller-controlled vector regardless of input text."""

    def __init__(self, query_vector: np.ndarray):
        self._vec = query_vector

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.tile(self._vec, (len(texts), 1))


def _store_with_one_chunk(chunk_vector: np.ndarray) -> FaissVectorStore:
    store = FaissVectorStore(dimension=4)
    chunk = TextChunk(
        text="Binary search runs in O(log n) time.",
        filename="Lecture3.pdf",
        page_number=7,
        subject="CS101",
        unit="Unit 2",
        chunk_index=0,
    )
    store.add(np.array([chunk_vector]), [chunk])
    return store


def test_low_similarity_triggers_guardrail_without_calling_llm(monkeypatch):
    chunk_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # orthogonal -> ~0 similarity
    store = _store_with_one_chunk(chunk_vec)

    pipeline = RAGPipeline(store, embedder=FakeEmbedder(query_vec))

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should never be called when the guardrail refuses")

    monkeypatch.setattr(pipeline, "_call_openai", _fail_if_called)

    resp = pipeline.generate_answer("irrelevant question", min_score=0.35)

    assert resp.error == "below_confidence_threshold"
    assert resp.answer == "This isn't covered in your uploaded materials."
    assert resp.sources  # still surfaces what was found, for transparency


def test_high_similarity_passes_guardrail_and_reaches_llm(monkeypatch):
    chunk_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identical -> similarity ~1.0
    store = _store_with_one_chunk(chunk_vec)

    pipeline = RAGPipeline(store, embedder=FakeEmbedder(query_vec))

    called = {}

    def _fake_call_openai(user_message):
        called["invoked"] = True
        return "O(log n).", 5.0, None

    monkeypatch.setattr(pipeline, "_call_openai", _fake_call_openai)

    resp = pipeline.generate_answer("What is the time complexity of binary search?", min_score=0.35)

    assert called.get("invoked") is True
    assert resp.error is None
    assert resp.answer == "O(log n)."


def test_threshold_is_configurable_per_call():
    chunk_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_vec = np.array([0.9, 0.436, 0.0, 0.0], dtype=np.float32)  # moderate similarity (~0.9)
    store = _store_with_one_chunk(chunk_vec)
    pipeline = RAGPipeline(store, embedder=FakeEmbedder(query_vec))

    # A strict threshold should refuse this moderate match...
    strict = pipeline.generate_answer("some question", min_score=0.95)
    assert strict.error == "below_confidence_threshold"


def test_no_results_at_all_gives_upload_prompt_not_guardrail_refusal():
    empty_store = FaissVectorStore(dimension=4)
    pipeline = RAGPipeline(empty_store, embedder=FakeEmbedder(np.zeros(4, dtype=np.float32)))

    resp = pipeline.generate_answer("anything", min_score=0.35)

    assert resp.error is None  # distinct from the guardrail's "below_confidence_threshold"
    assert "Upload PDFs" in resp.answer
