"""
Tests the follow-up query-condensing fix: extract_history_turns() (pure,
tests the chat_history pairing logic used by app.py) and
RAGPipeline._rewrite_standalone_question() / generate_answer()'s use of it
(fake embedder + monkeypatched _call_openai, no real network/model calls).
"""

import numpy as np

from utils.chunker import TextChunk
from utils.rag_pipeline import RAGPipeline, extract_history_turns
from utils.vector_store import FaissVectorStore


# ---------------------------------------------------------------------------
# extract_history_turns -- pure, no Streamlit/network dependency
# ---------------------------------------------------------------------------

def test_extract_history_turns_empty_list():
    assert extract_history_turns([]) == []


def test_extract_history_turns_pairs_user_then_assistant():
    messages = [
        {"role": "user", "content": "What is a hash table?"},
        {"role": "assistant", "content": "A data structure..."},
    ]
    assert extract_history_turns(messages) == [("What is a hash table?", "A data structure...")]


def test_extract_history_turns_caps_to_max_turns_most_recent_last():
    messages = []
    for i in range(5):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    result = extract_history_turns(messages, max_turns=2)
    assert result == [("q3", "a3"), ("q4", "a4")]


def test_extract_history_turns_ignores_dangling_trailing_user_message():
    # In-flight question not yet answered -- shouldn't happen given how
    # app.py calls this (before append_turn), but must not crash if it does.
    messages = [
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "q1 (unanswered)"},
    ]
    assert extract_history_turns(messages) == [("q0", "a0")]


# ---------------------------------------------------------------------------
# RAGPipeline query rewriting
# ---------------------------------------------------------------------------

class RecordingFakeEmbedder:
    """Records every query string it's asked to embed; returns a fixed vector."""

    def __init__(self, query_vector: np.ndarray):
        self._vec = query_vector
        self.embedded_queries: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.embedded_queries.append(text)
        return self._vec

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.tile(self._vec, (len(texts), 1))


def _store_with_one_chunk(chunk_vector: np.ndarray) -> FaissVectorStore:
    store = FaissVectorStore(dimension=4)
    chunk = TextChunk(
        text="Web analytics tracks user behavior on websites.",
        filename="Lecture5.pdf",
        page_number=2,
        subject="MKTG201",
        unit="Unit 3",
        chunk_index=0,
    )
    store.add(np.array([chunk_vector]), [chunk])
    return store


def _pipeline_with_fake_client(monkeypatch, embedder):
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    store = _store_with_one_chunk(vec)
    pipeline = RAGPipeline(store, embedder=embedder)
    monkeypatch.setattr(pipeline, "client", object())  # truthy: enables the rewrite path
    return pipeline


def test_no_history_skips_rewrite_entirely(monkeypatch):
    embedder = RecordingFakeEmbedder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pipeline = _pipeline_with_fake_client(monkeypatch, embedder)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("no history -> _call_openai must not be invoked for rewriting")

    monkeypatch.setattr(pipeline, "_call_openai", _fail_if_called)

    result = pipeline._rewrite_standalone_question("What is web analytics?", [])
    assert result == "What is web analytics?"


def test_no_client_configured_skips_rewrite():
    embedder = RecordingFakeEmbedder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    store = _store_with_one_chunk(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pipeline = RAGPipeline(store, embedder=embedder)  # no API key in test env -> client is None

    result = pipeline._rewrite_standalone_question(
        "explain the gist of it", [("What is web analytics?", "It tracks user behavior.")]
    )
    assert result == "explain the gist of it"


def test_vague_followup_gets_condensed_using_history(monkeypatch):
    embedder = RecordingFakeEmbedder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pipeline = _pipeline_with_fake_client(monkeypatch, embedder)

    captured_prompt = {}

    def _fake_call_openai(prompt):
        captured_prompt["text"] = prompt
        return "Explain the gist of Web Analytics", 5.0, None

    monkeypatch.setattr(pipeline, "_call_openai", _fake_call_openai)

    history = [("What is web analytics?", "Web analytics tracks user behavior on websites.")]
    result = pipeline._rewrite_standalone_question("explain the gist of it", history)

    assert result == "Explain the gist of Web Analytics"
    assert "web analytics" in captured_prompt["text"].lower()
    assert "explain the gist of it" in captured_prompt["text"]


def test_rewrite_failure_falls_back_to_original_question(monkeypatch):
    embedder = RecordingFakeEmbedder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pipeline = _pipeline_with_fake_client(monkeypatch, embedder)

    monkeypatch.setattr(pipeline, "_call_openai", lambda prompt: ("", 5.0, "rate_limit"))

    history = [("What is web analytics?", "It tracks user behavior.")]
    result = pipeline._rewrite_standalone_question("explain the gist of it", history)
    assert result == "explain the gist of it"


def test_empty_rewrite_response_falls_back_to_original_question(monkeypatch):
    embedder = RecordingFakeEmbedder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pipeline = _pipeline_with_fake_client(monkeypatch, embedder)

    monkeypatch.setattr(pipeline, "_call_openai", lambda prompt: ("   ", 5.0, None))

    history = [("What is web analytics?", "It tracks user behavior.")]
    result = pipeline._rewrite_standalone_question("explain the gist of it", history)
    assert result == "explain the gist of it"


def test_history_is_truncated_to_max_turns_in_prompt(monkeypatch):
    embedder = RecordingFakeEmbedder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pipeline = _pipeline_with_fake_client(monkeypatch, embedder)

    captured_prompt = {}

    def _fake_call_openai(prompt):
        captured_prompt["text"] = prompt
        return "standalone", 5.0, None

    monkeypatch.setattr(pipeline, "_call_openai", _fake_call_openai)

    history = [(f"q{i}", f"a{i}") for i in range(6)]  # more than MAX_HISTORY_TURNS_FOR_CONDENSING
    pipeline._rewrite_standalone_question("follow-up", history)

    # Only the most recent 3 turns should appear; the oldest should not.
    assert "q5" in captured_prompt["text"]
    assert "q3" in captured_prompt["text"]
    assert "q0" not in captured_prompt["text"]


def test_generate_answer_retrieves_using_the_rewritten_question(monkeypatch):
    """End-to-end: retrieval must actually search with the condensed query,
    not the raw vague fragment -- this is the actual reported bug."""
    embedder = RecordingFakeEmbedder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pipeline = _pipeline_with_fake_client(monkeypatch, embedder)

    calls = iter(
        [
            ("Explain the gist of Web Analytics", 5.0, None),  # rewrite call
            ("Web analytics is the study of...", 10.0, None),  # final answer call
        ]
    )
    monkeypatch.setattr(pipeline, "_call_openai", lambda prompt: next(calls))

    history = [("What is web analytics?", "Web analytics tracks user behavior on websites.")]
    resp = pipeline.generate_answer("explain the gist of it", min_score=0.35, history=history)

    assert embedder.embedded_queries == ["Explain the gist of Web Analytics"]
    assert resp.rewritten_question == "Explain the gist of Web Analytics"
    assert resp.answer == "Web analytics is the study of..."


def test_generate_answer_rewritten_question_is_none_when_unchanged(monkeypatch):
    embedder = RecordingFakeEmbedder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pipeline = _pipeline_with_fake_client(monkeypatch, embedder)

    calls = iter(
        [
            ("What is web analytics?", 5.0, None),  # rewrite returns it unchanged
            ("Web analytics is...", 10.0, None),
        ]
    )
    monkeypatch.setattr(pipeline, "_call_openai", lambda prompt: next(calls))

    history = [("prior q", "prior a")]
    resp = pipeline.generate_answer("What is web analytics?", min_score=0.35, history=history)

    assert resp.rewritten_question is None  # unchanged -> not surfaced as a rewrite
