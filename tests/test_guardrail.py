"""
Guardrail tests using REAL retrieval/embeddings (the actual reported bug was
about real scoring behavior) with the OpenAI generation call mocked out
(no real network cost for the LLM step, which isn't what's being tested here).
"""

from unittest.mock import patch

from llama_index.core import Document

from core.index_store import DoraeonIndex
from core.rag_pipeline import RAGPipeline

WEB_ANALYTICS_TEXT = """Web analytics is the measurement, collection, analysis, and reporting of
web data to understand and optimize web usage. Key metrics include page views, bounce rate,
session duration, and conversion rate. Google Analytics is the most widely used web analytics tool."""


def _pipeline_with_indexed_doc() -> RAGPipeline:
    idx = DoraeonIndex()
    doc = Document(
        text=WEB_ANALYTICS_TEXT,
        metadata={"filename": "Lecture5.pdf", "page_number": 2, "subject": "MKTG201", "unit": "Unit 3"},
    )
    idx.add_documents([doc], chunk_size=400, chunk_overlap=80)
    return RAGPipeline(idx)


def test_direct_match_question_passes_guardrail_and_calls_llm():
    pipeline = _pipeline_with_indexed_doc()

    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        mock_call.return_value = ("Web analytics tracks user behavior on websites.", 10.0, None)
        resp = pipeline.generate_answer("What is web analytics?", min_score=0.35)

    assert resp.error is None
    assert resp.answer == "Web analytics tracks user behavior on websites."
    assert mock_call.called
    assert resp.sources[0].confidence > 0.5


def test_unrelated_question_triggers_guardrail_without_calling_llm():
    pipeline = _pipeline_with_indexed_doc()

    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        resp = pipeline.generate_answer("What is the boiling point of water?", min_score=0.35)

    assert resp.error == "below_confidence_threshold"
    assert resp.answer == "This isn't covered in your uploaded materials."
    assert resp.sources  # still surfaced for transparency
    assert not mock_call.called  # never even attempted -- the whole point of the guardrail


def test_guardrail_threshold_is_configurable():
    pipeline = _pipeline_with_indexed_doc()

    # A moderately-related query that clears a lenient threshold but not a strict one
    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        mock_call.return_value = ("answer", 10.0, None)
        lenient = pipeline.generate_answer("bounce rate", min_score=0.05)
        strict = pipeline.generate_answer("bounce rate", min_score=0.99)

    assert lenient.error is None
    assert strict.error == "below_confidence_threshold"


def test_no_documents_indexed_gives_upload_prompt_not_guardrail_refusal():
    idx = DoraeonIndex()
    pipeline = RAGPipeline(idx)

    resp = pipeline.generate_answer("anything", min_score=0.35)

    assert resp.error is None  # distinct from "below_confidence_threshold"
    assert "Upload PDFs" in resp.answer


def test_rewritten_question_used_for_both_guardrail_and_generation():
    """The actual reported bug: retrieval/guardrail must see the condensed
    query, not the raw vague fragment."""
    pipeline = _pipeline_with_indexed_doc()
    history = [("What is web analytics?", "Web analytics tracks user behavior.")]

    with patch("core.rag_pipeline.rewrite_standalone_question") as mock_rewrite, \
         patch("core.rag_pipeline.call_chat_completion") as mock_call:
        mock_rewrite.return_value = "Explain the gist of Web Analytics"
        mock_call.return_value = ("It's about tracking behavior.", 10.0, None)

        resp = pipeline.generate_answer("explain the gist of it", min_score=0.35, history=history)

    assert resp.error is None  # would have been refused had the raw fragment been used
    assert resp.rewritten_question == "Explain the gist of Web Analytics"
