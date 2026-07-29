"""
Guardrail tests using REAL retrieval/embeddings (the actual reported bug was
about real scoring behavior) with the Ollama generation call mocked out (no
real local model needed for the LLM step, which isn't what's being tested here).
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
    assert resp.grounded is True
    assert resp.refused is False


def test_unrelated_question_triggers_guardrail_without_calling_llm_in_strict_mode():
    pipeline = _pipeline_with_indexed_doc()

    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        resp = pipeline.generate_answer("What is the boiling point of water?", min_score=0.35, strict_mode=True)

    assert resp.error == "below_confidence_threshold"
    assert resp.answer == "This isn't covered in your uploaded materials."
    assert resp.sources  # still surfaced for transparency
    assert resp.grounded is False
    assert resp.refused is True
    assert not mock_call.called  # never even attempted -- the whole point of strict mode


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


# ---------------------------------------------------------------------------
# Soft mode (strict_mode=False): fall back to general knowledge with a
# disclaimer instead of a hard refusal, when nothing clears the threshold.
# ---------------------------------------------------------------------------

def test_soft_mode_falls_back_to_general_knowledge_with_disclaimer():
    pipeline = _pipeline_with_indexed_doc()

    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        mock_call.return_value = ("Communication is the exchange of information between parties.", 10.0, None)
        resp = pipeline.generate_answer("what is communication", min_score=0.35, strict_mode=False)

    assert resp.error is None
    assert resp.refused is False
    assert resp.grounded is False
    assert resp.answer.startswith("This isn't directly covered in your uploaded materials, but generally speaking:")
    assert "Communication is the exchange of information between parties." in resp.answer
    assert mock_call.called  # unlike strict mode, the LLM IS consulted here


def test_soft_mode_still_surfaces_closest_matches_for_transparency():
    """Requirement: even in soft mode, show what was almost relevant."""
    pipeline = _pipeline_with_indexed_doc()

    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        mock_call.return_value = ("some general answer", 10.0, None)
        resp = pipeline.generate_answer("what is communication", min_score=0.35, strict_mode=False)

    assert resp.sources  # the near-miss chunks are still returned
    assert resp.grounded is False  # but explicitly flagged as not what grounded the answer


def test_soft_mode_general_knowledge_prompt_does_not_include_retrieved_context():
    """The general-knowledge fallback must not be fed the (irrelevant, below-
    threshold) retrieved chunks as if they were real context -- that's the
    whole point of not pretending this is grounded."""
    pipeline = _pipeline_with_indexed_doc()

    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        mock_call.return_value = ("answer", 10.0, None)
        pipeline.generate_answer("what is communication", min_score=0.35, strict_mode=False)

    sent_messages = mock_call.call_args.args[1]
    user_message = sent_messages[1]["content"]
    assert "Web analytics is the measurement" not in user_message  # no leaked irrelevant context
    assert "what is communication" in user_message.lower()


def test_soft_mode_does_not_change_behavior_when_confidence_clears_threshold():
    """Soft vs strict only matters when nothing clears the threshold -- a
    genuine match should answer normally either way."""
    pipeline = _pipeline_with_indexed_doc()

    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        mock_call.return_value = ("Web analytics tracks behavior.", 10.0, None)
        resp = pipeline.generate_answer("What is web analytics?", min_score=0.35, strict_mode=False)

    assert resp.grounded is True
    assert resp.refused is False
    assert not resp.answer.startswith("This isn't directly covered")


def test_soft_mode_fallback_llm_failure_is_reported_as_error_not_general_knowledge():
    pipeline = _pipeline_with_indexed_doc()

    with patch("core.rag_pipeline.call_chat_completion") as mock_call:
        mock_call.return_value = ("", 5.0, "connection_error")
        resp = pipeline.generate_answer("what is communication", min_score=0.35, strict_mode=False)

    assert resp.error == "connection_error"
    assert resp.grounded is False
    assert "ollama" in resp.answer.lower()
