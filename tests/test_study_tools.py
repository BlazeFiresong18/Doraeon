"""Study tools are thin wrappers around RAGPipeline.generate_answer() --
tests verify each constructs a sensible question and passes through params,
using a mock pipeline (the underlying generate_answer/guardrail/retrieval
behavior is already covered by test_guardrail.py and test_retrieval_scoring.py)."""

from unittest.mock import MagicMock

from core.study_tools import generate_flashcards, predict_exam_questions, summarize_topic


def test_summarize_topic_mentions_the_topic_and_passes_params():
    pipeline = MagicMock()
    pipeline.generate_answer.return_value = "mocked response"

    summarize_topic(pipeline, "Binary Search Trees", top_k=8, min_score=0.4)

    call = pipeline.generate_answer.call_args
    assert "Binary Search Trees" in call.args[0]
    assert call.kwargs["top_k"] == 8
    assert call.kwargs["min_score"] == 0.4


def test_generate_flashcards_includes_requested_count_and_topic():
    pipeline = MagicMock()
    generate_flashcards(pipeline, "Recursion", count=7, top_k=6)

    question = pipeline.generate_answer.call_args.args[0]
    assert "Recursion" in question
    assert "7" in question


def test_predict_exam_questions_includes_topic_and_count():
    pipeline = MagicMock()
    predict_exam_questions(pipeline, "Web Analytics", count=3, top_k=6)

    question = pipeline.generate_answer.call_args.args[0]
    assert "Web Analytics" in question
    assert "3" in question
    assert "exam questions" in question.lower()


def test_all_study_tools_return_the_pipeline_response_unchanged():
    pipeline = MagicMock()
    pipeline.generate_answer.return_value = "sentinel-response"

    assert summarize_topic(pipeline, "X") == "sentinel-response"
    assert generate_flashcards(pipeline, "X") == "sentinel-response"
    assert predict_exam_questions(pipeline, "X") == "sentinel-response"
