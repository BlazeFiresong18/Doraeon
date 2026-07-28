"""Topic-driven study tools built on top of RAGPipeline.generate_answer():
summarization, flashcard generation, and exam-question prediction. Each just
constructs a specific synthetic question for the topic and reuses the same
retrieval -> guardrail -> generation flow -- including the confidence
guardrail, so asking to summarize a topic that isn't actually in the
uploaded materials still gets refused rather than fabricated."""

from __future__ import annotations

from core.rag_pipeline import RAGPipeline, RAGResponse


def summarize_topic(pipeline: RAGPipeline, topic: str, top_k: int = 8, min_score: float | None = None) -> RAGResponse:
    question = f"Summarize the following academic topic based only on the materials: {topic}"
    return pipeline.generate_answer(question, top_k=top_k, min_score=min_score)


def generate_flashcards(
    pipeline: RAGPipeline, topic: str, count: int = 5, top_k: int = 8, min_score: float | None = None
) -> RAGResponse:
    question = (
        f"Create {count} study flashcards (Q&A pairs) about '{topic}' "
        "using ONLY the context. Format as numbered Q: / A: pairs."
    )
    return pipeline.generate_answer(question, top_k=top_k, min_score=min_score)


def predict_exam_questions(
    pipeline: RAGPipeline, topic: str, count: int = 5, top_k: int = 8, min_score: float | None = None
) -> RAGResponse:
    question = (
        f"Based ONLY on the provided materials about '{topic}', predict {count} likely exam "
        "questions a professor might ask, covering the key concepts, definitions, and any "
        "worked examples or formulas present. For each, briefly note which concept it tests. "
        "Format as numbered questions with a one-line 'Tests:' note after each."
    )
    return pipeline.generate_answer(question, top_k=top_k, min_score=min_score)
