"""Orchestrates one turn: condense follow-up -> retrieve -> confidence
guardrail -> generate -> attribution. Each step is independently testable
(see core/retrieval.py, core/query_rewriting.py) -- this module wires them
together and owns the OpenAI client and the guardrail decision."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI

from core.config import get_min_retrieval_score, get_openai_api_key, get_openai_model
from core.index_store import DoraeonIndex
from core.llm_client import call_chat_completion, error_message
from core.query_rewriting import HistoryTurn, rewrite_standalone_question
from core.retrieval import ScoredChunk, retrieve

SYSTEM_PROMPT = """You are Doraeon, an academic study assistant.
Answer ONLY using the provided academic context.
If the answer is not present in the context, say clearly: "I do not know based on the uploaded materials."
Cite sources by filename and page when possible.
Be concise, accurate, and student-friendly. Use clear paragraphs and bullet points when helpful."""

USER_PROMPT_TEMPLATE = """Answer ONLY using the provided academic context. If the answer is not present, say you do not know.

### Academic context
{context}

### Question
{question}

### Instructions
- Use only the context above.
- Mention source filename and page for key claims.
- Do not repeat raw context verbatim; synthesize a clear answer.
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[ScoredChunk] = field(default_factory=list)
    error: str | None = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    api_configured: bool = True
    # Set only when a follow-up was condensed and the rewrite actually
    # changed it -- lets the UI show what was actually searched for.
    rewritten_question: str | None = None


def _format_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.filename} | page {c.page_number}"
        if c.subject:
            header += f" | {c.subject}"
        if c.unit:
            header += f" | {c.unit}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, doraeon_index: DoraeonIndex, model: str | None = None):
        self.doraeon_index = doraeon_index
        self.model = model or get_openai_model()
        api_key = get_openai_api_key()
        self.api_configured = bool(api_key)
        self.client: OpenAI | None = OpenAI(api_key=api_key) if api_key else None

    def generate_answer(
        self,
        question: str,
        top_k: int = 5,
        subject_filter: str | None = None,
        unit_filter: str | None = None,
        min_score: float | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> RAGResponse:
        search_question = rewrite_standalone_question(self.client, self.model, question, history or [])
        rewritten = search_question if search_question != question else None

        t0 = time.perf_counter()
        results = retrieve(self.doraeon_index, search_question, top_k, subject_filter, unit_filter)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not results:
            return RAGResponse(
                answer="Upload PDFs and build your index first — I don't have any materials to search yet.",
                sources=[],
                retrieval_ms=retrieval_ms,
                api_configured=self.api_configured,
                rewritten_question=rewritten,
            )

        # Hallucination guardrail: if even the single best-matching chunk
        # falls below the confidence threshold, refuse rather than let the
        # LLM rationalize an answer from weak context. Still surface what
        # WAS found, so the refusal is transparent rather than a black box.
        # Runs against search_question (post-rewrite): a condensed, specific
        # query is what should clear this bar, not a raw context-free
        # fragment with nothing for the embedding to latch onto.
        threshold = get_min_retrieval_score() if min_score is None else min_score
        if results[0].confidence < threshold:
            return RAGResponse(
                answer="This isn't covered in your uploaded materials.",
                sources=results,
                error="below_confidence_threshold",
                retrieval_ms=retrieval_ms,
                api_configured=self.api_configured,
                rewritten_question=rewritten,
            )

        context = _format_context(results)
        user_message = USER_PROMPT_TEMPLATE.format(context=context, question=search_question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        answer, generation_ms, err = call_chat_completion(self.client, self.model, messages)

        if err:
            return RAGResponse(
                answer=error_message(err),
                sources=results,
                error=err,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                api_configured=self.api_configured,
                rewritten_question=rewritten,
            )

        return RAGResponse(
            answer=answer,
            sources=results,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            api_configured=True,
            rewritten_question=rewritten,
        )
