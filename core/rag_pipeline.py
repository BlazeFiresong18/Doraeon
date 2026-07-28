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

# Soft-mode fallback: used only when retrieval couldn't clear the confidence
# threshold AND strict_mode is off. Deliberately a different prompt from the
# grounded one above -- it must NOT claim the answer comes from the user's
# materials, since it doesn't.
GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """You are Doraeon, an academic study assistant.
The user's uploaded materials don't cover this question. Answer it using your own general
knowledge instead, clearly and helpfully. Be concise and accurate. Do not imply this information
comes from the user's uploaded materials -- it doesn't."""

GENERAL_KNOWLEDGE_USER_PROMPT_TEMPLATE = """The user's uploaded materials don't cover this question. \
Answer it using your general knowledge.

### Question
{question}

### Instructions
- Answer helpfully and accurately from general knowledge.
- Be concise.
- Do not claim or imply this comes from their course materials.
"""

DISCLAIMER_PREFIX = "This isn't directly covered in your uploaded materials, but generally speaking:"


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
    # True when the answer is actually grounded in retrieved chunks above the
    # confidence threshold. False for both a strict-mode refusal and a
    # soft-mode general-knowledge fallback -- in neither case should the UI
    # present `sources` as real attribution (see `refused` to tell those two
    # apart: refused=True means no answer was given at all).
    grounded: bool = True
    # True only for the strict-mode hard refusal ("This isn't covered...").
    # False for a normal grounded answer AND for a soft-mode fallback --
    # soft mode still produces a real answer, it's just not grounded.
    refused: bool = False


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
        strict_mode: bool = True,
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
                grounded=False,
            )

        # Hallucination guardrail: if even the single best-matching chunk
        # falls below the confidence threshold, the retrieved context isn't
        # trustworthy enough to ground an answer in. What happens next
        # depends on strict_mode:
        #   - strict (default): refuse outright, same as before. Still
        #     surface what WAS found, so the refusal is transparent.
        #   - soft: let the LLM answer from general knowledge instead, but
        #     with a disclaimer prefix and NOT presented as grounded --
        #     `sources` still carries the near-miss chunks for transparency
        #     (requirement: show "closest matches" even in soft mode), but
        #     `grounded=False` tells the UI never to render them as real
        #     attribution for this answer.
        # Runs against search_question (post-rewrite): a condensed, specific
        # query is what should clear this bar, not a raw context-free
        # fragment with nothing for the embedding to latch onto.
        threshold = get_min_retrieval_score() if min_score is None else min_score
        if results[0].confidence < threshold:
            if strict_mode:
                return RAGResponse(
                    answer="This isn't covered in your uploaded materials.",
                    sources=results,
                    error="below_confidence_threshold",
                    retrieval_ms=retrieval_ms,
                    api_configured=self.api_configured,
                    rewritten_question=rewritten,
                    grounded=False,
                    refused=True,
                )

            messages = [
                {"role": "system", "content": GENERAL_KNOWLEDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": GENERAL_KNOWLEDGE_USER_PROMPT_TEMPLATE.format(question=search_question),
                },
            ]
            raw_answer, generation_ms, err = call_chat_completion(self.client, self.model, messages)

            if err:
                return RAGResponse(
                    answer=error_message(err),
                    sources=results,
                    error=err,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    api_configured=self.api_configured,
                    rewritten_question=rewritten,
                    grounded=False,
                )

            return RAGResponse(
                answer=f"{DISCLAIMER_PREFIX}\n\n{raw_answer}",
                sources=results,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                api_configured=True,
                rewritten_question=rewritten,
                grounded=False,
                refused=False,
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
