"""RAG query pipeline: retrieve context and generate grounded answers via OpenAI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError

from utils.config import get_min_retrieval_score, get_openai_api_key, get_openai_model
from utils.embeddings import EmbeddingModel
from utils.text_cleaner import clean_pdf_text
from utils.vector_store import FaissVectorStore, SearchResult

SYSTEM_PROMPT = """You are Doraeon, an academic study assistant.
Answer ONLY using the provided academic context.
If the answer is not present in the context, say clearly: "I do not know based on the uploaded materials."
Cite sources by filename and page when possible.
Be concise, accurate, and student-friendly. Use clear paragraphs and bullet points when helpful."""

# Condenses a history-dependent follow-up ("explain the gist of it", "tell me
# more") into a standalone question before it ever reaches retrieval. Without
# this, a vague follow-up is embedded and searched literally as-is -- it has
# no antecedent for "it", so it can't retrieve well regardless of whether the
# underlying content exists, and the confidence guardrail then (correctly)
# refuses a genuinely underspecified query. This is the fix for that: give
# retrieval a real query to work with, not the guardrail a lower bar to clear.
CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a follow-up message, rewrite the \
follow-up as a standalone question that includes whatever context (topic, subject, or entity \
referred to by "it"/"this"/"that"/"the topic") is needed to understand it without the history. \
If the follow-up is already self-contained, return it unchanged. Respond with ONLY the rewritten \
question and nothing else -- no preamble, no quotes.

### Conversation history
{history}

### Follow-up message
{question}

### Standalone question
"""

# How many prior turns to include when condensing a follow-up. Bounds prompt
# size/cost; recent turns are what pronouns like "it"/"that" almost always
# refer back to.
MAX_HISTORY_TURNS_FOR_CONDENSING = 3
_HISTORY_ANSWER_TRUNCATE_CHARS = 300

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
    sources: list[SearchResult] = field(default_factory=list)
    error: str | None = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    api_configured: bool = True
    # Set only when a follow-up was condensed into a standalone question and
    # the rewrite actually changed it -- lets the UI show what was actually
    # searched for, so query rewriting is visible/debuggable, not a black box.
    rewritten_question: str | None = None


# A prior turn, for condensing follow-ups: (user_question, assistant_answer_text).
HistoryTurn = tuple[str, str]


def extract_history_turns(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS_FOR_CONDENSING) -> list[HistoryTurn]:
    """Pair up a flat chat_history-shaped list ([{"role":..., "content":...}, ...])
    into (question, answer) turns, most-recent-last, capped to max_turns.

    A pure function (no Streamlit dependency) so it's testable in isolation --
    app.py is a script with top-level Streamlit calls that execute on import,
    so logic that needs testing lives here instead, with app.py as a thin
    wrapper around st.session_state.chat_history.
    """
    pairs = [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(0, len(messages) - 1, 2)
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant"
    ]
    return pairs[-max_turns:] if max_turns else pairs


class RAGPipeline:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedder: EmbeddingModel | None = None,
        model: str | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder or EmbeddingModel()
        self.model = model or get_openai_model()
        api_key = get_openai_api_key()
        self.api_configured = bool(api_key)
        self.client = OpenAI(api_key=api_key) if api_key else None

    def _rewrite_standalone_question(self, question: str, history: list[HistoryTurn]) -> str:
        """Condense `question` + recent `history` into a standalone question
        for retrieval/generation. Returns `question` unchanged (no LLM call,
        no cost) when there's no history to draw on -- the common case for a
        first message. On any LLM failure, falls back to the original
        question rather than blocking the turn on a rewrite error."""
        if not history:
            return question
        if not self.client:
            return question

        recent = history[-MAX_HISTORY_TURNS_FOR_CONDENSING:]
        history_text = "\n".join(
            f"User: {q}\nAssistant: {a[:_HISTORY_ANSWER_TRUNCATE_CHARS]}" for q, a in recent
        )
        prompt = CONDENSE_PROMPT_TEMPLATE.format(history=history_text, question=question)
        rewritten, _ms, err = self._call_openai(prompt)
        if err or not rewritten.strip():
            return question
        return rewritten.strip()

    def _format_context(self, results: list[SearchResult]) -> str:
        blocks = []
        for i, r in enumerate(results, start=1):
            c = r.chunk
            text = clean_pdf_text(c.text)
            header = f"[{i}] {c.filename} | page {c.page_number}"
            if c.subject:
                header += f" | {c.subject}"
            if c.unit:
                header += f" | {c.unit}"
            blocks.append(f"{header}\n{text}")
        return "\n\n---\n\n".join(blocks)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        subject_filter: str | None = None,
        unit_filter: str | None = None,
    ) -> tuple[list[SearchResult], float]:
        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(question)
        results = self.vector_store.search(
            query_vec,
            top_k=top_k,
            subject_filter=subject_filter or None,
            unit_filter=unit_filter or None,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return results, elapsed_ms

    def _call_openai(self, user_message: str) -> tuple[str, float, str | None]:
        if not self.client:
            return "", 0.0, "missing_api_key"

        t0 = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            )
            answer = completion.choices[0].message.content or ""
            return answer, (time.perf_counter() - t0) * 1000, None
        except AuthenticationError:
            return "", (time.perf_counter() - t0) * 1000, "invalid_api_key"
        except RateLimitError:
            return "", (time.perf_counter() - t0) * 1000, "rate_limit"
        except APIConnectionError:
            return "", (time.perf_counter() - t0) * 1000, "connection_error"
        except APIStatusError as e:
            return "", (time.perf_counter() - t0) * 1000, f"api_error:{e.message}"
        except Exception as e:
            return "", (time.perf_counter() - t0) * 1000, f"api_error:{str(e)}"

    @staticmethod
    def _error_message(code: str | None) -> str:
        messages = {
            "missing_api_key": (
                "AI answer generation is disabled because no OpenAI API key is configured. "
                "Add your key to `.env` and refresh the app. Retrieved sources are still available below."
            ),
            "invalid_api_key": (
                "Your OpenAI API key appears invalid. Check `OPENAI_API_KEY` in `.env` and try again."
            ),
            "rate_limit": (
                "OpenAI rate limit reached. Please wait a moment and try again."
            ),
            "connection_error": (
                "Could not reach OpenAI. Check your internet connection and try again."
            ),
        }
        if code and code.startswith("api_error:"):
            return f"OpenAI API error: {code.split(':', 1)[1]}"
        return messages.get(code or "", "An unexpected error occurred while generating the answer.")

    def generate_answer(
        self,
        question: str,
        top_k: int = 5,
        subject_filter: str | None = None,
        unit_filter: str | None = None,
        min_score: float | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> RAGResponse:
        search_question = self._rewrite_standalone_question(question, history or [])
        rewritten = search_question if search_question != question else None

        results, retrieval_ms = self.retrieve(search_question, top_k, subject_filter, unit_filter)

        if not results:
            return RAGResponse(
                answer="Upload PDFs and build your index first — I don't have any materials to search yet.",
                sources=[],
                retrieval_ms=retrieval_ms,
                api_configured=self.api_configured,
                rewritten_question=rewritten,
            )

        # Hallucination guardrail: if even the single best-matching chunk falls
        # below the confidence threshold, refuse rather than let the LLM
        # rationalize an answer from weak context. Still surface what WAS
        # found, so the refusal is transparent rather than a black box.
        # Runs against search_question (post-rewrite) -- a condensed, specific
        # query is what should clear this bar, not the raw "explain the gist
        # of it" fragment, which has nothing for the embedding to latch onto.
        threshold = get_min_retrieval_score() if min_score is None else min_score
        if results[0].score < threshold:
            return RAGResponse(
                answer="This isn't covered in your uploaded materials.",
                sources=results,
                error="below_confidence_threshold",
                retrieval_ms=retrieval_ms,
                api_configured=self.api_configured,
                rewritten_question=rewritten,
            )

        context = self._format_context(results)
        user_message = USER_PROMPT_TEMPLATE.format(context=context, question=search_question)

        answer, generation_ms, err = self._call_openai(user_message)

        if err:
            return RAGResponse(
                answer=self._error_message(err),
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

    def summarize_topic(self, topic: str, top_k: int = 8, min_score: float | None = None) -> RAGResponse:
        question = f"Summarize the following academic topic based only on the materials: {topic}"
        return self.generate_answer(question, top_k=top_k, min_score=min_score)

    def generate_flashcards(
        self, topic: str, count: int = 5, top_k: int = 8, min_score: float | None = None
    ) -> RAGResponse:
        question = (
            f"Create {count} study flashcards (Q&A pairs) about '{topic}' "
            "using ONLY the context. Format as numbered Q: / A: pairs."
        )
        return self.generate_answer(question, top_k=top_k, min_score=min_score)
