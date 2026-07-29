"""Condenses a history-dependent follow-up ("explain the gist of it", "what
about the second point") into a standalone question before it reaches
retrieval. Without this, a vague follow-up is embedded and searched
literally as-is -- it has no antecedent for "it", so it can't retrieve well
regardless of whether the underlying content exists."""

from __future__ import annotations

from llama_index.llms.ollama import Ollama

from core.llm_client import call_chat_completion

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

# A prior turn: (user_question, assistant_answer_text).
HistoryTurn = tuple[str, str]


def extract_history_turns(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS_FOR_CONDENSING) -> list[HistoryTurn]:
    """Pair up a flat chat_history-shaped list ([{"role":..., "content":...}, ...])
    into (question, answer) turns, most-recent-last, capped to max_turns.

    A pure function (no Streamlit dependency) so it's testable in isolation --
    app.py is a script with top-level Streamlit calls that execute on import,
    so logic that needs testing lives here instead.
    """
    pairs = [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(0, len(messages) - 1, 2)
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant"
    ]
    return pairs[-max_turns:] if max_turns else pairs


def rewrite_standalone_question(
    llm: Ollama | None,
    question: str,
    history: list[HistoryTurn],
) -> str:
    """Returns `question` unchanged (no LLM call, no cost) when there's no
    history to draw on -- the common case for a first message. On any LLM
    failure, falls back to the original question rather than blocking the
    turn on a rewrite error."""
    if not history or not llm:
        return question

    recent = history[-MAX_HISTORY_TURNS_FOR_CONDENSING:]
    history_text = "\n".join(
        f"User: {q}\nAssistant: {a[:_HISTORY_ANSWER_TRUNCATE_CHARS]}" for q, a in recent
    )
    prompt = CONDENSE_PROMPT_TEMPLATE.format(history=history_text, question=question)
    rewritten, _ms, err = call_chat_completion(llm, [{"role": "user", "content": prompt}])
    if err or not rewritten.strip():
        return question
    return rewritten.strip()
