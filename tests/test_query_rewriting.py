"""Tests for extract_history_turns() (pure) and rewrite_standalone_question()
(mocked local Ollama LLM, no real network) -- the conversational-memory fix."""

from unittest.mock import MagicMock

from core.query_rewriting import extract_history_turns, rewrite_standalone_question


# ---------------------------------------------------------------------------
# extract_history_turns -- pure, no dependencies
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
    messages = [
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "q1 (unanswered)"},
    ]
    assert extract_history_turns(messages) == [("q0", "a0")]


# ---------------------------------------------------------------------------
# rewrite_standalone_question -- mocked local Ollama LLM, no real network
# ---------------------------------------------------------------------------

def _mock_llm(response_text: str):
    llm = MagicMock()
    response = MagicMock()
    response.message.content = response_text
    llm.chat.return_value = response
    return llm


def test_no_history_skips_rewrite_entirely():
    llm = MagicMock()
    result = rewrite_standalone_question(llm, "What is web analytics?", [])
    assert result == "What is web analytics?"
    llm.chat.assert_not_called()


def test_no_llm_configured_skips_rewrite():
    history = [("What is web analytics?", "It tracks user behavior.")]
    result = rewrite_standalone_question(None, "explain the gist of it", history)
    assert result == "explain the gist of it"


def test_vague_followup_gets_condensed_using_history():
    llm = _mock_llm("Explain the gist of Web Analytics")
    history = [("What is web analytics?", "Web analytics tracks user behavior on websites.")]

    result = rewrite_standalone_question(llm, "explain the gist of it", history)

    assert result == "Explain the gist of Web Analytics"
    sent_messages = llm.chat.call_args.args[0]
    prompt = sent_messages[0].content
    assert "web analytics" in prompt.lower()
    assert "explain the gist of it" in prompt


def test_rewrite_failure_falls_back_to_original_question():
    llm = MagicMock()
    llm.chat.side_effect = Exception("boom")
    history = [("What is web analytics?", "It tracks user behavior.")]

    result = rewrite_standalone_question(llm, "explain the gist of it", history)
    assert result == "explain the gist of it"


def test_empty_rewrite_response_falls_back_to_original_question():
    llm = _mock_llm("   ")
    history = [("What is web analytics?", "It tracks user behavior.")]

    result = rewrite_standalone_question(llm, "explain the gist of it", history)
    assert result == "explain the gist of it"


def test_history_is_truncated_to_max_turns_in_prompt():
    llm = _mock_llm("standalone")
    history = [(f"q{i}", f"a{i}") for i in range(6)]  # more than MAX_HISTORY_TURNS_FOR_CONDENSING

    rewrite_standalone_question(llm, "follow-up", history)

    prompt = llm.chat.call_args.args[0][0].content
    assert "q5" in prompt
    assert "q3" in prompt
    assert "q0" not in prompt
