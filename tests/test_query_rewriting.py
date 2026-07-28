"""Tests for extract_history_turns() (pure) and rewrite_standalone_question()
(mocked LLM call, no real network) -- the conversational-memory fix."""

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
# rewrite_standalone_question -- mocked OpenAI client, no real network
# ---------------------------------------------------------------------------

def _mock_client(response_text: str):
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=response_text))]
    client.chat.completions.create.return_value = completion
    return client


def test_no_history_skips_rewrite_entirely():
    client = MagicMock()
    result = rewrite_standalone_question(client, "gpt-4o-mini", "What is web analytics?", [])
    assert result == "What is web analytics?"
    client.chat.completions.create.assert_not_called()


def test_no_client_configured_skips_rewrite():
    history = [("What is web analytics?", "It tracks user behavior.")]
    result = rewrite_standalone_question(None, "gpt-4o-mini", "explain the gist of it", history)
    assert result == "explain the gist of it"


def test_vague_followup_gets_condensed_using_history():
    client = _mock_client("Explain the gist of Web Analytics")
    history = [("What is web analytics?", "Web analytics tracks user behavior on websites.")]

    result = rewrite_standalone_question(client, "gpt-4o-mini", "explain the gist of it", history)

    assert result == "Explain the gist of Web Analytics"
    call_kwargs = client.chat.completions.create.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]
    assert "web analytics" in prompt.lower()
    assert "explain the gist of it" in prompt


def test_rewrite_failure_falls_back_to_original_question():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("boom")
    history = [("What is web analytics?", "It tracks user behavior.")]

    result = rewrite_standalone_question(client, "gpt-4o-mini", "explain the gist of it", history)
    assert result == "explain the gist of it"


def test_empty_rewrite_response_falls_back_to_original_question():
    client = _mock_client("   ")
    history = [("What is web analytics?", "It tracks user behavior.")]

    result = rewrite_standalone_question(client, "gpt-4o-mini", "explain the gist of it", history)
    assert result == "explain the gist of it"


def test_history_is_truncated_to_max_turns_in_prompt():
    client = _mock_client("standalone")
    history = [(f"q{i}", f"a{i}") for i in range(6)]  # more than MAX_HISTORY_TURNS_FOR_CONDENSING

    rewrite_standalone_question(client, "gpt-4o-mini", "follow-up", history)

    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "q5" in prompt
    assert "q3" in prompt
    assert "q0" not in prompt
