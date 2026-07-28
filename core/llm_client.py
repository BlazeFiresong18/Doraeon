"""Shared OpenAI chat-completion wrapper with consistent error handling --
used by both query rewriting and answer generation so a rate limit / auth
failure / connection error is reported the same way in either path."""

from __future__ import annotations

import time

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)


def call_chat_completion(
    client: OpenAI | None,
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
) -> tuple[str, float, str | None]:
    """Returns (text, elapsed_ms, error_code). error_code is None on success."""
    if not client:
        return "", 0.0, "missing_api_key"

    t0 = time.perf_counter()
    try:
        completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        text = completion.choices[0].message.content or ""
        return text, (time.perf_counter() - t0) * 1000, None
    except AuthenticationError:
        return "", (time.perf_counter() - t0) * 1000, "invalid_api_key"
    except RateLimitError:
        return "", (time.perf_counter() - t0) * 1000, "rate_limit"
    except APIConnectionError:
        return "", (time.perf_counter() - t0) * 1000, "connection_error"
    except APIStatusError as e:
        return "", (time.perf_counter() - t0) * 1000, f"api_error:{e.message}"
    except Exception as e:  # noqa: BLE001 -- any other SDK failure still degrades gracefully
        return "", (time.perf_counter() - t0) * 1000, f"api_error:{e}"


def error_message(code: str | None) -> str:
    messages = {
        "missing_api_key": (
            "AI answer generation is disabled because no OpenAI API key is configured. "
            "Add your key to `.env` and refresh the app. Retrieved sources are still available below."
        ),
        "invalid_api_key": "Your OpenAI API key appears invalid. Check `OPENAI_API_KEY` in `.env` and try again.",
        "rate_limit": "OpenAI rate limit reached. Please wait a moment and try again.",
        "connection_error": "Could not reach OpenAI. Check your internet connection and try again.",
    }
    if code and code.startswith("api_error:"):
        return f"OpenAI API error: {code.split(':', 1)[1]}"
    return messages.get(code or "", "An unexpected error occurred while generating the answer.")
