"""Shared local-Ollama chat-completion wrapper with consistent error handling
-- used by both query rewriting and answer generation so a down server /
missing model is reported the same way in either path."""

from __future__ import annotations

import time

import ollama
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

_ROLE_MAP = {
    "system": MessageRole.SYSTEM,
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
}


def build_llm(model: str, base_url: str, temperature: float = 0.2, request_timeout: float = 120.0) -> Ollama:
    # Ollama's LlamaIndex wrapper fixes temperature at construction time --
    # there's no per-call override, so callers needing a different
    # temperature (e.g. the eval judge wants 0.0 for determinism) build a
    # second instance rather than passing temperature through call_chat_completion.
    #
    # context_window MUST be set explicitly: the wrapper's default (-1) asks
    # the model for its maximum trained context and requests that as num_ctx.
    # llama3.2 advertises 131072 tokens, whose KV cache alone is ~13.5 GB --
    # an instant out-of-memory on any normal machine (confirmed: exactly the
    # "failed to allocate buffer of size 14495514624" crash). 8192 tokens
    # comfortably fits this app's real prompts (top-k chunks + question) at
    # under 1 GB of KV cache.
    return Ollama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        request_timeout=request_timeout,
        context_window=8192,
    )


def call_chat_completion(llm: Ollama, messages: list[dict]) -> tuple[str, float, str | None]:
    """Returns (text, elapsed_ms, error_code). error_code is None on success."""
    chat_messages = [ChatMessage(role=_ROLE_MAP.get(m["role"], MessageRole.USER), content=m["content"]) for m in messages]

    t0 = time.perf_counter()
    try:
        response = llm.chat(chat_messages)
        text = response.message.content or ""
        return text, (time.perf_counter() - t0) * 1000, None
    except ConnectionError:
        # Raised by the ollama client itself when the server isn't reachable.
        return "", (time.perf_counter() - t0) * 1000, "connection_error"
    except ollama.ResponseError as e:
        if e.status_code == 404 or "not found" in str(e).lower():
            return "", (time.perf_counter() - t0) * 1000, "model_not_found"
        return "", (time.perf_counter() - t0) * 1000, f"ollama_error:{e.error}"
    except Exception as e:  # noqa: BLE001 -- any other failure still degrades gracefully
        return "", (time.perf_counter() - t0) * 1000, f"ollama_error:{e}"


def error_message(code: str | None) -> str:
    messages = {
        "connection_error": (
            "Can't reach Ollama. Make sure it's installed and running (`ollama serve`), "
            "then try again. Retrieved sources are still available below."
        ),
        "model_not_found": (
            "The configured Ollama model isn't pulled yet. Run `ollama pull <model>` "
            "(check OLLAMA_MODEL in `.env`) and try again."
        ),
    }
    if code and code.startswith("ollama_error:"):
        return f"Ollama error: {code.split(':', 1)[1]}"
    return messages.get(code or "", "An unexpected error occurred while generating the answer.")


def check_ollama_status(model: str, base_url: str, timeout: float = 2.0) -> tuple[bool, str]:
    """Best-effort connectivity probe for the sidebar status indicator. Not
    used on the generation path itself (that already has its own error
    handling via call_chat_completion) -- purely a fast, friendly UI signal,
    so failures here are always treated as 'not ready', never raised."""
    try:
        client = ollama.Client(host=base_url, timeout=timeout)
        response = client.list()
        available = {m.model for m in response.models if m.model}
        bare_names = {name.split(":")[0] for name in available}
        if model in available or model in bare_names:
            return True, f"Ollama ready ({model})"
        return False, f"Ollama is running, but '{model}' isn't pulled yet -- run: ollama pull {model}"
    except Exception:
        return False, "Ollama isn't reachable -- make sure it's installed and running (`ollama serve`)."
