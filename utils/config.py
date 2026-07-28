"""Centralized environment and OpenAI configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (parent of utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

_loaded = False


def load_settings() -> None:
    """Load .env from project root, then cwd (Streamlit may use either)."""
    global _loaded
    if _loaded:
        return
    load_dotenv(ENV_FILE, override=True)
    load_dotenv(override=False)
    _loaded = True


def get_openai_api_key() -> str | None:
    load_settings()
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or key.lower() in {"your_openai_api_key_here", "sk-your-key-here"}:
        return None
    return key


def get_openai_model() -> str:
    load_settings()
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_min_retrieval_score(default: float = 0.35) -> float:
    """
    Hallucination guardrail threshold: minimum cosine similarity (0-1) the
    single best-retrieved chunk must reach before the LLM is asked to answer
    at all. Below this, the app refuses rather than letting the model
    rationalize an answer from a weak match. Empirical/content-dependent --
    tune via MIN_RETRIEVAL_SCORE in .env, or live via the sidebar slider.
    """
    load_settings()
    raw = os.getenv("MIN_RETRIEVAL_SCORE")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(1.0, max(0.0, value))


def env_file_exists() -> bool:
    return ENV_FILE.is_file()


def env_setup_hint() -> str:
    env_path = ENV_FILE.resolve()
    if env_file_exists():
        return (
            f"`.env` exists at `{env_path}` but `OPENAI_API_KEY` is missing or still a placeholder. "
            "Add a valid key and refresh this page."
        )
    return (
        f"Create a `.env` file at:\n\n**`{env_path}`**\n\n"
        "```\nOPENAI_API_KEY=sk-your-key-here\nOPENAI_MODEL=gpt-4o-mini\n```\n\n"
        "Copy from `.env.example` in the project root."
    )
