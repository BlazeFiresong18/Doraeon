"""Centralized environment and OpenAI configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (parent of core/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

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
    Hallucination guardrail threshold, on the SAME normalized 0-1 scale as the
    displayed confidence percentage (i.e. 0.35 means "35%"). See
    core/retrieval.py for the normalization formula this is compared against.
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
