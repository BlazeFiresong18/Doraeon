"""Centralized environment and local-model configuration.

Doraeon runs fully offline after the one-time model downloads/pulls: the LLM
is a local Ollama model (no API key, no per-token cost), and embeddings are a
local sentence-transformers model loaded via LlamaIndex's HuggingFace wrapper.
"""

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
# Tried bge-small-en-v1.5 as an upgrade (empirically confirmed elsewhere to
# have somewhat better raw retrieval accuracy) but reverted: its cosine
# similarity distribution doesn't separate relevant from irrelevant text
# cleanly enough for a threshold-based guardrail to work. Confirmed against
# a real indexed document -- a nonsense query ("asdkjfh qwoeiur zxcvbn")
# scored 0.65-0.72, HIGHER than a genuine in-document term ("bounce rate",
# 0.64-0.69), with or without the model's recommended query instruction
# prefix. MiniLM keeps unrelated text near 0.0-0.2 and relevant text at
# 0.3-0.8+, a wide, reliable gap the guardrail's single global threshold
# depends on. Not a compatibility issue -- both models work fine mechanically,
# this one just isn't fit for this app's confidence-threshold design.
EMBEDDING_QUERY_INSTRUCTION = None

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

_loaded = False


def load_settings() -> None:
    """Load .env from project root, then cwd (Streamlit may use either)."""
    global _loaded
    if _loaded:
        return
    load_dotenv(ENV_FILE, override=True)
    load_dotenv(override=False)
    _loaded = True


def get_ollama_model() -> str:
    load_settings()
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def get_ollama_base_url() -> str:
    load_settings()
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


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
    model = get_ollama_model()
    base_url = get_ollama_base_url()
    return (
        "Doraeon needs a local [Ollama](https://ollama.com/download) server with a model pulled:\n\n"
        "```\n"
        "1. Install Ollama\n"
        f"2. ollama pull {model}\n"
        "3. ollama serve\n"
        "```\n\n"
        f"Optionally override the model or address in `.env`:\n\n```\nOLLAMA_MODEL={model}\n"
        f"OLLAMA_BASE_URL={base_url}\n```"
    )
