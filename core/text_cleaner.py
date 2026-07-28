"""Normalize and repair text extracted from academic PDFs."""

from __future__ import annotations

import re


def _fix_hyphenation(text: str) -> str:
    """Join words split across lines: 'exam-\\nple' -> 'example'."""
    return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)


def _collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def _collapse_repeated_chars_in_word(word: str) -> str:
    """Reduce PDF artifacts like 'Thhe' -> 'The' (3+ same char -> 2)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


def _remove_consecutive_duplicate_words(text: str) -> str:
    """Remove stutter duplicates: 'the the' -> 'the'."""
    words = text.split()
    if not words:
        return text
    cleaned: list[str] = []
    prev = ""
    for w in words:
        core = re.sub(r"[^\w]", "", w.lower())
        prev_core = re.sub(r"[^\w]", "", prev.lower())
        if core and core == prev_core:
            continue
        cleaned.append(_collapse_repeated_chars_in_word(w))
        prev = w
    return " ".join(cleaned)


def _merge_short_lines(text: str) -> str:
    """
    Rejoin lines that look like broken paragraphs (single short lines).
    Preserves blank-line paragraph breaks.
    """
    paragraphs: list[str] = []
    buffer: list[str] = []

    for line in text.split("\n"):
        if not line.strip():
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            paragraphs.append("")
            continue
        if len(line.split()) <= 3 and buffer:
            buffer.append(line.strip())
        elif buffer and len(line) < 60 and not line.endswith((".", "!", "?", ":")):
            buffer.append(line.strip())
        else:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            buffer = [line.strip()]

    if buffer:
        paragraphs.append(" ".join(buffer))

    return "\n\n".join(p for p in paragraphs if p)


def clean_pdf_text(text: str) -> str:
    """Full cleanup pipeline for one page or chunk of PDF text."""
    if not text or not text.strip():
        return ""
    text = _fix_hyphenation(text)
    text = _collapse_whitespace(text)
    text = _merge_short_lines(text)
    # Per-line word dedup then paragraph-level
    paragraphs = []
    for para in text.split("\n\n"):
        para = _remove_consecutive_duplicate_words(para)
        para = re.sub(r"([.!?])\s*\1+", r"\1", para)
        if para.strip():
            paragraphs.append(para.strip())
    return "\n\n".join(paragraphs)


def preview_snippet(text: str, max_chars: int = 220) -> str:
    """Short preview for UI cards; never dumps full chunk."""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"
