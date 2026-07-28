"""Split extracted PDF text into overlapping semantic chunks with metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.pdf_loader import PageText


@dataclass
class TextChunk:
    """A retrievable text segment with academic source metadata."""

    text: str
    filename: str
    page_number: int
    subject: str
    unit: str
    chunk_index: int


def _word_count(text: str) -> int:
    return len(text.split())


def _split_into_words(text: str) -> list[str]:
    return text.split()


def _words_to_text(words: list[str]) -> str:
    return " ".join(words)


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[TextChunk]:
    """
    Merge pages per file, then split into word-based chunks.
    chunk_size and overlap are in words (target ~300–500 words).
    """
    if chunk_size <= overlap:
        overlap = max(0, chunk_size // 4)

    # Group consecutive pages by filename to preserve reading order
    by_file: dict[str, list[PageText]] = {}
    for page in pages:
        by_file.setdefault(page.filename, []).append(page)

    chunks: list[TextChunk] = []
    global_index = 0

    for filename, file_pages in by_file.items():
        file_pages.sort(key=lambda p: p.page_number)
        # Track which page each word-span came from for citation
        page_spans: list[tuple[str, int, str, str]] = []  # text, page_num, subject, unit
        for p in file_pages:
            page_spans.append((p.text, p.page_number, p.subject, p.unit))

        full_parts: list[str] = []
        page_map: list[int] = []  # page number per word (approximate)
        # Heading detection must run on each page's ORIGINAL text (newlines
        # intact) before word-flattening -- word.split()/" ".join() discards
        # all line breaks, so running detection on the already-flattened
        # chunk text (as this used to do) collapses every chunk into one
        # giant "line" that's virtually never short enough to match. Detect
        # per-page instead, then look up the primary page's heading below.
        heading_by_page: dict[int, str] = {}
        subject = file_pages[0].subject if file_pages else ""
        unit = file_pages[0].unit if file_pages else ""

        for text, pg, subj, u in page_spans:
            if subj:
                subject = subj
            if u:
                unit = u
            heading_by_page[pg] = _detect_topic_heading(text)
            words = _split_into_words(text)
            full_parts.extend(words)
            page_map.extend([pg] * len(words))

        if not full_parts:
            continue

        start = 0
        while start < len(full_parts):
            end = min(start + chunk_size, len(full_parts))
            window = full_parts[start:end]
            chunk_text = _words_to_text(window)
            # Primary page = mode of pages in this window
            window_pages = page_map[start:end]
            primary_page = max(set(window_pages), key=window_pages.count) if window_pages else 1

            # Use the primary page's detected heading, if any; else inherit
            # the running subject/unit guess.
            detected_unit = heading_by_page.get(primary_page, "") or unit

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    filename=filename,
                    page_number=primary_page,
                    subject=subject,
                    unit=detected_unit,
                    chunk_index=global_index,
                )
            )
            global_index += 1
            if end >= len(full_parts):
                break
            start = end - overlap

    return chunks


def _detect_topic_heading(text: str) -> str:
    """Pick short lines that look like section headings (Unit 2, Chapter 3, etc.)."""
    first_lines = text.strip().split("\n")[:3]
    pattern = re.compile(
        r"^(unit|chapter|ch\.?|module|topic|lecture)\s*[\d.:]+",
        re.IGNORECASE,
    )
    for line in first_lines:
        line = line.strip()
        if len(line) < 80 and pattern.match(line):
            return line
    return ""
