"""Extract text and page metadata from academic PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import pdfplumber
from pypdf import PdfReader

from utils.text_cleaner import clean_pdf_text


@dataclass
class PageText:
    """Single page of extracted text with source metadata."""

    filename: str
    page_number: int
    text: str
    subject: str = ""
    unit: str = ""


def _guess_subject_unit(filename: str) -> tuple[str, str]:
    """
    Heuristic: infer subject/unit from filename tokens.
    e.g. 'CS101_Unit2_Lecture.pdf' -> subject=CS101, unit=Unit2
    """
    base = filename.rsplit(".", 1)[0]
    parts = [p for p in base.replace("-", "_").split("_") if p]
    subject = parts[0] if parts else ""
    unit = ""
    for part in parts[1:]:
        lower = part.lower()
        if lower.startswith("unit") or lower.startswith("ch") or lower.startswith("module"):
            unit = part
            break
    return subject, unit


def extract_with_pdfplumber(
    file_obj: BinaryIO,
    filename: str,
    subject: str = "",
    unit: str = "",
) -> list[PageText]:
    """Primary extractor using pdfplumber (better layout preservation)."""
    pages: list[PageText] = []
    guessed_subject, guessed_unit = _guess_subject_unit(filename)
    subj = subject or guessed_subject
    u = unit or guessed_unit

    with pdfplumber.open(file_obj) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = clean_pdf_text(page.extract_text() or "")
            if text.strip():
                pages.append(
                    PageText(
                        filename=filename,
                        page_number=i,
                        text=text,
                        subject=subj,
                        unit=u,
                    )
                )
    return pages


def extract_with_pypdf(
    file_obj: BinaryIO,
    filename: str,
    subject: str = "",
    unit: str = "",
) -> list[PageText]:
    """Fallback extractor when pdfplumber fails."""
    pages: list[PageText] = []
    guessed_subject, guessed_unit = _guess_subject_unit(filename)
    subj = subject or guessed_subject
    u = unit or guessed_unit

    reader = PdfReader(file_obj)
    for i, page in enumerate(reader.pages, start=1):
        text = clean_pdf_text(page.extract_text() or "")
        if text.strip():
            pages.append(
                PageText(
                    filename=filename,
                    page_number=i,
                    text=text,
                    subject=subj,
                    unit=u,
                )
            )
    return pages


def load_pdf_from_bytes(
    data: bytes,
    filename: str,
    subject: str = "",
    unit: str = "",
) -> list[PageText]:
    """
    Load one PDF from raw bytes. Tries pdfplumber first, then PyPDF.
    """
    buffer = BytesIO(data)
    try:
        buffer.seek(0)
        pages = extract_with_pdfplumber(buffer, filename, subject, unit)
        if pages:
            return pages
    except Exception:
        pass

    buffer.seek(0)
    return extract_with_pypdf(buffer, filename, subject, unit)


def load_multiple_pdfs(
    files: list[tuple[bytes, str]],
    default_subject: str = "",
    default_unit: str = "",
) -> list[PageText]:
    """Load several uploaded PDFs; each item is (bytes, filename)."""
    all_pages: list[PageText] = []
    for data, name in files:
        pages = load_pdf_from_bytes(data, name, default_subject, default_unit)
        all_pages.extend(pages)
    return all_pages
