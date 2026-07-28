"""Extract text from academic PDFs into LlamaIndex Documents, one per page,
carrying source metadata (filename, page number, subject, unit) that survives
chunking into Nodes -- this is what source attribution reads back later."""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import pdfplumber
from llama_index.core import Document
from pypdf import PdfReader

from core.text_cleaner import clean_pdf_text


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


def _extract_with_pdfplumber(file_obj: BinaryIO) -> list[str]:
    """Primary extractor (better layout preservation). Returns raw per-page text."""
    pages: list[str] = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def _extract_with_pypdf(file_obj: BinaryIO) -> list[str]:
    """Fallback extractor when pdfplumber fails."""
    reader = PdfReader(file_obj)
    return [page.extract_text() or "" for page in reader.pages]


def load_pdf_documents(
    data: bytes,
    filename: str,
    subject: str = "",
    unit: str = "",
) -> list[Document]:
    """Load one PDF's bytes into a list of Documents, one per non-empty page."""
    guessed_subject, guessed_unit = _guess_subject_unit(filename)
    subj = subject or guessed_subject
    u = unit or guessed_unit

    buffer = BytesIO(data)
    try:
        buffer.seek(0)
        raw_pages = _extract_with_pdfplumber(buffer)
        if not any(p.strip() for p in raw_pages):
            raise ValueError("pdfplumber extracted no text")
    except Exception:
        buffer.seek(0)
        raw_pages = _extract_with_pypdf(buffer)

    documents: list[Document] = []
    for i, raw_text in enumerate(raw_pages, start=1):
        text = clean_pdf_text(raw_text)
        if not text.strip():
            continue
        documents.append(
            Document(
                text=text,
                metadata={
                    "filename": filename,
                    "page_number": i,
                    "subject": subj,
                    "unit": u,
                },
                # Metadata is needed for LLM citations but must NOT be folded
                # into the embedded text -- LlamaIndex prepends visible
                # metadata to the text before embedding by default, which
                # would inject filename/page boilerplate into every chunk's
                # vector and skew similarity scores in an uncontrolled way.
                # Embed purely on the actual content, same as the prior
                # implementation (which only ever embedded `chunk.text`).
                excluded_llm_metadata_keys=[],
                excluded_embed_metadata_keys=["filename", "page_number", "subject", "unit"],
            )
        )
    return documents


def load_multiple_pdfs(
    files: list[tuple[bytes, str]],
    default_subject: str = "",
    default_unit: str = "",
) -> list[Document]:
    """Load several uploaded PDFs; each item is (bytes, filename)."""
    all_documents: list[Document] = []
    for data, name in files:
        all_documents.extend(load_pdf_documents(data, name, default_subject, default_unit))
    return all_documents
