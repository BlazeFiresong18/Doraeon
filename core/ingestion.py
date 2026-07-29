"""Extract text from academic PDFs into LlamaIndex Documents, one per page,
carrying source metadata (filename, page number, subject, unit) that survives
chunking into Nodes -- this is what source attribution reads back later.

Per-page quality checking, not whole-document fallback: a real PDF (confirmed
against "Web Analytics Demystified") can have one font work fine on some
pages and fail on others -- pdfplumber emitted literal "(cid:N)" glyph-ID
placeholders on every page of that file, while pypdf produced raw control
bytes on most pages but happened to decode a couple of pages correctly.
Neither failure mode is "no text" (empty), so a whole-document
succeeded/failed check never catches it -- each page is checked and each
extractor tried independently instead.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO, Callable

import pdfplumber
from llama_index.core import Document
from pypdf import PdfReader

from core.text_cleaner import clean_pdf_text


@dataclass
class PageExtractionIssue:
    """A page that neither extractor could turn into readable text."""

    filename: str
    page_number: int
    reason: str


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


def _looks_like_real_text(text: str, min_printable_ratio: float = 0.85) -> bool:
    """Distinguishes genuinely-extracted language from two known PDF
    extraction failure modes, both caused by a font with no proper
    ToUnicode character mapping (confirmed against a real file):

    - pdfplumber emits literal "(cid:N)" glyph-ID placeholders instead of
      decoded characters.
    - pypdf emits raw control-character bytes instead of text.

    Neither is empty text, so a plain "did we get anything back" check
    doesn't catch either. This checks whether what came back is actually
    printable language, not just non-empty.
    """
    if not text or not text.strip():
        return False
    if "(cid:" in text:
        return False
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    return (printable / len(text)) >= min_printable_ratio


def _extract_with_pdfplumber(file_obj: BinaryIO) -> list[str]:
    """Returns raw per-page text (may include garbled/undecoded pages)."""
    pages: list[str] = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def _extract_with_pypdf(file_obj: BinaryIO) -> list[str]:
    """Returns raw per-page text (may include garbled/undecoded pages)."""
    reader = PdfReader(file_obj)
    return [page.extract_text() or "" for page in reader.pages]


# Standard Windows install locations, tried when tesseract isn't on PATH --
# a fresh install doesn't update the PATH of already-running processes, so
# "installed but this process can't see it yet" is a common state.
_TESSERACT_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


@functools.cache
def _ocr_available() -> bool:
    """OCR needs the Tesseract binary installed system-wide, not just the
    pytesseract wrapper -- probe once per process and cache the answer."""
    try:
        import pytesseract
    except Exception:
        return False

    import os

    candidates = [None] + [p for p in _TESSERACT_FALLBACK_PATHS if os.path.isfile(p)]
    for candidate in candidates:
        if candidate is not None:
            pytesseract.pytesseract.tesseract_cmd = candidate
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            continue
    return False


def _ocr_pdf_page(data: bytes, page_index: int) -> str:
    """Rasterize one page at ~230 DPI and run Tesseract on the image.
    Returns "" on any failure -- callers treat that the same as an
    unreadable page."""
    try:
        import pypdfium2 as pdfium
        import pytesseract

        pdf = pdfium.PdfDocument(data)
        try:
            page = pdf[page_index]
            bitmap = page.render(scale=3.2)
            image = bitmap.to_pil()
        finally:
            pdf.close()
        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""


def load_pdf_documents(
    data: bytes,
    filename: str,
    subject: str = "",
    unit: str = "",
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[Document], list[PageExtractionIssue]]:
    """Load one PDF's bytes into a list of Documents (one per readable page)
    plus a list of pages that couldn't be read by either extractor."""
    guessed_subject, guessed_unit = _guess_subject_unit(filename)
    subj = subject or guessed_subject
    u = unit or guessed_unit

    buffer = BytesIO(data)
    try:
        buffer.seek(0)
        plumber_pages = _extract_with_pdfplumber(buffer)
    except Exception:
        plumber_pages = []

    try:
        buffer.seek(0)
        pypdf_pages = _extract_with_pypdf(buffer)
    except Exception:
        pypdf_pages = []

    n_pages = max(len(plumber_pages), len(pypdf_pages))
    documents: list[Document] = []
    issues: list[PageExtractionIssue] = []

    for i in range(n_pages):
        plumber_text = plumber_pages[i] if i < len(plumber_pages) else ""
        pypdf_text = pypdf_pages[i] if i < len(pypdf_pages) else ""

        if _looks_like_real_text(plumber_text):
            raw_text = plumber_text
        elif _looks_like_real_text(pypdf_text):
            raw_text = pypdf_text
        else:
            # Both extractors failed: either a broken font (garbled output) or
            # a scanned/image page (empty output). OCR the rendered page image
            # as a last resort -- it reads the pixels, so it works in both cases.
            # OCR is ~100x slower than text extraction, so report progress:
            # a document full of failing pages would otherwise look frozen.
            if _ocr_available():
                if on_progress:
                    on_progress(f"OCR: {filename} page {i + 1}/{n_pages}…")
                ocr_text = _ocr_pdf_page(data, i)
            else:
                ocr_text = ""
            if _looks_like_real_text(ocr_text):
                raw_text = ocr_text
            elif not plumber_text.strip() and not pypdf_text.strip() and not ocr_text.strip():
                # Genuinely blank page (e.g. a separator page) -- not a failure,
                # nothing to report, just nothing to index.
                continue
            else:
                hint = (
                    "OCR also failed to produce readable text"
                    if _ocr_available()
                    else "install Tesseract OCR to recover pages like this"
                )
                issues.append(
                    PageExtractionIssue(
                        filename=filename,
                        page_number=i + 1,
                        reason="Neither extractor produced readable text (likely an unusual or "
                        f"embedded font without a proper character mapping; {hint}) -- "
                        "page skipped, not indexed as garbage.",
                    )
                )
                continue

        text = clean_pdf_text(raw_text)
        if not text.strip():
            continue
        documents.append(
            Document(
                text=text,
                metadata={
                    "filename": filename,
                    "page_number": i + 1,
                    "subject": subj,
                    "unit": u,
                },
                # Metadata is needed for LLM citations but must NOT be folded
                # into the embedded text -- LlamaIndex prepends visible
                # metadata to the text before embedding by default, which
                # would inject filename/page boilerplate into every chunk's
                # vector and skew similarity scores in an uncontrolled way.
                excluded_llm_metadata_keys=[],
                excluded_embed_metadata_keys=["filename", "page_number", "subject", "unit"],
            )
        )
    return documents, issues


def load_multiple_pdfs(
    files: list[tuple[bytes, str]],
    default_subject: str = "",
    default_unit: str = "",
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[Document], list[PageExtractionIssue]]:
    """Load several uploaded PDFs; each item is (bytes, filename)."""
    all_documents: list[Document] = []
    all_issues: list[PageExtractionIssue] = []
    for data, name in files:
        docs, issues = load_pdf_documents(data, name, default_subject, default_unit, on_progress=on_progress)
        all_documents.extend(docs)
        all_issues.extend(issues)
    return all_documents, all_issues
