from unittest.mock import patch

from tests._pdf_helpers import make_minimal_pdf
from core.ingestion import PageExtractionIssue, _looks_like_real_text, load_multiple_pdfs, load_pdf_documents


def test_extracts_text_and_metadata_from_real_pdf():
    pdf_bytes = make_minimal_pdf(
        ["Unit 2: Algorithms", "Binary search runs in O(log n) time.", "It requires a sorted array."]
    )
    docs, issues = load_pdf_documents(pdf_bytes, "CS101_Unit2_Lecture.pdf")

    assert issues == []
    assert len(docs) == 1
    doc = docs[0]
    assert "Binary search" in doc.text
    assert doc.metadata["filename"] == "CS101_Unit2_Lecture.pdf"
    assert doc.metadata["page_number"] == 1
    assert doc.metadata["subject"] == "CS101"
    assert doc.metadata["unit"] == "Unit2"


def test_explicit_subject_unit_override_filename_guess():
    pdf_bytes = make_minimal_pdf(["Some content."])
    docs, _issues = load_pdf_documents(pdf_bytes, "random.pdf", subject="MATH201", unit="Week3")

    assert docs[0].metadata["subject"] == "MATH201"
    assert docs[0].metadata["unit"] == "Week3"


def test_metadata_excluded_from_embedding_text():
    pdf_bytes = make_minimal_pdf(["Some academic content."])
    docs, _issues = load_pdf_documents(pdf_bytes, "CS101_Unit2.pdf")
    doc = docs[0]

    embed_text = doc.get_content(metadata_mode="embed")
    assert "CS101_Unit2.pdf" not in embed_text
    assert "page_number" not in embed_text.lower()
    assert "Some academic content." in embed_text


def test_metadata_available_for_llm_context():
    pdf_bytes = make_minimal_pdf(["Some academic content."])
    docs, _issues = load_pdf_documents(pdf_bytes, "CS101_Unit2.pdf")
    doc = docs[0]

    llm_text = doc.get_content(metadata_mode="llm")
    assert "CS101_Unit2.pdf" in llm_text


def test_load_multiple_pdfs_combines_all_files():
    pdf1 = make_minimal_pdf(["Content of file one."])
    pdf2 = make_minimal_pdf(["Content of file two."])
    docs, issues = load_multiple_pdfs([(pdf1, "a.pdf"), (pdf2, "b.pdf")])

    assert issues == []
    assert len(docs) == 2
    assert {d.metadata["filename"] for d in docs} == {"a.pdf", "b.pdf"}


def test_empty_pdf_produces_no_documents_and_no_issues():
    # A page with literally zero text (rather than garbled text) isn't an
    # extraction failure -- it's just blank, so it's silently skipped, same
    # as before this fix, not reported as an issue.
    pdf_bytes = make_minimal_pdf([])
    docs, issues = load_pdf_documents(pdf_bytes, "empty.pdf")
    assert docs == []
    assert issues == []


# ---------------------------------------------------------------------------
# _looks_like_real_text -- the garbage detector, tested against the two real
# failure modes confirmed against an actual PDF (see conversation/commit
# history): pdfplumber's "(cid:N)" placeholders and pypdf's raw control bytes.
# ---------------------------------------------------------------------------

def test_looks_like_real_text_accepts_normal_english():
    assert _looks_like_real_text("Web analytics is the measurement and analysis of web data.")


def test_looks_like_real_text_rejects_cid_placeholder_garbage():
    garbled = "(cid:1)(cid:24)(cid:2)(cid:25)(cid:5)(cid:24)(cid:10)(cid:6)(cid:5)(cid:26)"
    assert not _looks_like_real_text(garbled)


def test_looks_like_real_text_rejects_control_character_garbage():
    garbled = "\x18\x02\x19\x05\x18\n\x06\x05\x1a\x13\x0c\x0f\x19\r\x03\x0b\x04\x05\x1b"
    assert not _looks_like_real_text(garbled)


def test_looks_like_real_text_rejects_empty_string():
    assert not _looks_like_real_text("")
    assert not _looks_like_real_text("   ")


def test_looks_like_real_text_rejects_mostly_garbage_with_a_little_real_text():
    # A single readable phrase buried in mostly control-character noise --
    # confirmed real-world case (pypdf on a page of the actual failing PDF).
    mostly_garbage = "\x1f \n\x12 \x03\n\x15\x05\x07\x1d\x05\x18\n\x06\x05\x1a\x13Web Server Performance Data\x11\t\x14\x01 \x16\n\x14\x06\x16\x14"
    assert not _looks_like_real_text(mostly_garbage)


def test_page_garbled_in_both_extractors_is_skipped_and_reported():
    # Can't easily fabricate a real broken-font PDF in a unit test (that
    # requires an actual font missing its ToUnicode CMap) -- so the two
    # low-level extractors are mocked to reproduce exactly what was observed
    # against the real failing file: pdfplumber's "(cid:N)" placeholders and
    # pypdf's control-character garbage, for the same page.
    with patch("core.ingestion._extract_with_pdfplumber", return_value=["(cid:1)(cid:24)(cid:2)"]), \
         patch("core.ingestion._extract_with_pypdf", return_value=["\x18\x02\x19\x05\x18\x06\x05"]):
        docs, issues = load_pdf_documents(b"fake-pdf-bytes", "broken_font.pdf")

    assert docs == []
    assert len(issues) == 1
    assert issues[0] == PageExtractionIssue(
        filename="broken_font.pdf", page_number=1, reason=issues[0].reason
    )
    assert "readable text" in issues[0].reason


def test_page_readable_via_pypdf_fallback_when_pdfplumber_fails():
    # Confirmed real-world case: some pages of the actual failing PDF
    # extracted as garbage via pdfplumber but were fine via pypdf.
    with patch("core.ingestion._extract_with_pdfplumber", return_value=["(cid:1)(cid:24)(cid:2)"]), \
         patch("core.ingestion._extract_with_pypdf", return_value=["This page reads fine via pypdf."]):
        docs, issues = load_pdf_documents(b"fake-pdf-bytes", "mixed_fonts.pdf")

    assert issues == []
    assert len(docs) == 1
    assert "This page reads fine via pypdf" in docs[0].text


# ---------------------------------------------------------------------------
# OCR fallback -- triggers only when both text extractors fail (broken font
# garbage OR a scanned/image page with no text layer). pytesseract and the
# page rasterization are mocked: the Tesseract binary isn't a test dependency.
# ---------------------------------------------------------------------------

def test_ocr_recovers_page_that_both_extractors_garbled():
    with patch("core.ingestion._extract_with_pdfplumber", return_value=["(cid:1)(cid:24)(cid:2)"]), \
         patch("core.ingestion._extract_with_pypdf", return_value=["\x18\x02\x19\x05"]), \
         patch("core.ingestion._ocr_available", return_value=True), \
         patch("core.ingestion._ocr_pdf_page", return_value="Recovered by OCR: web analytics basics.") as mock_ocr:
        docs, issues = load_pdf_documents(b"fake-pdf-bytes", "broken_font.pdf")

    assert issues == []
    assert len(docs) == 1
    assert "Recovered by OCR" in docs[0].text
    mock_ocr.assert_called_once_with(b"fake-pdf-bytes", 0)


def test_ocr_recovers_scanned_page_with_no_text_layer():
    # A scanned page: both extractors return EMPTY (not garbled) text.
    # Previously silently skipped -- with OCR available it should be read.
    with patch("core.ingestion._extract_with_pdfplumber", return_value=[""]), \
         patch("core.ingestion._extract_with_pypdf", return_value=[""]), \
         patch("core.ingestion._ocr_available", return_value=True), \
         patch("core.ingestion._ocr_pdf_page", return_value="Scanned page text found by OCR."):
        docs, issues = load_pdf_documents(b"fake-pdf-bytes", "scanned.pdf")

    assert issues == []
    assert len(docs) == 1
    assert "Scanned page text" in docs[0].text


def test_ocr_not_attempted_when_normal_extraction_works():
    with patch("core.ingestion._ocr_pdf_page") as mock_ocr:
        pdf_bytes = make_minimal_pdf(["Perfectly normal text page."])
        docs, issues = load_pdf_documents(pdf_bytes, "normal.pdf")

    assert len(docs) == 1
    assert not mock_ocr.called  # zero OCR cost on healthy PDFs


def test_missing_tesseract_reports_issue_with_install_hint():
    with patch("core.ingestion._extract_with_pdfplumber", return_value=["(cid:1)(cid:24)(cid:2)"]), \
         patch("core.ingestion._extract_with_pypdf", return_value=["\x18\x02\x19\x05"]), \
         patch("core.ingestion._ocr_available", return_value=False):
        docs, issues = load_pdf_documents(b"fake-pdf-bytes", "broken_font.pdf")

    assert docs == []
    assert len(issues) == 1
    assert "install Tesseract" in issues[0].reason


def test_ocr_failure_still_reports_issue_not_garbage_index():
    with patch("core.ingestion._extract_with_pdfplumber", return_value=["(cid:1)(cid:24)(cid:2)"]), \
         patch("core.ingestion._extract_with_pypdf", return_value=["\x18\x02\x19\x05"]), \
         patch("core.ingestion._ocr_available", return_value=True), \
         patch("core.ingestion._ocr_pdf_page", return_value=""):
        docs, issues = load_pdf_documents(b"fake-pdf-bytes", "broken_font.pdf")

    assert docs == []
    assert len(issues) == 1
    assert "OCR also failed" in issues[0].reason


def test_blank_page_stays_silently_skipped_even_with_ocr_available():
    with patch("core.ingestion._extract_with_pdfplumber", return_value=[""]), \
         patch("core.ingestion._extract_with_pypdf", return_value=[""]), \
         patch("core.ingestion._ocr_available", return_value=True), \
         patch("core.ingestion._ocr_pdf_page", return_value=""):
        docs, issues = load_pdf_documents(b"fake-pdf-bytes", "blank.pdf")

    assert docs == []
    assert issues == []  # truly blank is not a failure


def test_mixed_document_indexes_good_pages_and_reports_bad_ones():
    # Page 1 clean via pdfplumber, page 2 garbled in both extractors.
    with patch(
        "core.ingestion._extract_with_pdfplumber",
        return_value=["A perfectly normal readable page.", "(cid:1)(cid:24)(cid:2)"],
    ), patch(
        "core.ingestion._extract_with_pypdf",
        return_value=["A perfectly normal readable page.", "\x18\x02\x19\x05\x18\x06\x05"],
    ):
        docs, issues = load_pdf_documents(b"fake-pdf-bytes", "mixed.pdf")

    assert len(docs) == 1
    assert docs[0].metadata["page_number"] == 1
    assert len(issues) == 1
    assert issues[0].page_number == 2
