from tests._pdf_helpers import make_minimal_pdf
from core.ingestion import load_multiple_pdfs, load_pdf_documents


def test_extracts_text_and_metadata_from_real_pdf():
    pdf_bytes = make_minimal_pdf(
        ["Unit 2: Algorithms", "Binary search runs in O(log n) time.", "It requires a sorted array."]
    )
    docs = load_pdf_documents(pdf_bytes, "CS101_Unit2_Lecture.pdf")

    assert len(docs) == 1
    doc = docs[0]
    assert "Binary search" in doc.text
    assert doc.metadata["filename"] == "CS101_Unit2_Lecture.pdf"
    assert doc.metadata["page_number"] == 1
    assert doc.metadata["subject"] == "CS101"
    assert doc.metadata["unit"] == "Unit2"


def test_explicit_subject_unit_override_filename_guess():
    pdf_bytes = make_minimal_pdf(["Some content."])
    docs = load_pdf_documents(pdf_bytes, "random.pdf", subject="MATH201", unit="Week3")

    assert docs[0].metadata["subject"] == "MATH201"
    assert docs[0].metadata["unit"] == "Week3"


def test_metadata_excluded_from_embedding_text():
    pdf_bytes = make_minimal_pdf(["Some academic content."])
    docs = load_pdf_documents(pdf_bytes, "CS101_Unit2.pdf")
    doc = docs[0]

    embed_text = doc.get_content(metadata_mode="embed")
    assert "CS101_Unit2.pdf" not in embed_text
    assert "page_number" not in embed_text.lower()
    assert "Some academic content." in embed_text


def test_metadata_available_for_llm_context():
    pdf_bytes = make_minimal_pdf(["Some academic content."])
    docs = load_pdf_documents(pdf_bytes, "CS101_Unit2.pdf")
    doc = docs[0]

    llm_text = doc.get_content(metadata_mode="llm")
    assert "CS101_Unit2.pdf" in llm_text


def test_load_multiple_pdfs_combines_all_files():
    pdf1 = make_minimal_pdf(["Content of file one."])
    pdf2 = make_minimal_pdf(["Content of file two."])
    docs = load_multiple_pdfs([(pdf1, "a.pdf"), (pdf2, "b.pdf")])

    assert len(docs) == 2
    assert {d.metadata["filename"] for d in docs} == {"a.pdf", "b.pdf"}


def test_empty_pdf_produces_no_documents():
    pdf_bytes = make_minimal_pdf([])
    docs = load_pdf_documents(pdf_bytes, "empty.pdf")
    assert docs == []
