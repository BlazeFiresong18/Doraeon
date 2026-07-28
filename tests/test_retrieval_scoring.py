"""
Retrieval + confidence scoring tests using the REAL embedding model (not a
fake/mock) -- this is the Phase 2 checkpoint the rebuild explicitly required:
verify with a real query that a clearly-matching document scores
appropriately high, not negative, not near-zero, before building anything
on top of it. Slower than the rest of the suite (loads a real model) but
that's the point -- this is the one area a synthetic vector test can't
actually validate.
"""

from llama_index.core import Document

from core.index_store import DoraeonIndex
from core.retrieval import normalize_score, retrieve

WEB_ANALYTICS_TEXT = """Web analytics is the measurement, collection, analysis, and reporting of
web data to understand and optimize web usage. Key metrics include page views, bounce rate,
session duration, and conversion rate. Google Analytics is the most widely used web analytics
tool, allowing businesses to track visitor behavior, traffic sources, and campaign performance."""


def _index_with_web_analytics_doc(subject: str = "MKTG201", unit: str = "Unit 3") -> DoraeonIndex:
    idx = DoraeonIndex()
    doc = Document(text=WEB_ANALYTICS_TEXT, metadata={
        "filename": "Lecture5.pdf", "page_number": 2, "subject": subject, "unit": unit,
    })
    idx.add_documents([doc], chunk_size=400, chunk_overlap=80)
    return idx


def test_direct_match_scores_high_not_negative_not_near_zero():
    idx = _index_with_web_analytics_doc()
    results = retrieve(idx, "What is web analytics?", top_k=3)

    assert results
    top = results[0]
    assert top.raw_score > 0.5, f"expected a strong match, got raw_score={top.raw_score}"
    assert top.confidence > 0.5
    assert top.confidence <= 1.0


def test_unrelated_query_never_shows_negative_confidence():
    idx = _index_with_web_analytics_doc()
    results = retrieve(idx, "What is the boiling point of water?", top_k=3)

    assert results
    for r in results:
        assert r.confidence >= 0.0, f"confidence must never be negative, got {r.confidence}"
        # The raw score itself CAN legitimately be slightly negative (cosine
        # similarity's real range) -- only the normalized display value is clamped.
        assert r.confidence == max(0.0, r.raw_score)


def test_unrelated_query_scores_much_lower_than_direct_match():
    idx = _index_with_web_analytics_doc()
    direct = retrieve(idx, "What is web analytics?", top_k=1)[0]
    unrelated = retrieve(idx, "What is the boiling point of water?", top_k=1)[0]

    assert direct.raw_score > unrelated.raw_score
    assert direct.raw_score - unrelated.raw_score > 0.4  # a clear, meaningful gap


def test_metadata_propagates_to_scored_chunks():
    idx = _index_with_web_analytics_doc(subject="MKTG201", unit="Unit 3")
    results = retrieve(idx, "What is web analytics?", top_k=1)

    chunk = results[0]
    assert chunk.filename == "Lecture5.pdf"
    assert chunk.page_number == 2
    assert chunk.subject == "MKTG201"
    assert chunk.unit == "Unit 3"


def test_subject_filter_excludes_non_matching_documents():
    idx = DoraeonIndex()
    idx.add_documents(
        [
            Document(text=WEB_ANALYTICS_TEXT, metadata={"filename": "a.pdf", "page_number": 1, "subject": "MKTG201", "unit": ""}),
            Document(text="Binary search runs in O(log n) time on a sorted array.", metadata={"filename": "b.pdf", "page_number": 1, "subject": "CS101", "unit": ""}),
        ],
        chunk_size=400,
        chunk_overlap=80,
    )

    results = retrieve(idx, "What is web analytics?", top_k=5, subject_filter="CS101")
    assert all(r.subject == "CS101" for r in results)


def test_top_k_limits_result_count():
    idx = DoraeonIndex()
    docs = [
        Document(
            text=f"{WEB_ANALYTICS_TEXT} Section {i}.",
            metadata={"filename": f"doc{i}.pdf", "page_number": 1, "subject": "MKTG201", "unit": ""},
        )
        for i in range(5)
    ]
    idx.add_documents(docs, chunk_size=400, chunk_overlap=80)

    results = retrieve(idx, "What is web analytics?", top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# normalize_score -- pure function, no model needed
# ---------------------------------------------------------------------------

def test_normalize_score_clamps_negative_to_zero():
    assert normalize_score(-0.05) == 0.0
    assert normalize_score(-1.0) == 0.0


def test_normalize_score_passes_through_non_negative_unchanged():
    assert normalize_score(0.0) == 0.0
    assert normalize_score(0.42) == 0.42
    assert normalize_score(1.0) == 1.0
