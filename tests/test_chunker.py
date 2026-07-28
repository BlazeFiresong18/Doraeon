from utils.chunker import _detect_topic_heading, chunk_pages
from utils.pdf_loader import PageText


def _page(text: str, page_number: int, filename: str = "doc.pdf", subject: str = "CS101", unit: str = "") -> PageText:
    return PageText(filename=filename, page_number=page_number, text=text, subject=subject, unit=unit)


def test_single_short_page_becomes_one_chunk():
    pages = [_page("one two three four five", 1)]
    chunks = chunk_pages(pages, chunk_size=400, overlap=80)
    assert len(chunks) == 1
    assert chunks[0].text == "one two three four five"
    assert chunks[0].page_number == 1
    assert chunks[0].subject == "CS101"


def test_long_text_splits_into_multiple_overlapping_chunks():
    words = [f"word{i}" for i in range(1000)]
    pages = [_page(" ".join(words), 1)]
    overlap = 80
    chunks = chunk_pages(pages, chunk_size=400, overlap=overlap)
    assert len(chunks) > 1
    # Chunk 1 starts `overlap` words before chunk 0 ended -- that exact
    # trailing slice of chunk 0 should equal the leading slice of chunk 1.
    assert chunks[0].text.split()[-overlap:] == chunks[1].text.split()[:overlap]


def test_chunks_preserve_reading_order_across_pages():
    pages = [_page("alpha beta gamma", 1), _page("delta epsilon zeta", 2)]
    chunks = chunk_pages(pages, chunk_size=400, overlap=0)
    assert chunks[0].text.split() == ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def test_primary_page_is_mode_of_window():
    # First page contributes 2 words, second page contributes 8 -- window should attribute to page 2
    pages = [_page("a b", 1), _page("c d e f g h i j", 2)]
    chunks = chunk_pages(pages, chunk_size=400, overlap=0)
    assert chunks[0].page_number == 2


def test_files_grouped_and_chunked_independently():
    pages = [_page("file one content here", 1, filename="a.pdf"), _page("file two content here", 1, filename="b.pdf")]
    chunks = chunk_pages(pages, chunk_size=400, overlap=0)
    filenames = {c.filename for c in chunks}
    assert filenames == {"a.pdf", "b.pdf"}


def test_empty_pages_list_returns_no_chunks():
    assert chunk_pages([], chunk_size=400, overlap=80) == []


def test_overlap_larger_than_chunk_size_is_clamped_not_infinite_loop():
    words = [f"w{i}" for i in range(50)]
    pages = [_page(" ".join(words), 1)]
    # overlap >= chunk_size would otherwise never advance `start`
    chunks = chunk_pages(pages, chunk_size=10, overlap=10)
    assert len(chunks) > 0  # completes without hanging


def test_chunk_index_increments_globally_across_files():
    pages = [_page("a b c d e f g h", 1, filename="a.pdf"), _page("i j k l m n o p", 1, filename="b.pdf")]
    chunks = chunk_pages(pages, chunk_size=400, overlap=0)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_detect_topic_heading_matches_common_patterns():
    assert _detect_topic_heading("Unit 2: Recursion\nSome body text") == "Unit 2: Recursion"
    assert _detect_topic_heading("Chapter 5\nMore text") == "Chapter 5"
    assert _detect_topic_heading("Module 3.1 Intro") == "Module 3.1 Intro"


def test_detect_topic_heading_ignores_body_text():
    assert _detect_topic_heading("This is just a regular sentence about recursion.") == ""


def test_detected_heading_overrides_page_level_unit():
    pages = [_page("Unit 5: Graphs\nBody text about graphs here", 1, unit="Unit1")]
    chunks = chunk_pages(pages, chunk_size=400, overlap=0)
    assert chunks[0].unit == "Unit 5: Graphs"
