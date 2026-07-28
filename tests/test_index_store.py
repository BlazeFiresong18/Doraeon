"""Tests DoraeonIndex's chunking behavior: word-based chunk sizing (not
LlamaIndex's default subword-token sizing) and per-chunk heading detection
that upgrades the unit tag beyond a filename-level guess."""

from llama_index.core import Document

from core.index_store import DoraeonIndex, _detect_topic_heading


def test_chunk_size_is_interpreted_as_words_not_tokens():
    idx = DoraeonIndex()
    text = " ".join(f"word{i}" for i in range(1000))
    doc = Document(text=text, metadata={"filename": "a.pdf", "page_number": 1, "subject": "", "unit": ""})

    idx.add_documents([doc], chunk_size=400, chunk_overlap=0)

    # Not exactly 400: SentenceSplitter still respects sentence/paragraph
    # boundaries even with a word tokenizer, so it can stop a bit short
    # rather than cut mid-sentence. The point of this test is the ballpark --
    # ~400 words, not ~200, which is what the default subword tokenizer
    # would produce for this same chunk_size value.
    first_chunk_word_count = len(idx.nodes[0].get_content().split())
    assert 350 <= first_chunk_word_count <= 400, (
        f"expected ~400 words (word-based sizing), got {first_chunk_word_count} "
        "-- check the tokenizer= override is still wired in"
    )


def test_heading_detected_in_chunk_text_overrides_filename_level_unit_guess():
    idx = DoraeonIndex()
    text = "Unit 5: Graphs\nGraph traversal includes BFS and DFS algorithms for exploring nodes."
    doc = Document(text=text, metadata={"filename": "a.pdf", "page_number": 1, "subject": "CS101", "unit": "Unit1"})

    idx.add_documents([doc], chunk_size=400, chunk_overlap=0)

    assert idx.nodes[0].metadata["unit"] == "Unit 5: Graphs"


def test_no_heading_in_chunk_keeps_filename_level_unit_guess():
    idx = DoraeonIndex()
    text = "This is just regular body text with no heading-like line at all."
    doc = Document(text=text, metadata={"filename": "a.pdf", "page_number": 1, "subject": "CS101", "unit": "Unit1"})

    idx.add_documents([doc], chunk_size=400, chunk_overlap=0)

    assert idx.nodes[0].metadata["unit"] == "Unit1"


def test_detect_topic_heading_matches_common_patterns():
    assert _detect_topic_heading("Unit 2: Recursion\nSome body text") == "Unit 2: Recursion"
    assert _detect_topic_heading("Chapter 5\nMore text") == "Chapter 5"
    assert _detect_topic_heading("Module 3.1 Intro") == "Module 3.1 Intro"


def test_detect_topic_heading_ignores_body_text():
    assert _detect_topic_heading("This is just a regular sentence about recursion.") == ""


def test_clear_resets_index_and_node_tracking():
    idx = DoraeonIndex()
    doc = Document(text="Some content.", metadata={"filename": "a.pdf", "page_number": 1, "subject": "", "unit": ""})
    idx.add_documents([doc], chunk_size=400, chunk_overlap=0)
    assert idx.node_count > 0

    idx.clear()
    assert idx.node_count == 0
    assert idx.document_count == 0
    assert idx.nodes == []
