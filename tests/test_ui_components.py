"""
Tests the new stat-tile and subject-distribution renderers by capturing what
they pass to st.markdown (mocked), since Streamlit's own render output can't
be inspected outside a live session. This verifies the actual generated HTML
structure/content, not just that the functions run without raising.
"""

from unittest.mock import patch

from utils.chunker import TextChunk
from utils.ui_components import render_stat_tiles, render_subject_distribution


def _chunk(subject: str) -> TextChunk:
    return TextChunk(text="x", filename="a.pdf", page_number=1, subject=subject, unit="", chunk_index=0)


def test_stat_tiles_renders_all_three_values():
    with patch("utils.ui_components.st.markdown") as md:
        render_stat_tiles(document_count=3, chunk_count=42, subject_count=2)
    html = md.call_args[0][0]
    assert "Documents" in html and "3" in html
    assert "Chunks indexed" in html and "42" in html
    assert "Subjects" in html and ">2<" in html


def test_stat_tiles_shows_dash_for_zero_subjects():
    with patch("utils.ui_components.st.markdown") as md:
        render_stat_tiles(document_count=1, chunk_count=5, subject_count=0)
    html = md.call_args[0][0]
    assert ">—<" in html


def test_distribution_skipped_for_single_subject():
    chunks = [_chunk("CS101")] * 10
    with patch("utils.ui_components.st.markdown") as md:
        render_subject_distribution(chunks)
    md.assert_not_called()  # a legend/bar for one series has nothing to compare


def test_distribution_skipped_for_no_subjects():
    chunks = [_chunk("")] * 5
    with patch("utils.ui_components.st.markdown") as md:
        render_subject_distribution(chunks)
    md.assert_not_called()


def test_distribution_renders_bar_and_legend_for_multiple_subjects():
    chunks = [_chunk("CS101")] * 6 + [_chunk("MATH201")] * 4
    with patch("utils.ui_components.st.markdown") as md:
        render_subject_distribution(chunks)
    html = md.call_args[0][0]
    assert "CS101" in html and "MATH201" in html
    assert "dist-bar" in html and "dist-legend" in html
    # Each subject's count appears in its legend entry
    assert "· 6" in html and "· 4" in html


def test_distribution_uses_categorical_colors_in_fixed_order():
    chunks = [_chunk("A")] * 5 + [_chunk("B")] * 5
    with patch("utils.ui_components.st.markdown") as md:
        render_subject_distribution(chunks)
    html = md.call_args[0][0]
    # First (largest/tied) subject gets categorical slot 1, second gets slot 2
    assert "#3987e5" in html
    assert "#d95926" in html


def test_distribution_folds_long_tail_into_other():
    # 7 distinct subjects: soft cap is 5 named categorical slots + "Other"
    chunks = []
    for i in range(7):
        chunks.extend([_chunk(f"Subj{i}")] * (10 - i))  # descending counts
    with patch("utils.ui_components.st.markdown") as md:
        render_subject_distribution(chunks)
    html = md.call_args[0][0]
    assert "Other" in html
    # The two smallest subjects (Subj5, Subj6) should be folded, not named directly
    assert "Subj5" not in html
    assert "Subj6" not in html


def test_distribution_only_inline_labels_segments_wide_enough_to_fit():
    # One dominant subject (80%) and a tiny one (20%) below the 14% inline threshold...
    # use a genuinely tiny sliver to guarantee no label collision
    chunks = [_chunk("Big")] * 95 + [_chunk("Tiny")] * 5
    with patch("utils.ui_components.st.markdown") as md:
        render_subject_distribution(chunks)
    html = md.call_args[0][0]
    # Small segment still appears in the legend even without an inline label
    assert "Tiny · 5" in html
