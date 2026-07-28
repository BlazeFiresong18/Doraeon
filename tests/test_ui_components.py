"""
Tests the stat-tile and subject-distribution renderers by capturing what
they pass to st.markdown (mocked), since Streamlit's own render output can't
be inspected outside a live session.
"""

from unittest.mock import patch

from core.ui_components import render_stat_tiles, render_subject_distribution


def test_stat_tiles_renders_all_three_values():
    with patch("core.ui_components.st.markdown") as md:
        render_stat_tiles(document_count=3, chunk_count=42, subject_count=2)
    html = md.call_args[0][0]
    assert "Documents" in html and "3" in html
    assert "Chunks indexed" in html and "42" in html
    assert "Subjects" in html and ">2<" in html


def test_stat_tiles_shows_dash_for_zero_subjects():
    with patch("core.ui_components.st.markdown") as md:
        render_stat_tiles(document_count=1, chunk_count=5, subject_count=0)
    html = md.call_args[0][0]
    assert ">—<" in html


def test_distribution_skipped_for_single_subject():
    subjects = ["CS101"] * 10
    with patch("core.ui_components.st.markdown") as md:
        render_subject_distribution(subjects)
    md.assert_not_called()


def test_distribution_skipped_for_no_subjects():
    subjects = [""] * 5
    with patch("core.ui_components.st.markdown") as md:
        render_subject_distribution(subjects)
    md.assert_not_called()


def test_distribution_renders_bar_and_legend_for_multiple_subjects():
    subjects = ["CS101"] * 6 + ["MATH201"] * 4
    with patch("core.ui_components.st.markdown") as md:
        render_subject_distribution(subjects)
    html = md.call_args[0][0]
    assert "CS101" in html and "MATH201" in html
    assert "dist-bar" in html and "dist-legend" in html
    assert "· 6" in html and "· 4" in html


def test_distribution_uses_categorical_colors_in_fixed_order():
    subjects = ["A"] * 5 + ["B"] * 5
    with patch("core.ui_components.st.markdown") as md:
        render_subject_distribution(subjects)
    html = md.call_args[0][0]
    assert "#3987e5" in html
    assert "#d95926" in html


def test_distribution_folds_long_tail_into_other():
    subjects = []
    for i in range(7):
        subjects.extend([f"Subj{i}"] * (10 - i))
    with patch("core.ui_components.st.markdown") as md:
        render_subject_distribution(subjects)
    html = md.call_args[0][0]
    assert "Other" in html
    assert "Subj5" not in html
    assert "Subj6" not in html


def test_distribution_only_inline_labels_segments_wide_enough_to_fit():
    subjects = ["Big"] * 95 + ["Tiny"] * 5
    with patch("core.ui_components.st.markdown") as md:
        render_subject_distribution(subjects)
    html = md.call_args[0][0]
    assert "Tiny · 5" in html
