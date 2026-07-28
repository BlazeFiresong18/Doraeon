from core.text_cleaner import clean_pdf_text, preview_snippet


def test_fixes_hyphenation_across_linebreak():
    result = clean_pdf_text("This is an exam-\nple of hyphenation.")
    assert "example" in result
    assert "exam-" not in result


def test_collapses_excess_whitespace():
    result = clean_pdf_text("Too    many     spaces")
    assert "  " not in result


def test_removes_consecutive_duplicate_words():
    result = clean_pdf_text("the the quick brown brown fox")
    words = result.lower().split()
    assert words.count("the") == 1
    assert words.count("brown") == 1


def test_empty_input_returns_empty():
    assert clean_pdf_text("") == ""
    assert clean_pdf_text("   ") == ""


def test_preview_snippet_truncates_long_text():
    long_text = " ".join(["word"] * 100)
    preview = preview_snippet(long_text, max_chars=50)
    assert len(preview) <= 51  # allows the trailing ellipsis char
    assert preview.endswith("…")


def test_preview_snippet_leaves_short_text_untouched():
    short_text = "A short sentence."
    assert preview_snippet(short_text, max_chars=220) == short_text


def test_preview_snippet_never_returns_full_text_verbatim_when_long():
    long_text = "x " * 500
    preview = preview_snippet(long_text)
    assert len(preview) < len(long_text)
