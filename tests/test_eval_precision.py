import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_eval import precision_at_k  # noqa: E402
from utils.chunker import TextChunk
from utils.vector_store import SearchResult


def _result(filename: str, page: int) -> SearchResult:
    chunk = TextChunk(text="x", filename=filename, page_number=page, subject="", unit="", chunk_index=0)
    return SearchResult(chunk=chunk, score=0.9)


def test_exact_page_match_hits():
    assert precision_at_k([_result("Lecture3.pdf", 7)], "Lecture3.pdf", 7)


def test_within_tolerance_still_hits():
    assert precision_at_k([_result("Lecture3.pdf", 8)], "Lecture3.pdf", 7, page_tolerance=1)


def test_outside_tolerance_misses():
    assert not precision_at_k([_result("Lecture3.pdf", 20)], "Lecture3.pdf", 7, page_tolerance=1)


def test_wrong_filename_misses_even_on_correct_page():
    assert not precision_at_k([_result("Other.pdf", 7)], "Lecture3.pdf", 7)


def test_hit_if_any_result_in_top_k_matches():
    results = [_result("Other.pdf", 1), _result("Lecture3.pdf", 7), _result("Another.pdf", 2)]
    assert precision_at_k(results, "Lecture3.pdf", 7)


def test_empty_results_is_a_miss():
    assert not precision_at_k([], "Lecture3.pdf", 7)


def test_zero_tolerance_requires_exact_page():
    assert precision_at_k([_result("Lecture3.pdf", 7)], "Lecture3.pdf", 7, page_tolerance=0)
    assert not precision_at_k([_result("Lecture3.pdf", 8)], "Lecture3.pdf", 7, page_tolerance=0)
