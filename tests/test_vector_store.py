import numpy as np

from utils.chunker import TextChunk
from utils.vector_store import FaissVectorStore


def _chunk(i: int, subject: str = "CS101", unit: str = "") -> TextChunk:
    return TextChunk(text=f"chunk {i}", filename="doc.pdf", page_number=i, subject=subject, unit=unit, chunk_index=i)


def _unit_vector(direction: np.ndarray) -> np.ndarray:
    return (direction / np.linalg.norm(direction)).astype(np.float32)


def test_empty_store_search_returns_nothing():
    store = FaissVectorStore(dimension=4)
    result = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert result == []
    assert store.size == 0


def test_add_and_search_returns_closest_match_first():
    store = FaissVectorStore(dimension=4)
    vectors = np.array(
        [
            _unit_vector(np.array([1.0, 0.0, 0.0, 0.0])),
            _unit_vector(np.array([0.0, 1.0, 0.0, 0.0])),
            _unit_vector(np.array([0.9, 0.1, 0.0, 0.0])),
        ]
    )
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    store.add(vectors, chunks)

    query = _unit_vector(np.array([1.0, 0.0, 0.0, 0.0]))
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.page_number == 0  # exact match should rank first
    assert results[0].score > results[1].score


def test_add_length_mismatch_raises():
    store = FaissVectorStore(dimension=4)
    vectors = np.array([_unit_vector(np.array([1.0, 0.0, 0.0, 0.0]))])
    try:
        store.add(vectors, [_chunk(0), _chunk(1)])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_subject_filter_excludes_non_matching_chunks():
    store = FaissVectorStore(dimension=4)
    vectors = np.array(
        [_unit_vector(np.array([1.0, 0.0, 0.0, 0.0])), _unit_vector(np.array([1.0, 0.01, 0.0, 0.0]))]
    )
    chunks = [_chunk(0, subject="CS101"), _chunk(1, subject="MATH201")]
    store.add(vectors, chunks)

    query = _unit_vector(np.array([1.0, 0.0, 0.0, 0.0]))
    results = store.search(query, top_k=5, subject_filter="MATH201")

    assert len(results) == 1
    assert results[0].chunk.subject == "MATH201"


def test_unit_filter_is_substring_match():
    store = FaissVectorStore(dimension=4)
    vectors = np.array([_unit_vector(np.array([1.0, 0.0, 0.0, 0.0]))])
    chunks = [_chunk(0, unit="Unit 2: Recursion")]
    store.add(vectors, chunks)

    query = _unit_vector(np.array([1.0, 0.0, 0.0, 0.0]))
    assert len(store.search(query, unit_filter="Unit 2")) == 1
    assert len(store.search(query, unit_filter="Unit 9")) == 0


def test_clear_resets_store():
    store = FaissVectorStore(dimension=4)
    vectors = np.array([_unit_vector(np.array([1.0, 0.0, 0.0, 0.0]))])
    store.add(vectors, [_chunk(0)])
    assert store.size == 1
    store.clear()
    assert store.size == 0
