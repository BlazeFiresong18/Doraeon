"""Retrieval wrapper + confidence normalization.

Confidence formula (verified against a real query before this was written --
see the Phase 2 diagnostic in the accompanying conversation/commit history):

    confidence_pct = round(max(0.0, cosine_similarity) * 100)

Why: embeddings are L2-normalized (HuggingFaceEmbedding(normalize=True),
confirmed empirically to produce unit vectors) and the FAISS index is
IndexFlatIP, so the score LlamaIndex returns on each NodeWithScore IS cosine
similarity -- confirmed empirically to match raw sentence-transformers
cosine computed independently, not some other FAISS-internal convention.
Cosine similarity mathematically ranges [-1, 1]; for real text pairs under
this model, unrelated content lands very close to 0 (sometimes epsilon
positive, sometimes epsilon negative) rather than near -1. A negative value
means "less related than two random sentences" -- for display purposes
that's indistinguishable from "not similar," so it's clamped to 0% rather
than shown as a confusing negative percentage. This is NOT a rescaled/
stretched calibration (no invented anchors) -- it's the direct cosine value,
just floored at zero for display and thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore

from core.index_store import DoraeonIndex


@dataclass
class ScoredChunk:
    """One retrieved chunk, with both the raw and normalized confidence score."""

    text: str
    filename: str
    page_number: int
    subject: str
    unit: str
    raw_score: float
    confidence: float  # normalized 0.0-1.0, see module docstring


def normalize_score(raw_score: float) -> float:
    """The confidence formula: clamp negative cosine similarity to 0, no rescaling."""
    return max(0.0, raw_score)


def _to_scored_chunk(node_with_score: NodeWithScore) -> ScoredChunk:
    meta = node_with_score.node.metadata
    raw = node_with_score.score if node_with_score.score is not None else 0.0
    return ScoredChunk(
        text=node_with_score.node.get_content(),
        filename=meta.get("filename", ""),
        page_number=meta.get("page_number", 0),
        subject=meta.get("subject", ""),
        unit=meta.get("unit", ""),
        raw_score=raw,
        confidence=normalize_score(raw),
    )


def retrieve(
    doraeon_index: DoraeonIndex,
    query: str,
    top_k: int = 5,
    subject_filter: str | None = None,
    unit_filter: str | None = None,
) -> list[ScoredChunk]:
    """Retrieve top_k chunks for `query`, with normalized confidence scores.

    Filtering is applied post-retrieval (over-fetching first) rather than via
    LlamaIndex's native MetadataFilters, since unit_filter is a substring
    match (e.g. "Unit 2" matching a detected heading "Unit 2: Recursion"),
    which native exact-match filters don't support -- same semantics as the
    prior implementation's filtering.
    """
    fetch_k = top_k * 4 if (subject_filter or unit_filter) else top_k
    retriever = doraeon_index.index.as_retriever(similarity_top_k=max(fetch_k, top_k))
    raw_results = retriever.retrieve(query)

    chunks = [_to_scored_chunk(r) for r in raw_results]

    if subject_filter:
        chunks = [c for c in chunks if c.subject.lower() == subject_filter.lower()]
    if unit_filter:
        chunks = [c for c in chunks if unit_filter.lower() in c.unit.lower()]

    return chunks[:top_k]
