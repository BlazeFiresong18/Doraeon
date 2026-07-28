"""FAISS-backed LlamaIndex vector store: chunking + indexing of ingested Documents.

Uses a whitespace tokenizer for SentenceSplitter rather than LlamaIndex's
default (empirically confirmed to be a GPT-style subword tokenizer, which
would silently make chunk_size=400 produce ~200-word chunks, not 400) --
this keeps chunk_size/overlap meaning WORD counts, matching the sidebar
sliders' labels and the prior implementation's semantics.
"""

from __future__ import annotations

import re

import faiss
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from core.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

_HEADING_PATTERN = re.compile(r"^(unit|chapter|ch\.?|module|topic|lecture)\s*[\d.:]+", re.IGNORECASE)


def _detect_topic_heading(text: str) -> str:
    """Pick a short heading-like first line (Unit 2, Chapter 3, etc.) from a
    chunk's own text, if present -- upgrades the unit tag beyond the
    filename-level guess when a chunk actually starts at a real heading.
    Must run on each NODE's text (post-chunking), not the source page/document
    text -- a chunk boundary can start mid-page at a heading that the
    document-level filename guess has no way to see."""
    first_lines = text.strip().split("\n")[:3]
    for line in first_lines:
        line = line.strip()
        if len(line) < 80 and _HEADING_PATTERN.match(line):
            return line
    return ""


def build_embed_model() -> HuggingFaceEmbedding:
    # normalize=True -> L2-normalized embeddings, so inner product on the
    # FAISS IndexFlatIP below is exactly cosine similarity, not raw dot product.
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, normalize=True)


def _word_splitter(chunk_size: int, chunk_overlap: int) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=lambda text: text.split(),
    )


class DoraeonIndex:
    """Wraps a FAISS-backed VectorStoreIndex with incremental document insertion."""

    def __init__(self, embed_model: HuggingFaceEmbedding | None = None):
        self.embed_model = embed_model or build_embed_model()
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context, embed_model=self.embed_model
        )
        self._document_count = 0
        self._node_count = 0
        self._nodes: list[BaseNode] = []  # kept for subject/unit stats + tests

    def add_documents(self, documents: list[Document], chunk_size: int = 400, chunk_overlap: int = 80) -> int:
        """Chunk and index `documents`; returns the number of chunks (nodes) created."""
        if not documents:
            return 0
        splitter = _word_splitter(chunk_size, chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)
        for node in nodes:
            heading = _detect_topic_heading(node.get_content())
            if heading:
                node.metadata["unit"] = heading
        if nodes:
            self.index.insert_nodes(nodes)
        self._nodes.extend(nodes)
        self._document_count += len(documents)
        self._node_count += len(nodes)
        return len(nodes)

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def node_count(self) -> int:
        return self._node_count

    @property
    def nodes(self) -> list[BaseNode]:
        return self._nodes

    def clear(self) -> None:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context, embed_model=self.embed_model
        )
        self._document_count = 0
        self._node_count = 0
        self._nodes = []
