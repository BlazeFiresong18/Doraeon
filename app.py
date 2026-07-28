"""
Doraeon — AI-powered academic RAG assistant.
Polished Streamlit frontend with answer-centric layout.
"""

from __future__ import annotations

import streamlit as st

from utils.chunker import TextChunk, chunk_pages
from utils.config import env_setup_hint, get_min_retrieval_score, get_openai_api_key, load_settings
from utils.embeddings import EmbeddingModel
from utils.pdf_loader import load_multiple_pdfs
from utils.rag_pipeline import RAGPipeline, RAGResponse
from utils.ui_components import (
    inject_global_css,
    render_answer_card,
    render_api_setup_banner,
    render_citation_pills,
    render_empty_chat_state,
    render_empty_upload_state,
    render_hero,
    render_meta_strip,
    render_source_expander,
    render_stat_tiles,
    render_subject_distribution,
)
from utils.vector_store import FaissVectorStore

load_settings()

st.set_page_config(
    page_title="Doraeon",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# --- Session state ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = FaissVectorStore()
if "embedder" not in st.session_state:
    st.session_state.embedder = EmbeddingModel()
if "chunks" not in st.session_state:
    st.session_state.chunks: list[TextChunk] = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files: list[str] = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0


def build_index(
    uploaded_files: list,
    subject: str,
    unit: str,
    chunk_size: int,
    overlap: int,
) -> tuple[int, int]:
    file_data = [(f.getvalue(), f.name) for f in uploaded_files]
    pages = load_multiple_pdfs(file_data, default_subject=subject, default_unit=unit)
    if not pages:
        return 0, 0

    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    texts = [c.text for c in chunks]
    embeddings = st.session_state.embedder.embed_texts(texts)

    store: FaissVectorStore = st.session_state.vector_store
    store.add(embeddings, chunks)
    st.session_state.chunks.extend(chunks)
    st.session_state.indexed_files.extend([f.name for f in uploaded_files])
    return len(pages), len(chunks)


def export_chat_markdown() -> str:
    """Render the conversation (questions, answers, citations) as Markdown --
    useful as study notes, independent of the app."""
    lines = ["# Doraeon study session", ""]
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            lines.append(f"## Q: {msg['content']}")
        else:
            lines.append(msg["content"])
            sources = msg.get("sources") or []
            if sources:
                lines.append("")
                lines.append("**Sources:**")
                for r in sources:
                    c = r.chunk
                    section = f" ({c.unit})" if c.unit else ""
                    lines.append(f"- {c.filename}, p.{c.page_number}{section} — {r.score:.0%} match")
        lines.append("")
    return "\n".join(lines)


def append_turn(user_text: str, resp: RAGResponse) -> None:
    st.session_state.msg_counter += 1
    st.session_state.chat_history.append({"role": "user", "content": user_text})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": resp.answer,
            "sources": resp.sources,
            "retrieval_ms": resp.retrieval_ms,
            "generation_ms": resp.generation_ms,
            "error": resp.error,
            "msg_id": st.session_state.msg_counter,
        }
    )


def run_rag(
    question: str, top_k: int, subj: str | None, unit: str | None, min_score: float
) -> RAGResponse:
    pipeline = RAGPipeline(st.session_state.vector_store, st.session_state.embedder)
    return pipeline.generate_answer(
        question, top_k=top_k, subject_filter=subj, unit_filter=unit, min_score=min_score
    )


def render_assistant_message(msg: dict) -> None:
    msg_id = msg.get("msg_id", 0)
    sources = msg.get("sources") or []

    if msg.get("error") == "missing_api_key":
        st.info(msg["content"])
    elif msg.get("error") == "below_confidence_threshold":
        st.warning(f"🚫 {msg['content']}")
    else:
        render_answer_card(msg["content"])

    if sources:
        label = (
            "##### 📎 Closest matches found (below confidence threshold)"
            if msg.get("error") == "below_confidence_threshold"
            else "##### 📎 Sources"
        )
        st.markdown(label)
        render_citation_pills(sources)
        render_meta_strip(
            len(sources),
            retrieval_ms=msg.get("retrieval_ms"),
            generation_ms=msg.get("generation_ms"),
        )
        render_source_expander(sources, expander_key=f"msg_{msg_id}")


# --- Sidebar (compact) ---
with st.sidebar:
    st.markdown("### 📚 Doraeon")
    api_ok = bool(get_openai_api_key())
    if api_ok:
        st.caption("🟢 OpenAI ready")
    else:
        st.caption("🟡 API key not configured")

    with st.expander("⚙️ Index & filters", expanded=True):
        default_subject = st.text_input("Subject", placeholder="CS101", label_visibility="collapsed")
        st.caption("Default subject (optional)")
        default_unit = st.text_input("Unit", placeholder="Unit 2", label_visibility="collapsed")
        st.caption("Default unit (optional)")
        chunk_size = st.slider("Chunk words", 300, 500, 400)
        overlap = st.slider("Overlap", 40, 120, 80)
        top_k = st.slider("Sources to retrieve", 3, 10, 5)
        min_score = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            get_min_retrieval_score(),
            step=0.05,
            help="Below this similarity score, Doraeon refuses to answer instead of guessing.",
        )
        filter_subject = st.text_input("Filter subject", placeholder="All")
        filter_unit = st.text_input("Filter unit", placeholder="All")

    with st.expander("📤 Upload PDFs", expanded=st.session_state.vector_store.size == 0):
        uploaded = st.file_uploader(
            "PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button("Build index", type="primary", use_container_width=True):
            with st.spinner("Indexing…"):
                n_pages, n_chunks = build_index(
                    uploaded,
                    (default_subject or "").strip(),
                    (default_unit or "").strip(),
                    chunk_size,
                    overlap,
                )
            if n_chunks:
                st.success(f"✓ {n_pages} pages → {n_chunks} chunks")
            else:
                st.warning("No text extracted.")

    st.caption(f"📎 {st.session_state.vector_store.size} chunks indexed")

    if st.session_state.chat_history:
        st.download_button(
            "⬇️ Download conversation (.md)",
            data=export_chat_markdown(),
            file_name="doraeon_study_session.md",
            mime="text/markdown",
            use_container_width=True,
        )

    col_clear_chat, col_clear_all = st.columns(2)
    with col_clear_chat:
        if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.chat_history):
            st.session_state.chat_history = []
            st.rerun()
    with col_clear_all:
        if st.button("Clear all", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.chunks = []
            st.session_state.chat_history = []
            st.session_state.indexed_files = []
            st.rerun()

    if not api_ok:
        with st.expander("🔑 Setup OpenAI"):
            st.markdown(env_setup_hint())

# --- Main ---
render_hero()

if st.session_state.vector_store.size > 0:
    chunks: list[TextChunk] = st.session_state.chunks
    document_count = len(set(st.session_state.indexed_files))
    subject_count = len({c.subject.strip() for c in chunks if c.subject and c.subject.strip()})
    render_stat_tiles(document_count, st.session_state.vector_store.size, subject_count)
    render_subject_distribution(chunks)

if not get_openai_api_key():
    render_api_setup_banner(env_setup_hint())

subj_filter = (filter_subject or "").strip() or None
unit_filter = (filter_unit or "").strip() or None

tab_chat, tab_tools = st.tabs(["💬 Chat", "📝 Study tools"])

with tab_tools:
    st.markdown("##### Summarize or create flashcards from your index")
    topic_input = st.text_input("Topic", placeholder="e.g. Binary Search Trees", key="tool_topic")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        n_cards = st.number_input("Cards", 3, 15, 5)
    with c2:
        if st.button("📋 Summarize", use_container_width=True) and topic_input:
            if st.session_state.vector_store.size == 0:
                st.warning("Build an index first.")
            else:
                with st.spinner("Summarizing…"):
                    p = RAGPipeline(st.session_state.vector_store, st.session_state.embedder)
                    resp = p.summarize_topic(topic_input, top_k=top_k, min_score=min_score)
                append_turn(f"Summarize: {topic_input}", resp)
                st.rerun()
    with c3:
        if st.button("🃏 Flashcards", use_container_width=True) and topic_input:
            if st.session_state.vector_store.size == 0:
                st.warning("Build an index first.")
            else:
                with st.spinner("Creating flashcards…"):
                    p = RAGPipeline(st.session_state.vector_store, st.session_state.embedder)
                    resp = p.generate_flashcards(topic_input, int(n_cards), top_k=top_k, min_score=min_score)
                append_turn(f"Flashcards: {topic_input}", resp)
                st.rerun()

with tab_chat:
    if st.session_state.vector_store.size == 0:
        render_empty_upload_state()
    elif not st.session_state.chat_history:
        render_empty_chat_state()

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📚"):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_assistant_message(msg)

    # Auto-scroll anchor for latest message
    if st.session_state.chat_history:
        st.markdown(
            '<div id="doraeon-chat-end"></div>'
            '<script>document.getElementById("doraeon-chat-end")?.scrollIntoView({behavior:"smooth"});</script>',
            unsafe_allow_html=True,
        )

    question = st.chat_input("Ask about your course materials…")

    if question:
        if st.session_state.vector_store.size == 0:
            st.warning("Upload PDFs and build your index in the sidebar first.")
        else:
            with st.status("Thinking…", expanded=False) as status:
                status.update(label="🔍 Retrieving relevant sources…")
                resp = run_rag(question, top_k, subj_filter, unit_filter, min_score)
                status.update(label="✨ Generating answer…")
                status.update(state="complete")
            append_turn(question, resp)
            st.rerun()
