"""
Doraeon — AI-powered academic RAG assistant.
Streamlit frontend built on LlamaIndex + FAISS + OpenAI (see core/).
"""

from __future__ import annotations

import streamlit as st

from core.config import env_setup_hint, get_min_retrieval_score, get_openai_api_key, load_settings
from core.index_store import DoraeonIndex
from core.ingestion import load_multiple_pdfs
from core.query_rewriting import extract_history_turns
from core.rag_pipeline import RAGPipeline, RAGResponse
from core.study_tools import generate_flashcards, predict_exam_questions, summarize_topic
from core.ui_components import (
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

load_settings()

st.set_page_config(
    page_title="Doraeon",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# --- Session state ---
if "doraeon_index" not in st.session_state:
    st.session_state.doraeon_index = DoraeonIndex()
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files: list[str] = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0
if "extraction_issues" not in st.session_state:
    st.session_state.extraction_issues: list = []


def build_index(
    uploaded_files: list,
    subject: str,
    unit: str,
    chunk_size: int,
    overlap: int,
    on_progress=None,
) -> tuple[int, int]:
    file_data = [(f.getvalue(), f.name) for f in uploaded_files]
    documents, issues = load_multiple_pdfs(
        file_data, default_subject=subject, default_unit=unit, on_progress=on_progress
    )
    # Surfaced regardless of whether any documents came back -- a PDF that
    # fails on every page should be visible, not silently produce "0 chunks"
    # with no explanation.
    st.session_state.extraction_issues.extend(issues)
    if not documents:
        return 0, 0

    index: DoraeonIndex = st.session_state.doraeon_index
    n_chunks = index.add_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
    st.session_state.indexed_files.extend([f.name for f in uploaded_files])
    return len(documents), n_chunks


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
                for c in sources:
                    section = f" ({c.unit})" if c.unit else ""
                    lines.append(f"- {c.filename}, p.{c.page_number}{section} — {c.confidence:.0%} match")
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
            "rewritten_question": resp.rewritten_question,
            "grounded": resp.grounded,
            "refused": resp.refused,
            "msg_id": st.session_state.msg_counter,
        }
    )


def run_rag(
    question: str, top_k: int, subj: str | None, unit: str | None, min_score: float, strict_mode: bool
) -> RAGResponse:
    pipeline = RAGPipeline(st.session_state.doraeon_index)
    return pipeline.generate_answer(
        question,
        top_k=top_k,
        subject_filter=subj,
        unit_filter=unit,
        min_score=min_score,
        # Only complete turns already in chat_history at call time -- the
        # in-flight question is appended separately afterward via append_turn.
        history=extract_history_turns(st.session_state.chat_history),
        strict_mode=strict_mode,
    )


def render_assistant_message(msg: dict) -> None:
    msg_id = msg.get("msg_id", 0)
    sources = msg.get("sources") or []
    grounded = msg.get("grounded", True)
    refused = msg.get("refused", False)

    if msg.get("rewritten_question"):
        st.caption(f"🔎 Searched for: _{msg['rewritten_question']}_")

    if msg.get("error") == "missing_api_key":
        st.info(msg["content"])
    elif refused:
        st.warning(f"🚫 {msg['content']}")
    else:
        if not grounded and not msg.get("error"):
            # Soft-mode fallback that actually succeeded: a real answer, but
            # not from the uploaded materials -- flagged visually beyond the
            # inline disclaimer text so it's never mistaken for a grounded,
            # cited answer. Not shown if this is actually an error message
            # (e.g. the general-knowledge call itself hit a rate limit).
            st.caption("🌐 General knowledge (not from your uploaded materials)")
        render_answer_card(msg["content"])

    if sources:
        label = "##### 📎 Sources" if grounded else "##### 📎 Closest matches found (below confidence threshold)"
        st.markdown(label)
        render_citation_pills(sources)
        render_meta_strip(len(sources), retrieval_ms=msg.get("retrieval_ms"), generation_ms=msg.get("generation_ms"))
        render_source_expander(sources, expander_key=f"msg_{msg_id}")


# --- Sidebar ---
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
            help="Minimum similarity for an answer to count as grounded in your materials. "
            "Scores of 40-70% are often legitimately good matches in RAG -- don't set this too high.",
        )
        strict_mode = st.checkbox(
            "Strict mode: only answer from uploaded materials",
            value=False,
            help="On: refuse when nothing clears the confidence threshold (exam-prep grounding). "
            "Off: fall back to general knowledge with a clear disclaimer, and no source citations "
            "on that fallback answer (general help).",
        )
        filter_subject = st.text_input("Filter subject", placeholder="All")
        filter_unit = st.text_input("Filter unit", placeholder="All")

    with st.expander("📤 Upload PDFs", expanded=st.session_state.doraeon_index.node_count == 0):
        uploaded = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded and st.button("Build index", type="primary", use_container_width=True):
            issues_before = len(st.session_state.extraction_issues)
            # st.status instead of st.spinner: OCR fallback pages are slow
            # (seconds per page), and the label + body update live so a big
            # scanned PDF shows per-page progress instead of appearing frozen.
            # Starts expanded so that progress is visible as it happens, then
            # collapses once done to keep the sidebar tidy.
            with st.status("Indexing your PDFs…", expanded=True) as status:
                def _report(msg: str) -> None:
                    status.update(label=msg)
                    st.caption(f"🔎 {msg}")

                n_docs, n_chunks = build_index(
                    uploaded,
                    (default_subject or "").strip(),
                    (default_unit or "").strip(),
                    chunk_size,
                    overlap,
                    on_progress=_report,
                )
                status.update(
                    label=f"Indexed {n_docs} page(s) → {n_chunks} chunk(s)",
                    state="complete",
                    expanded=False,
                )
            new_issues = st.session_state.extraction_issues[issues_before:]
            if n_chunks:
                st.success(f"✓ {n_docs} pages → {n_chunks} chunks")
            else:
                st.warning("No text extracted.")
            if new_issues:
                st.warning(
                    f"⚠️ {len(new_issues)} page(s) could not be read cleanly (unusual/embedded "
                    "font) and were skipped rather than indexed as garbage — see below."
                )

    if st.session_state.extraction_issues:
        with st.expander(f"⚠️ {len(st.session_state.extraction_issues)} unreadable page(s)", expanded=False):
            for issue in st.session_state.extraction_issues:
                st.caption(f"📄 {issue.filename}, page {issue.page_number}: {issue.reason}")

    st.caption(f"📎 {st.session_state.doraeon_index.node_count} chunks indexed")

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
            st.session_state.doraeon_index.clear()
            st.session_state.chat_history = []
            st.session_state.indexed_files = []
            st.session_state.extraction_issues = []
            st.rerun()

    if not api_ok:
        with st.expander("🔑 Setup OpenAI"):
            st.markdown(env_setup_hint())

# --- Main ---
render_hero()

if st.session_state.doraeon_index.node_count > 0:
    nodes = st.session_state.doraeon_index.nodes
    chunks_meta = [n.metadata for n in nodes]
    document_count = len(set(st.session_state.indexed_files))
    subject_count = len({m.get("subject", "").strip() for m in chunks_meta if m.get("subject", "").strip()})
    render_stat_tiles(document_count, st.session_state.doraeon_index.node_count, subject_count)
    render_subject_distribution([m.get("subject", "") for m in chunks_meta])

if not get_openai_api_key():
    render_api_setup_banner(env_setup_hint())

subj_filter = (filter_subject or "").strip() or None
unit_filter = (filter_unit or "").strip() or None

tab_chat, tab_tools = st.tabs(["💬 Chat", "📝 Study tools"])

with tab_tools:
    st.markdown("##### Summarize, create flashcards, or predict exam questions from your index")
    topic_input = st.text_input("Topic", placeholder="e.g. Binary Search Trees", key="tool_topic")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        n_cards = st.number_input("Cards", 3, 15, 5)
    with c2:
        if st.button("📋 Summarize", use_container_width=True) and topic_input:
            if st.session_state.doraeon_index.node_count == 0:
                st.warning("Build an index first.")
            else:
                with st.spinner("Summarizing…"):
                    p = RAGPipeline(st.session_state.doraeon_index)
                    resp = summarize_topic(p, topic_input, top_k=top_k, min_score=min_score, strict_mode=strict_mode)
                append_turn(f"Summarize: {topic_input}", resp)
                st.rerun()
    with c3:
        if st.button("🃏 Flashcards", use_container_width=True) and topic_input:
            if st.session_state.doraeon_index.node_count == 0:
                st.warning("Build an index first.")
            else:
                with st.spinner("Creating flashcards…"):
                    p = RAGPipeline(st.session_state.doraeon_index)
                    resp = generate_flashcards(
                        p, topic_input, int(n_cards), top_k=top_k, min_score=min_score, strict_mode=strict_mode
                    )
                append_turn(f"Flashcards: {topic_input}", resp)
                st.rerun()
    with c4:
        if st.button("🎯 Predict exam Qs", use_container_width=True) and topic_input:
            if st.session_state.doraeon_index.node_count == 0:
                st.warning("Build an index first.")
            else:
                with st.spinner("Predicting exam questions…"):
                    p = RAGPipeline(st.session_state.doraeon_index)
                    resp = predict_exam_questions(
                        p, topic_input, int(n_cards), top_k=top_k, min_score=min_score, strict_mode=strict_mode
                    )
                append_turn(f"Predict exam questions: {topic_input}", resp)
                st.rerun()

with tab_chat:
    if st.session_state.doraeon_index.node_count == 0:
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

    if st.session_state.chat_history:
        st.markdown(
            '<div id="doraeon-chat-end"></div>'
            '<script>document.getElementById("doraeon-chat-end")?.scrollIntoView({behavior:"smooth"});</script>',
            unsafe_allow_html=True,
        )

    question = st.chat_input("Ask about your course materials…")

    if question:
        if st.session_state.doraeon_index.node_count == 0:
            st.warning("Upload PDFs and build your index in the sidebar first.")
        else:
            with st.status("Thinking…", expanded=False) as status:
                status.update(label="🔍 Retrieving relevant sources…")
                resp = run_rag(question, top_k, subj_filter, unit_filter, min_score, strict_mode)
                status.update(label="✨ Generating answer…")
                status.update(state="complete")
            append_turn(question, resp)
            st.rerun()
