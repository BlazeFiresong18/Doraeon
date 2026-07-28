"""Streamlit UI helpers: theme CSS, citations, chunk cards, chat layout."""

from __future__ import annotations

import html

import streamlit as st

from utils.text_cleaner import preview_snippet
from utils.vector_store import SearchResult


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 920px;
        }

        .doraeon-hero {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #1e293b 100%);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }
        .doraeon-hero h1 {
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            color: #f8fafc;
        }
        .doraeon-hero p {
            margin: 0;
            color: #c7d2fe;
            font-size: 0.95rem;
        }

        .answer-card {
            background: #1a1d29;
            border: 1px solid #2d3348;
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            margin: 0.5rem 0 1rem 0;
            box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        }
        .answer-card .label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #818cf8;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .citation-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0 0.25rem 0;
        }
        .cite-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: #252a3a;
            border: 1px solid #3d4460;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-size: 0.8rem;
            color: #e2e8f0;
        }
        .score-badge {
            background: #312e81;
            color: #c7d2fe;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.15rem 0.45rem;
            border-radius: 6px;
            margin-left: 0.25rem;
        }

        .chunk-card {
            background: #141820;
            border: 1px solid #2a3142;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
            transition: border-color 0.15s ease;
        }
        .chunk-card:hover {
            border-color: #4f46e5;
        }
        .chunk-meta {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-bottom: 0.45rem;
        }
        .chunk-preview {
            font-size: 0.88rem;
            line-height: 1.55;
            color: #cbd5e1;
        }

        .empty-state {
            text-align: center;
            padding: 2rem 1.5rem;
            border: 1px dashed #3d4460;
            border-radius: 14px;
            color: #94a3b8;
            background: #12151c;
        }
        .empty-state h3 {
            color: #e2e8f0;
            margin-bottom: 0.5rem;
        }

        .meta-strip {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.35rem;
        }

        div[data-testid="stSidebar"] {
            background: #0f1117;
        }
        div[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        .stChatMessage {
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="doraeon-hero">
            <h1>📚 Doraeon</h1>
            <p>Your AI academic assistant — grounded answers from your lecture PDFs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_label(score: float) -> str:
    pct = int(max(0, min(100, score * 100)))
    if pct >= 75:
        return "High"
    if pct >= 50:
        return "Med"
    return "Low"


def _section_label(chunk) -> str:
    """Subject/unit suffix for a citation, e.g. ' · CS101 · Unit 2' -- omits
    parts that weren't detected rather than showing empty placeholders."""
    parts = [p for p in (chunk.subject, chunk.unit) if p]
    return f" · {' · '.join(html.escape(p) for p in parts)}" if parts else ""


def render_citation_pills(sources: list[SearchResult]) -> None:
    if not sources:
        return
    pills = []
    for r in sources:
        c = r.chunk
        label = f"📄 {html.escape(c.filename)} — Page {c.page_number}{_section_label(c)}"
        badge = f'<span class="score-badge">{score_label(r.score)} {r.score:.0%}</span>'
        pills.append(f'<span class="cite-pill">{label}{badge}</span>')
    st.markdown(
        f'<div class="citation-row">{"".join(pills)}</div>',
        unsafe_allow_html=True,
    )


def render_answer_card(answer: str, label: str = "Answer") -> None:
    st.markdown(
        f'<div class="answer-card"><div class="label">✨ {html.escape(label)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(answer)


def render_meta_strip(
    chunk_count: int,
    retrieval_ms: float | None = None,
    generation_ms: float | None = None,
) -> None:
    parts = [f"📎 {chunk_count} source{'s' if chunk_count != 1 else ''}"]
    if retrieval_ms is not None:
        parts.append(f"⏱ retrieval {retrieval_ms:.0f}ms")
    if generation_ms is not None:
        parts.append(f"⏱ generation {generation_ms:.0f}ms")
    st.markdown(
        f'<p class="meta-strip">{" · ".join(parts)}</p>',
        unsafe_allow_html=True,
    )


def render_source_expander(
    sources: list[SearchResult],
    expander_key: str,
) -> None:
    if not sources:
        return
    with st.expander(f"📑 View retrieved context ({len(sources)})", expanded=False):
        for i, r in enumerate(sources, start=1):
            c = r.chunk
            preview = preview_snippet(c.text)
            title = f"Source {i} · 📄 {c.filename} · p.{c.page_number}{_section_label(c)} · {r.score:.0%}"
            with st.expander(title, expanded=False):
                st.markdown(
                    f'<p class="chunk-preview">{html.escape(preview)}</p>',
                    unsafe_allow_html=True,
                )
                if len(c.text) > len(preview) + 5:
                    with st.expander("Show more", expanded=False):
                        st.markdown(c.text)


def render_empty_upload_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>📤 Upload your materials</h3>
            <p>Add lecture slides, notes, syllabi, or past papers in the sidebar.<br>
            Then click <strong>Build index</strong> to start asking questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_chat_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>💬 Ask anything</h3>
            <p>Your index is ready. Type a question below — answers cite your PDFs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_api_setup_banner(hint: str) -> None:
    st.warning("🔑 **OpenAI API key required for AI answers**")
    st.markdown(hint)
