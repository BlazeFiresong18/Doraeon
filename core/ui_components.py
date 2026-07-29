"""Streamlit UI helpers: theme CSS, citations, chunk cards, chat layout, stats."""

from __future__ import annotations

import html
from collections import Counter

import streamlit as st

from core.retrieval import ScoredChunk
from core.text_cleaner import preview_snippet

# Categorical palette (dark-mode steps), fixed order -- never reassigned per
# render, never cycled past the validated set. Beyond this many series the
# distribution chart folds the tail into "Other" (muted gray) rather than
# generating a 9th hue.
_CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
_OTHER_COLOR = "#64748b"


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 960px;
        }

        .doraeon-hero {
            background:
                radial-gradient(circle at 15% 20%, rgba(129, 140, 248, 0.25), transparent 45%),
                linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #1e293b 100%);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 32px rgba(30, 27, 75, 0.35);
        }
        .doraeon-hero h1 {
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: #f8fafc;
            letter-spacing: -0.01em;
        }
        .doraeon-hero p {
            margin: 0;
            color: #c7d2fe;
            font-size: 0.95rem;
        }

        /* ---- KPI stat tiles ---------------------------------------------- */
        .stat-row {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.1rem;
            flex-wrap: wrap;
        }
        .stat-tile {
            flex: 1 1 140px;
            background: #161a26;
            border: 1px solid #2a3142;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            transition: border-color 0.15s ease, transform 0.15s ease;
        }
        .stat-tile:hover {
            border-color: #4f46e5;
            transform: translateY(-1px);
        }
        .stat-tile .stat-label {
            font-size: 0.72rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }
        .stat-tile .stat-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #e2e8f0;
            font-variant-numeric: proportional-nums;
        }

        /* ---- Subject distribution (part-to-whole) ------------------------ */
        .dist-wrap {
            background: #161a26;
            border: 1px solid #2a3142;
            border-radius: 12px;
            padding: 0.9rem 1rem 1rem 1rem;
            margin-bottom: 1.1rem;
        }
        .dist-title {
            font-size: 0.78rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.6rem;
        }
        .dist-bar {
            display: flex;
            gap: 2px;
            height: 22px;
            border-radius: 6px;
            overflow: hidden;
            background: #0f1117;
        }
        .dist-seg {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.68rem;
            font-weight: 600;
            color: #f8fafc;
            white-space: nowrap;
            overflow: hidden;
        }
        .dist-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem 1rem;
            margin-top: 0.6rem;
        }
        .dist-legend-item {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.78rem;
            color: #cbd5e1;
        }
        .dist-swatch {
            width: 10px;
            height: 10px;
            border-radius: 3px;
            flex-shrink: 0;
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
            transition: border-color 0.15s ease;
        }
        .cite-pill:hover {
            border-color: #6366f1;
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
        .example-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 0.9rem;
        }
        .example-chip {
            background: #1e2436;
            border: 1px solid #3d4460;
            border-radius: 999px;
            padding: 0.3rem 0.8rem;
            font-size: 0.78rem;
            color: #a5b4fc;
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

        .stButton > button {
            transition: transform 0.1s ease;
        }
        .stButton > button:active {
            transform: scale(0.98);
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


def render_stat_tiles(document_count: int, chunk_count: int, subject_count: int) -> None:
    """KPI row: a handful of headline numbers, not a chart (see dataviz skill's
    choosing-a-form guidance -- this is exactly the 'not a chart' case)."""
    tiles = [
        ("Documents", document_count),
        ("Chunks indexed", chunk_count),
        ("Subjects", subject_count if subject_count else "—"),
    ]
    cells = "".join(
        f'<div class="stat-tile"><div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{value}</div></div>'
        for label, value in tiles
    )
    st.markdown(f'<div class="stat-row">{cells}</div>', unsafe_allow_html=True)


def render_subject_distribution(subjects: list[str]) -> None:
    """Part-to-whole breakdown of indexed chunks by subject: a horizontal
    stacked bar with a legend, per the dataviz skill's part-to-whole guidance.
    Skipped entirely when there's only one subject (or none) -- a single-series
    bar has nothing to compare and a legend for one color restates the title.

    Takes raw subject strings (one per chunk, blanks allowed) rather than
    chunk-like objects -- that's all this ever needed, and it avoids callers
    having to construct throwaway chunk objects just to satisfy a type."""
    subjects = [s.strip() for s in subjects if s and s.strip()]
    if len(set(subjects)) < 2:
        return

    counts = Counter(subjects)
    ranked = counts.most_common()
    total = sum(counts.values())

    # Series-count ladder: soft cap at 5 named slots, fold the tail into "Other"
    # rather than generating a 6th+ hue (see dataviz skill, categorical palette).
    top = ranked[: len(_CATEGORICAL)]
    rest = ranked[len(_CATEGORICAL) :]
    rest_total = sum(n for _, n in rest)

    segments = [(name, n, _CATEGORICAL[i]) for i, (name, n) in enumerate(top)]
    if rest_total:
        segments.append((f"Other ({len(rest)})", rest_total, _OTHER_COLOR))

    bar_html = []
    legend_html = []
    for name, n, color in segments:
        pct = n / total
        # Only inline-label a segment when the text will actually fit --
        # never clip/overflow a label inside its own segment.
        label = f"{html.escape(name)} {pct:.0%}" if pct >= 0.14 else ""
        bar_html.append(
            f'<div class="dist-seg" style="flex:{pct};background:{color}">{label}</div>'
        )
        legend_html.append(
            f'<span class="dist-legend-item"><span class="dist-swatch" '
            f'style="background:{color}"></span>{html.escape(name)} · {n}</span>'
        )

    st.markdown(
        f"""
        <div class="dist-wrap">
            <div class="dist-title">Index composition by subject</div>
            <div class="dist-bar">{"".join(bar_html)}</div>
            <div class="dist-legend">{"".join(legend_html)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_label(confidence: float) -> str:
    pct = int(max(0, min(100, confidence * 100)))
    if pct >= 75:
        return "High"
    if pct >= 50:
        return "Med"
    return "Low"


def _section_label(chunk: ScoredChunk) -> str:
    """Subject/unit suffix for a citation, e.g. ' · CS101 · Unit 2' -- omits
    parts that weren't detected rather than showing empty placeholders."""
    parts = [p for p in (chunk.subject, chunk.unit) if p]
    return f" · {' · '.join(html.escape(p) for p in parts)}" if parts else ""


def render_citation_pills(sources: list[ScoredChunk]) -> None:
    if not sources:
        return
    pills = []
    for c in sources:
        label = f"📄 {html.escape(c.filename)} — Page {c.page_number}{_section_label(c)}"
        badge = f'<span class="score-badge">{score_label(c.confidence)} {c.confidence:.0%}</span>'
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
    sources: list[ScoredChunk],
    expander_key: str,
) -> None:
    if not sources:
        return
    with st.expander(f"📑 View retrieved context ({len(sources)})", expanded=False):
        for i, c in enumerate(sources, start=1):
            preview = preview_snippet(c.text)
            title = f"Source {i} · 📄 {c.filename} · p.{c.page_number}{_section_label(c)} · {c.confidence:.0%}"
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
            <div class="example-chip-row">
                <span class="example-chip">"Explain [a concept from your notes]"</span>
                <span class="example-chip">"What does [Lecture X] say about [topic]?"</span>
                <span class="example-chip">"Summarize [Unit Y]"</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_setup_banner(hint: str) -> None:
    st.warning("⚙️ **Ollama isn't reachable -- AI answers are disabled**")
    st.markdown(hint)
