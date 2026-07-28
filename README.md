# Doraeon

An academic RAG (retrieval-augmented generation) assistant. Upload lecture PDFs, ask questions in a chat UI, and get answers grounded in your own materials — with page/section-level citations, a confidence-based refusal guardrail against hallucination, and an evaluation harness to actually measure retrieval and answer quality rather than just eyeballing it.

## Stack

| Layer | Technology |
|---|---|
| UI | **Streamlit** |
| Embeddings | **SentenceTransformers** (`all-MiniLM-L6-v2`, 384-dim) |
| Vector search | **FAISS** (`IndexFlatIP` — inner product on L2-normalized vectors = cosine similarity) |
| LLM synthesis + eval grading | **OpenAI API** (`gpt-4o-mini` by default, configurable) |
| PDF extraction | `pdfplumber` (primary), `pypdf` (fallback if pdfplumber fails) |
| Config | `python-dotenv` |
| Tests | `pytest` |

No LlamaIndex, no LangChain, no vector database server — the retrieval pipeline (chunking, embedding, FAISS index, similarity search) is implemented directly in `utils/`, which keeps the whole flow inspectable in a few hundred lines rather than behind a framework's abstractions.

## How it works

```
Upload PDFs → extract per-page text (pdfplumber/pypdf) → clean text
  → word-based chunking (overlap, page/subject/section metadata)
  → embed chunks (SentenceTransformers) → FAISS index
                                                │
Question → embed query → FAISS similarity search → top-k chunks
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          │ best match < confidence threshold?         │
                          │  YES → refuse: "not covered in materials"  │
                          │  NO  → send chunks + question to OpenAI    │
                          └─────────────────────┬─────────────────────┘
                                                 │
                        Answer + citation pills (filename, page, section, score)
```

## Features

- Multi-PDF upload with cleaned text extraction (hyphenation repair, whitespace/duplicate-word cleanup for PDF extraction artifacts)
- Word-based chunking with overlap, tracking filename/page/subject/section per chunk — section is detected from heading-like lines (`Unit 2`, `Chapter 3`, `Module 4`, etc.) at the start of each source page, falling back to a filename-guessed unit when no heading is found
- FAISS semantic search with optional subject/section filtering
- Answer-centric chat UI with citation pills (filename, page, section, confidence score) and collapsible retrieved-context view
- **Hallucination guardrail**: if the best-retrieved chunk's similarity score falls below a configurable threshold, Doraeon refuses ("This isn't covered in your uploaded materials.") instead of letting the LLM answer from general knowledge — the LLM is never even called in that case. Tunable via `.env` (`MIN_RETRIEVAL_SCORE`) or live in the sidebar.
- OpenAI synthesis with graceful error handling (missing/invalid key, rate limits, connection errors all produce a clear message instead of a crash)
- Study tools: summarization and flashcard generation from indexed materials
- **Evaluation harness** (`eval/`): precision@k retrieval scoring + LLM-graded answer accuracy against a labeled question set — see `eval/README.md`
- Unit test suite (`tests/`) covering chunking, text cleanup, vector search/filtering, the guardrail, config parsing, and eval precision logic — all offline, no API key or model download required to run

## Project structure

```
doraeon/
├── app.py                  # Streamlit UI + session state
├── requirements.txt
├── requirements-dev.txt    # adds pytest
├── pytest.ini
├── .env.example
├── .streamlit/config.toml
├── utils/
│   ├── config.py           # .env loading, API key + guardrail threshold
│   ├── text_cleaner.py     # PDF text normalization
│   ├── pdf_loader.py       # PDF -> per-page text + metadata
│   ├── chunker.py          # page text -> overlapping word chunks + section detection
│   ├── embeddings.py       # SentenceTransformer wrapper
│   ├── vector_store.py     # FAISS index + similarity search
│   ├── rag_pipeline.py     # retrieval -> guardrail -> OpenAI synthesis
│   └── ui_components.py    # CSS, citation pills, chat layout
├── eval/
│   ├── README.md           # how to run the evaluation harness
│   ├── eval_set.json       # labeled Q&A pairs (fill in with your own materials)
│   ├── run_eval.py         # precision@k + LLM-graded accuracy
│   └── materials/          # your course PDFs for evaluation (gitignored)
└── tests/                  # pytest suite, fully offline
```

## Setup

### 1. Python environment

```powershell
cd d:\Doraeon
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. OpenAI API key (required for AI answers and eval grading)

1. Create an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. Copy the example env file:

```powershell
copy .env.example .env
```

3. Edit `.env` in the **project root** (same folder as `app.py`):

```env
OPENAI_API_KEY=sk-proj-your-real-key-here
OPENAI_MODEL=gpt-4o-mini
MIN_RETRIEVAL_SCORE=0.35
```

4. Restart or refresh the Streamlit app after saving `.env`.

Doraeon loads `.env` automatically via `python-dotenv`. The key must not be the placeholder `your_openai_api_key_here`.

**Without a valid key:** indexing and retrieval still work; the UI shows setup instructions instead of dumping raw chunk walls into the answer.

### 3. Run

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Usage

1. Open the sidebar → **Upload PDFs** → **Build index**.
2. Adjust the **Confidence threshold** slider if needed (higher = stricter about refusing uncertain answers).
3. Go to the **Chat** tab and ask a question.
4. Read the **Answer** card first; expand **Sources** and **View retrieved context** only when needed.

## Evaluation

See `eval/README.md`. In short: put real course PDFs in `eval/materials/`, fill in `eval/eval_set.json` with real question/answer pairs from them, then:

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py
```

Produces `eval/eval_report.md`/`.csv` with per-question precision@k and LLM-judged accuracy, plus aggregate scores.

## Tests

```powershell
pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

40 tests, fully offline — no OpenAI key or model download needed. Covers chunking (including overlap correctness and section-heading detection), text cleanup, FAISS search/filtering, the hallucination guardrail's threshold logic, config parsing, and the eval harness's precision@k function.

## Troubleshooting

| Issue | Fix |
|--------|-----|
| API key not detected | Ensure `.env` is in the project root, not inside `utils/`. Refresh the browser. |
| Invalid API key | Regenerate the key on OpenAI; no extra quotes in `.env`. |
| `429` / "exceeded your current quota" | Your OpenAI account has hit its usage/billing limit — check [platform.openai.com/usage](https://platform.openai.com/usage), not a bug in the app. Retrieval and precision@k still work without it. |
| Rate limit | Wait and retry; use a smaller `top-k`. |
| No text extracted | Scanned PDFs need OCR (not included). |
| Everything gets refused ("not covered in materials") | Lower the confidence threshold slider, or check the material actually got indexed (chunk count in the sidebar). |
| `python` not found on Windows | Use `py` instead. |

## Known limitations

- **Section detection is heuristic**: it matches heading-like lines ("Unit 2", "Chapter 3", etc.) at the start of a source page. Materials without explicit headings fall back to a coarser filename-guessed unit, or no section label at all — this is disclosed in the UI (no section shown) rather than faked.
- **No persistence**: the index lives in Streamlit's session state — it's rebuilt from scratch each time the server restarts or a new session starts. Fine for single-user study sessions, not for a multi-user always-on deployment.
- **Single fixed embedding model/dimension** (`all-MiniLM-L6-v2`, 384-dim) — swapping models means clearing and rebuilding the index, not a live migration.
- **Confidence threshold is a single global cutoff on the top result**, not a per-topic or per-subject calibration — a threshold that works well for one course's material might need retuning for another with very different vocabulary.
- This is a personal/academic-use tool, not hardened for production multi-tenant deployment (no auth, no rate limiting on the OpenAI calls beyond what the API itself enforces).

## License

MIT
