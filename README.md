# Doraeon

An academic RAG (retrieval-augmented generation) assistant. Upload lecture PDFs, ask questions in a chat UI, and get answers grounded in your own materials — with page-level citations, a confidence-based hallucination guardrail (strict refusal or a disclaimed general-knowledge fallback), and an evaluation harness to actually measure retrieval and answer quality rather than just eyeballing it. Runs **fully locally** — no API key, no per-token cost, no internet dependency beyond the one-time model downloads.

## Stack

| Layer | Technology |
|---|---|
| UI | **Streamlit** |
| Retrieval framework | **LlamaIndex** (`VectorStoreIndex`, `SentenceSplitter`) |
| Embeddings | **BAAI/bge-small-en-v1.5** (local, via `sentence-transformers`/`llama-index-embeddings-huggingface`, 384-dim) |
| Vector search | **FAISS** (`IndexFlatIP` — inner product on L2-normalized vectors = cosine similarity) |
| LLM synthesis + eval grading | **Ollama** (local model, e.g. `llama3.2` — no API key, no cost) |
| PDF extraction | `pdfplumber` (primary), `pypdf` (fallback), OCR fallback (`pypdfium2` + `pytesseract` + Tesseract) for scanned/broken-font pages |
| Config | `python-dotenv` |
| Tests | `pytest` |

## How it works

```
Upload PDFs → per-page text extraction (pdfplumber/pypdf, OCR fallback for
  unreadable pages) → word-based chunking (LlamaIndex SentenceSplitter,
  page/subject/unit metadata) → embed chunks (bge-small-en-v1.5) → FAISS index
                                                │
Question → condense follow-up using chat history → embed query → FAISS
  similarity search → top-k chunks
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          │ best match < confidence threshold?         │
                          │  strict mode → refuse: "not covered"       │
                          │  soft mode   → answer from general         │
                          │                knowledge, clearly disclaimed│
                          │  NO  → send chunks + question to Ollama    │
                          └─────────────────────┬─────────────────────┘
                                                 │
                        Answer + citation pills (filename, page, unit, score)
```

## Features

- Multi-PDF upload with per-page text extraction and an OCR fallback (rasterize + Tesseract) for pages a broken font or scanner leaves unreadable — triggers only on pages that actually fail normal extraction, at zero cost to healthy PDFs
- Word-based chunking with overlap, tracking filename/page/subject/unit per chunk — unit is detected from heading-like lines (`Unit 2`, `Chapter 3`, `Module 4`, etc.) in each chunk, falling back to a filename-guessed unit when no heading is found
- FAISS semantic search with optional subject/unit filtering
- Conversational memory: vague follow-ups ("explain the gist of it") are condensed into a standalone query using recent chat history before retrieval
- Answer-centric chat UI with citation pills (filename, page, unit, confidence score) and a collapsible retrieved-context view
- **Hallucination guardrail** with a strict/soft toggle: if the best-retrieved chunk's similarity score falls below a configurable threshold —
  - **Strict mode**: refuses ("This isn't covered in your uploaded materials.") without calling the LLM at all — best for exam-prep grounding.
  - **Soft mode** (default): answers from the LLM's general knowledge instead, clearly prefixed ("This isn't directly covered in your uploaded materials, but generally speaking: …") and never shown with source attribution, since there isn't any.
  - Both modes still surface the closest (below-threshold) matches for transparency.
  - Tunable via `.env` (`MIN_RETRIEVAL_SCORE`) or live in the sidebar.
- Fully local LLM synthesis via Ollama, with graceful error handling (server unreachable, model not pulled, or any other failure produce a clear message instead of a crash)
- Study tools: summarization, flashcard generation, and exam-question prediction from indexed materials — same retrieval/guardrail pipeline as chat
- **Evaluation harness** (`eval/`): precision@k retrieval scoring + LLM-graded answer accuracy against a labeled question set — see `eval/README.md`
- Unit test suite (`tests/`) covering ingestion (including the OCR fallback), chunking, retrieval scoring, the guardrail (strict + soft mode), config parsing, and eval precision logic — all offline via mocks, no live Ollama server or model download required to run

## Project structure

```
doraeon/
├── app.py                  # Streamlit UI + session state
├── requirements.txt
├── requirements-dev.txt    # adds pytest
├── pytest.ini
├── .env.example
├── .streamlit/config.toml
├── core/
│   ├── config.py           # .env loading, Ollama + guardrail + embedding config
│   ├── text_cleaner.py     # PDF text normalization
│   ├── ingestion.py        # PDF -> per-page Documents + metadata, OCR fallback
│   ├── index_store.py      # LlamaIndex/FAISS index, chunking, embedding model
│   ├── retrieval.py        # similarity search + confidence normalization
│   ├── query_rewriting.py  # condense follow-ups using chat history
│   ├── llm_client.py       # Ollama chat wrapper, error handling, status probe
│   ├── rag_pipeline.py     # retrieval -> guardrail -> Ollama synthesis
│   ├── study_tools.py      # summarize / flashcards / exam questions
│   └── ui_components.py    # CSS, citation pills, chat layout
├── eval/
│   ├── README.md           # how to run the evaluation harness
│   ├── eval_set.json       # labeled Q&A pairs (fill in with your own materials)
│   ├── run_eval.py         # precision@k + LLM-graded accuracy
│   └── materials/          # your course PDFs for evaluation (gitignored)
└── tests/                  # pytest suite, offline via mocks
```

## Setup

### 1. Python environment

```powershell
cd d:\Doraeon
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Ollama (local LLM — required for AI answers and eval grading)

1. Install Ollama: [ollama.com/download](https://ollama.com/download).
2. Pull a model:

```powershell
ollama pull llama3.2
```

`llama3.2` (3B) is the default — it's fast enough for interactive use on a typical laptop with no dedicated GPU. If your machine has 16GB+ RAM and you want noticeably better answer quality/instruction-following at the cost of being roughly 2-4x slower per response, pull `mistral` (7B) instead and set `OLLAMA_MODEL=mistral` below.

3. Start the server (if it isn't already running as a background service):

```powershell
ollama serve
```

4. Copy the example env file and adjust if needed:

```powershell
copy .env.example .env
```

```env
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
MIN_RETRIEVAL_SCORE=0.35
```

**Without Ollama running:** indexing and retrieval still work (embeddings are local and independent of Ollama); the UI shows setup instructions instead of failing generation silently.

### 3. Run

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Usage

1. Open the sidebar → **Upload PDFs** → **Build index**.
2. Adjust the **Confidence threshold** slider if needed (higher = stricter about refusing/falling back on uncertain answers), and toggle **Strict mode** depending on whether you want hard refusals (exam-prep grounding) or general-knowledge fallback (general help).
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

Fully offline via mocks — no live Ollama server or model download needed to run (the first run does download the local embedding model, `bge-small-en-v1.5`, ~130MB, once). Covers ingestion (including the OCR fallback), retrieval scoring against real embeddings, the hallucination guardrail's strict/soft logic, config parsing, and the eval harness's precision@k function.

## Troubleshooting

| Issue | Fix |
|--------|-----|
| Sidebar shows "Ollama isn't reachable" | Make sure Ollama is installed and `ollama serve` is running. Check `OLLAMA_BASE_URL` in `.env` if it's not on the default port. |
| "'model' isn't pulled yet" | Run `ollama pull <model>` for whatever `OLLAMA_MODEL` is set to. |
| Answers are slow | Local models are CPU/GPU-bound by your hardware, not a fixed API latency — try the smaller `llama3.2` if you're on `mistral`, or reduce `top_k`. |
| No text extracted, OCR not attempted | Check Tesseract is installed system-wide (`tesseract --version`) — see `eval/README.md` install notes; OCR only runs when both text extractors fail on a page. |
| Everything gets refused/falls back to general knowledge | Lower the confidence threshold slider, or check the material actually got indexed (chunk count in the sidebar). |
| `python` not found on Windows | Use `py` instead. |

## Known limitations

- **Unit detection is heuristic**: it matches heading-like lines ("Unit 2", "Chapter 3", etc.) at the start of a chunk. Materials without explicit headings fall back to a coarser filename-guessed unit, or no unit label at all — this is disclosed in the UI rather than faked.
- **No persistence**: the index lives in Streamlit's session state — it's rebuilt from scratch each time the server restarts or a new session starts. Fine for single-user study sessions, not for a multi-user always-on deployment.
- **Single fixed embedding model/dimension** (`bge-small-en-v1.5`, 384-dim) — swapping models means clearing and rebuilding the index, not a live migration.
- **Confidence threshold is a single global cutoff on the top result**, not a per-topic or per-subject calibration — a threshold that works well for one course's material might need retuning for another with very different vocabulary.
- **Local model quality/speed depends entirely on your hardware and chosen model** — a 3B model on a CPU-only laptop will be faster but less capable than a 7B+ model on a machine with a GPU.
- This is a personal/academic-use tool, not hardened for production multi-tenant deployment (no auth, no rate limiting).

## License

MIT
