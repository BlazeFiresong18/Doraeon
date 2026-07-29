# Doraeon evaluation harness

## 1. Add your materials

Create `eval/materials/` and drop in the actual course PDFs you want to evaluate against (the same files you'd upload in the app).

## 2. Fill in eval_set.json

Replace the placeholder entries in `eval_set.json` with 20-30 real question/answer pairs from those PDFs:

```json
{
  "id": "q1",
  "question": "What is the time complexity of binary search?",
  "expected_answer": "O(log n)",
  "expected_source_filename": "Lecture3.pdf",
  "expected_source_page": 7
}
```

`expected_source_filename`/`expected_source_page` should point at the page where the answer actually appears — this is what precision@k checks against.

## 3. Run it

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py
```

Optional flags: `--materials-dir`, `--eval-set`, `--top-k` (default 5), `--page-tolerance` (default ±1 page).

## What it measures

- **Precision@k**: did the real retrieval pipeline actually surface the correct chunk in its top-k results for each question?
- **Accuracy**: an LLM-as-judge call (your local Ollama model, configured via `OLLAMA_MODEL`) grades each generated answer against your `expected_answer` as correct/partial/incorrect. Uses a separate temperature=0.0 instance for deterministic grading.
- **Refused**: questions the hallucination guardrail declined to answer (below the confidence threshold). A refusal on a question that *does* have a real answer in your materials is a sign the threshold may be set too high — not counted as "wrong," reported separately so it's visible.

Requires Ollama installed and running (`ollama serve`) with the configured model pulled. Without it, generation and grading fail per-question (reported as "ungraded"); precision@k still runs regardless since it's pure retrieval, no LLM needed.

Output: `eval_report.md` (human-readable) and `eval_report.csv` (for a spreadsheet), both in `eval/` by default.
