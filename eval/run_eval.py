"""
Evaluation harness for Doraeon: retrieval precision@k + LLM-graded answer accuracy.

Usage:
    python eval/run_eval.py --materials-dir eval/materials --eval-set eval/eval_set.json

Builds a fresh index from the PDFs in --materials-dir using the exact same
ingestion/index_store/rag_pipeline code the app uses (nothing reimplemented
here), runs every labeled question through the real RAGPipeline (guardrail
included), and reports precision@k, LLM-graded accuracy, and guardrail
refusals as a distinct category (see eval/README.md for details).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_ollama_base_url, get_ollama_model, load_settings  # noqa: E402
from core.ingestion import load_multiple_pdfs  # noqa: E402
from core.index_store import DoraeonIndex  # noqa: E402
from core.llm_client import build_llm, call_chat_completion, check_ollama_status, error_message  # noqa: E402
from core.rag_pipeline import RAGPipeline  # noqa: E402

JUDGE_PROMPT = """You are grading a student-facing academic answer for correctness.

Question: {question}

Expected answer (ground truth, from the course materials): {expected}

Generated answer (to grade): {generated}

Judge whether the generated answer is correct, partially correct, or incorrect
relative to the expected answer. Minor wording differences don't matter --
judge factual/conceptual correctness only.

Respond in EXACTLY this format, nothing else:
VERDICT: correct|partial|incorrect
REASON: <one concise sentence>
"""


@dataclass
class EvalResult:
    id: str
    question: str
    expected_source: str
    retrieved_sources: str
    precision_hit: bool
    top_score: float
    generated_answer: str
    verdict: str  # "correct" | "partial" | "incorrect" | "refused" | "ungraded"
    reason: str = ""
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0


def build_index(materials_dir: Path) -> DoraeonIndex:
    pdf_paths = sorted(materials_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {materials_dir} -- nothing to index.")
        sys.exit(1)

    files = [(p.read_bytes(), p.name) for p in pdf_paths]
    documents, issues = load_multiple_pdfs(files)
    if issues:
        print(f"WARNING: {len(issues)} page(s) could not be read cleanly and were skipped:")
        for issue in issues:
            print(f"  - {issue.filename} p.{issue.page_number}: {issue.reason}")

    idx = DoraeonIndex()
    n_chunks = idx.add_documents(documents)
    print(f"Indexed {len(pdf_paths)} PDF(s) -> {len(documents)} pages -> {n_chunks} chunks.")
    return idx


def precision_at_k(sources, expected_filename: str, expected_page: int, page_tolerance: int = 1) -> bool:
    for c in sources:
        if c.filename == expected_filename and abs(c.page_number - expected_page) <= page_tolerance:
            return True
    return False


def grade_with_llm(judge_llm, question: str, expected: str, generated: str) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, generated=generated)
    text, _ms, err = call_chat_completion(judge_llm, [{"role": "user", "content": prompt}])
    if err:
        return "ungraded", f"LLM judge call failed: {error_message(err)}"

    verdict_match = re.search(r"VERDICT:\s*(correct|partial|incorrect)", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    verdict = verdict_match.group(1).lower() if verdict_match else "ungraded"
    reason = reason_match.group(1).strip() if reason_match else text.strip()[:200]
    return verdict, reason


def run(materials_dir: Path, eval_set_path: Path, top_k: int, page_tolerance: int) -> list[EvalResult]:
    load_settings()
    idx = build_index(materials_dir)
    pipeline = RAGPipeline(idx)

    eval_items = json.loads(eval_set_path.read_text(encoding="utf-8"))
    judge_model = get_ollama_model()
    # temperature=0.0, separate from the pipeline's own LLM (which uses 0.2) --
    # deterministic grading, and Ollama fixes temperature at construction time.
    judge_llm = build_llm(judge_model, get_ollama_base_url(), temperature=0.0)

    ollama_ready, status_msg = check_ollama_status(judge_model, get_ollama_base_url())
    if not ollama_ready:
        print(
            f"WARNING: {status_msg}\nAnswer generation and accuracy grading will fail per-question "
            "(reported as 'ungraded' below). Precision@k still runs -- it's pure retrieval, no LLM needed."
        )

    results: list[EvalResult] = []
    for item in eval_items:
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_filename = item.get("expected_source_filename", "")
        expected_page = int(item.get("expected_source_page", 0))
        expected_source = f"{expected_filename} p.{expected_page}"

        if "REPLACE ME" in question:
            print(f"Skipping placeholder item '{item.get('id', '?')}' -- fill in eval_set.json first.")
            continue

        resp = pipeline.generate_answer(question, top_k=top_k)
        precision_hit = precision_at_k(resp.sources, expected_filename, expected_page, page_tolerance)
        top_score = resp.sources[0].confidence if resp.sources else 0.0

        if resp.error == "below_confidence_threshold":
            verdict, reason = "refused", "Guardrail refused to answer (below confidence threshold)."
        elif resp.error:
            verdict, reason = "ungraded", f"Generation failed: {error_message(resp.error)}"
        else:
            verdict, reason = grade_with_llm(judge_llm, question, expected_answer, resp.answer)

        results.append(
            EvalResult(
                id=item.get("id", ""),
                question=question,
                expected_source=expected_source,
                retrieved_sources="; ".join(f"{c.filename} p.{c.page_number}" for c in resp.sources),
                precision_hit=precision_hit,
                top_score=top_score,
                generated_answer=resp.answer,
                verdict=verdict,
                reason=reason,
                retrieval_ms=resp.retrieval_ms,
                generation_ms=resp.generation_ms,
            )
        )
    return results


def write_reports(results: list[EvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_report.csv"
    md_path = output_dir / "eval_report.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "question", "expected_source", "retrieved_sources", "precision_hit",
             "top_score", "verdict", "reason", "generated_answer", "retrieval_ms", "generation_ms"]
        )
        for r in results:
            writer.writerow(
                [r.id, r.question, r.expected_source, r.retrieved_sources, r.precision_hit,
                 f"{r.top_score:.3f}", r.verdict, r.reason, r.generated_answer,
                 f"{r.retrieval_ms:.0f}", f"{r.generation_ms:.0f}"]
            )

    n = len(results)
    precision_hits = sum(1 for r in results if r.precision_hit)
    graded = [r for r in results if r.verdict in {"correct", "partial", "incorrect"}]
    refused = [r for r in results if r.verdict == "refused"]
    ungraded = [r for r in results if r.verdict == "ungraded"]
    verdict_score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    avg_accuracy = sum(verdict_score[r.verdict] for r in graded) / len(graded) if graded else None

    lines = ["# Doraeon Evaluation Report", ""]
    lines.append(f"- Questions evaluated: {n}")
    lines.append(f"- Precision@k: {precision_hits}/{n} ({precision_hits / n:.0%})" if n else "- Precision@k: n/a")
    if avg_accuracy is not None:
        lines.append(f"- Average accuracy (graded questions only, correct=1/partial=0.5/incorrect=0): {avg_accuracy:.0%}")
    lines.append(f"- Refused by hallucination guardrail: {len(refused)}/{n}")
    if ungraded:
        lines.append(f"- Ungraded (no API key or judge call failed): {len(ungraded)}/{n}")
    lines.append("")
    lines.append("| ID | Question | Precision@k | Top score | Verdict | Reason |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        hit = "✅" if r.precision_hit else "❌"
        q = r.question[:60] + ("…" if len(r.question) > 60 else "")
        reason = r.reason[:80] + ("…" if len(r.reason) > 80 else "")
        lines.append(f"| {r.id} | {q} | {hit} | {r.top_score:.2f} | {r.verdict} | {reason} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {csv_path} and {md_path}")
    print(f"Precision@k: {precision_hits}/{n}" if n else "No questions evaluated.")
    if avg_accuracy is not None:
        print(f"Average accuracy: {avg_accuracy:.0%} (over {len(graded)} graded, {len(refused)} refused)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Doraeon's retrieval and answer quality.")
    parser.add_argument("--materials-dir", type=Path, default=Path(__file__).parent / "materials")
    parser.add_argument("--eval-set", type=Path, default=Path(__file__).parent / "eval_set.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--page-tolerance", type=int, default=1)
    args = parser.parse_args()

    if not args.materials_dir.exists():
        print(f"Materials directory {args.materials_dir} doesn't exist -- create it and add your course PDFs.")
        sys.exit(1)

    results = run(args.materials_dir, args.eval_set, args.top_k, args.page_tolerance)
    if not results:
        print("No questions were evaluated -- check eval_set.json has real (non-placeholder) entries.")
        sys.exit(1)
    write_reports(results, args.output_dir)
