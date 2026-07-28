"""
Evaluation harness for Doraeon: retrieval precision@k + LLM-graded answer accuracy.

Usage:
    python eval/run_eval.py --materials-dir eval/materials --eval-set eval/eval_set.json

Builds a fresh in-memory index from the PDFs in --materials-dir using the exact
same pdf_loader/chunker/embeddings/vector_store/rag_pipeline code the app uses
(nothing reimplemented here), runs every labeled question through the real
RAGPipeline (guardrail included), and reports:

  - precision@k: did a retrieved chunk actually match the expected
    (filename, page) for that question, within a small page tolerance to
    absorb chunk-boundary slop?
  - accuracy: an LLM-as-judge verdict (correct/partial/incorrect) comparing
    the generated answer to your labeled expected_answer. Questions the
    hallucination guardrail refused are reported separately as "refused",
    not graded as wrong -- a refusal on a question that DOES have a real
    answer in your materials is itself a useful signal that the confidence
    threshold may be set too high.

Requires OPENAI_API_KEY in .env for the accuracy/judge parts. Precision@k
still runs without it (retrieval doesn't need the LLM).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI  # noqa: E402

from utils.chunker import chunk_pages  # noqa: E402
from utils.config import get_openai_api_key, get_openai_model, load_settings  # noqa: E402
from utils.embeddings import EmbeddingModel  # noqa: E402
from utils.pdf_loader import load_multiple_pdfs  # noqa: E402
from utils.rag_pipeline import RAGPipeline  # noqa: E402
from utils.vector_store import FaissVectorStore  # noqa: E402

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


def build_index(materials_dir: Path) -> tuple[FaissVectorStore, EmbeddingModel]:
    pdf_paths = sorted(materials_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {materials_dir} -- nothing to index.")
        sys.exit(1)

    files = [(p.read_bytes(), p.name) for p in pdf_paths]
    pages = load_multiple_pdfs(files)
    chunks = chunk_pages(pages)

    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts([c.text for c in chunks])

    store = FaissVectorStore()
    store.add(embeddings, chunks)
    print(f"Indexed {len(pdf_paths)} PDF(s) -> {len(pages)} pages -> {len(chunks)} chunks.")
    return store, embedder


def precision_at_k(sources, expected_filename: str, expected_page: int, page_tolerance: int = 1) -> bool:
    for r in sources:
        c = r.chunk
        if c.filename == expected_filename and abs(c.page_number - expected_page) <= page_tolerance:
            return True
    return False


def grade_with_llm(client: OpenAI, model: str, question: str, expected: str, generated: str) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, generated=generated)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = completion.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001 -- a judge-call failure shouldn't crash the whole eval run
        return "ungraded", f"LLM judge call failed: {e}"

    verdict_match = re.search(r"VERDICT:\s*(correct|partial|incorrect)", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    verdict = verdict_match.group(1).lower() if verdict_match else "ungraded"
    reason = reason_match.group(1).strip() if reason_match else text.strip()[:200]
    return verdict, reason


def run(materials_dir: Path, eval_set_path: Path, top_k: int, page_tolerance: int) -> list[EvalResult]:
    load_settings()
    store, embedder = build_index(materials_dir)
    pipeline = RAGPipeline(store, embedder)

    eval_items = json.loads(eval_set_path.read_text(encoding="utf-8"))
    api_key = get_openai_api_key()
    judge_client = OpenAI(api_key=api_key) if api_key else None
    judge_model = get_openai_model()

    if not api_key:
        print(
            "WARNING: no OpenAI API key configured -- answer generation and accuracy "
            "grading will be skipped. Precision@k will still run (retrieval only)."
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

        if not api_key:
            retrieved, retrieval_ms = pipeline.retrieve(question, top_k=top_k)
            results.append(
                EvalResult(
                    id=item.get("id", ""),
                    question=question,
                    expected_source=expected_source,
                    retrieved_sources="; ".join(f"{r.chunk.filename} p.{r.chunk.page_number}" for r in retrieved),
                    precision_hit=precision_at_k(retrieved, expected_filename, expected_page, page_tolerance),
                    top_score=retrieved[0].score if retrieved else 0.0,
                    generated_answer="(skipped -- no API key)",
                    verdict="ungraded",
                    retrieval_ms=retrieval_ms,
                )
            )
            continue

        resp = pipeline.generate_answer(question, top_k=top_k)
        precision_hit = precision_at_k(resp.sources, expected_filename, expected_page, page_tolerance)
        top_score = resp.sources[0].score if resp.sources else 0.0

        if resp.error == "below_confidence_threshold":
            verdict, reason = "refused", "Guardrail refused to answer (below confidence threshold)."
        else:
            verdict, reason = grade_with_llm(judge_client, judge_model, question, expected_answer, resp.answer)

        results.append(
            EvalResult(
                id=item.get("id", ""),
                question=question,
                expected_source=expected_source,
                retrieved_sources="; ".join(f"{r.chunk.filename} p.{r.chunk.page_number}" for r in resp.sources),
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
