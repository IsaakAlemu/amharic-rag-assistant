"""Phase 5: bounded 10-question end-to-end grounded generation pilot (eval only)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.errors import GenerationError
from src.eval_utils import load_eval_qas, normalize_text, save_json
from src.history_manager import ConversationState, HistoryManager, format_history_for_prompt
from src.input_validation import validate_query
from src.llm import REFUSAL_PHRASE, GenerationResult, generate_answer
from src.logging_config import setup_logging
from src.pipeline import load_vector_collection
from src.prompt_builder import build_conversational_prompt
from src.query_rewriter import rewrite_query
from src.retriever import retrieve
from src.token_counter import TokenCounter

PILOT_SIZE = 10
MAX_GENERATION_CALLS = 10
MAX_RETRIES_PER_CALL = 3
MAX_RUNTIME_SECONDS = 600
MAX_RATE_LIMIT_ERRORS = 2
OUTPUT_PATH = "results/phase5_generation_pilot_10.json"
BASELINE_PATH = "results/normalization_experiment_baseline.json"

# Deterministic spread through holdout baseline (5 retrieval hits, 5 misses).
SUCCESS_INDICES = [0, 48, 96, 144, 192]
FAILURE_INDICES = [0, 17, 34, 51, 68]


def gold_rank(sources: list[dict], gold_id: str) -> int | None:
    for doc in sources:
        if doc["document_id"] == gold_id:
            return doc["rank"]
    return None


def select_pilot_question_ids(
    baseline_rows: list[dict[str, Any]],
    question_to_id: dict[str, int],
) -> list[int]:
    successes: list[int] = []
    failures: list[int] = []
    for row in baseline_rows:
        qid = question_to_id.get(row["question"])
        if qid is None:
            continue
        if row.get("hit_at_1"):
            successes.append(qid)
        else:
            failures.append(qid)

    selected: list[int] = []
    for idx in SUCCESS_INDICES:
        if idx < len(successes):
            selected.append(successes[idx])
    for idx in FAILURE_INDICES:
        if idx < len(failures):
            selected.append(failures[idx])

    if len(selected) != PILOT_SIZE:
        raise RuntimeError(
            f"Expected {PILOT_SIZE} pilot question IDs, selected {len(selected)}."
        )
    return selected


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or ("rate" in message and "limit" in message)


def tag_groundedness_heuristic(
    answer: str,
    sources: list[dict],
    ground_truth: str,
    *,
    refusal: bool,
) -> str:
    """Pilot-only heuristic label for manual follow-up; not a metric."""
    if refusal or REFUSAL_PHRASE in answer:
        return "unable_to_answer_refusal"

    if not answer.strip():
        return "unable_to_answer_refusal"

    context = normalize_text(" ".join(source["text"] for source in sources))
    answer_norm = normalize_text(answer)
    gt_norm = normalize_text(ground_truth) if ground_truth else ""

    if gt_norm and gt_norm in answer_norm:
        return "clearly_supported"

    answer_tokens = {t for t in answer_norm.split() if len(t) >= 3}
    context_tokens = set(context.split())
    if not answer_tokens:
        return "unsupported_hallucinated"

    overlap_ratio = len(answer_tokens & context_tokens) / len(answer_tokens)
    if overlap_ratio >= 0.45:
        return "clearly_supported"
    if overlap_ratio >= 0.15:
        return "partially_supported"
    return "unsupported_hallucinated"


def generate_with_retry(
    prompt: str,
    client: Groq,
    *,
    model: str,
    temperature: float,
    generation_budget: dict[str, int],
    rate_limit_counter: dict[str, int],
) -> tuple[GenerationResult | None, int, str | None]:
    retries = 0
    last_error: str | None = None

    while True:
        if generation_budget["used"] >= MAX_GENERATION_CALLS:
            return None, retries, "Generation call budget exhausted (10 max)."

        generation_budget["used"] += 1
        try:
            result = generate_answer(
                prompt,
                client,
                model=model,
                temperature=temperature,
            )
            return result, retries, None
        except GenerationError as exc:
            last_error = str(exc)
            if is_rate_limit_error(exc):
                rate_limit_counter["count"] += 1
                if rate_limit_counter["count"] >= MAX_RATE_LIMIT_ERRORS:
                    raise RuntimeError(
                        "Repeated rate-limit errors from generation API; stopping pilot."
                    ) from exc
            if retries >= MAX_RETRIES_PER_CALL:
                return None, retries, last_error
            if generation_budget["used"] >= MAX_GENERATION_CALLS:
                return None, retries, last_error
            retries += 1
            time.sleep(min(2.0 * retries, 6.0))


def run_pilot() -> dict[str, Any]:
    settings = get_settings(require_groq=True)
    eval_settings = replace(settings, llm_model=settings.model_70b)
    setup_logging(settings.log_level)

    experiment_start = time.perf_counter()
    baseline_path = ROOT / BASELINE_PATH
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline_payload = json.load(handle)

    holdout_qas = load_eval_qas(holdout_split_path(settings))
    question_to_id = {qa.question: qa.question_id for qa in holdout_qas if qa.question_id}
    qa_by_id = {qa.question_id: qa for qa in holdout_qas if qa.question_id is not None}

    pilot_ids = select_pilot_question_ids(baseline_payload["per_question"], question_to_id)
    print("Selected pilot question IDs:")
    for qid in pilot_ids:
        print(f"  {qid}")

    client = Groq(api_key=settings.groq_api_key)
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    token_counter = TokenCounter()
    history_manager = HistoryManager(
        max_turns=settings.max_history_turns,
        max_history_tokens=settings.max_history_tokens,
    )

    generation_budget = {"used": 0}
    rate_limit_counter = {"count": 0}
    per_question: list[dict[str, Any]] = []
    stopped_reason = "completed"

    for qid in pilot_ids:
        elapsed = time.perf_counter() - experiment_start
        if elapsed >= MAX_RUNTIME_SECONDS:
            stopped_reason = "max_runtime_exceeded"
            break

        qa = qa_by_id[qid]
        row: dict[str, Any] = {
            "question_id": qid,
            "original_question": qa.question,
            "gold_document_id": qa.document_id,
            "ground_truth": qa.ground_truth,
        }

        try:
            cleaned_query = validate_query(qa.question, max_chars=settings.max_query_chars)
        except Exception as exc:
            row.update(
                {
                    "generation_success": False,
                    "generation_error": str(exc),
                    "groundedness_heuristic": "unable_to_answer_refusal",
                }
            )
            per_question.append(row)
            continue

        conversation = ConversationState()
        conversation.add_user(cleaned_query)
        rewrite_history = history_manager.select_for_rewrite(conversation.messages)
        prompt_history = history_manager.select_for_prompt(conversation.messages)

        rewrite = rewrite_query(
            cleaned_query,
            rewrite_history,
            client=client,
            model=settings.rewrite_model,
            temperature=0.0,
        )
        row["rewritten_query"] = rewrite.rewritten_query
        row["rewrite_applied"] = rewrite.rewritten_query.strip() != cleaned_query.strip()

        sources = retrieve(
            rewrite.rewritten_query,
            collection,
            embed_model,
            top_k=settings.top_k,
        )
        row["retrieved_document_ids"] = [s["document_id"] for s in sources]
        row["retrieved_ranks"] = [s["rank"] for s in sources]
        row["gold_document_rank"] = gold_rank(sources, qa.document_id)

        if not sources:
            row.update(
                {
                    "generated_answer": REFUSAL_PHRASE,
                    "generation_success": True,
                    "generation_skipped": True,
                    "generation_latency_ms": 0.0,
                    "retries": 0,
                    "groundedness_heuristic": "unable_to_answer_refusal",
                }
            )
            per_question.append(row)
            continue

        history_text = format_history_for_prompt(prompt_history)
        prompt = build_conversational_prompt(cleaned_query, sources, history_text)
        row["prompt_tokens_estimated"] = token_counter.count(prompt)

        gen_start = time.perf_counter()
        generation, retries, gen_error = generate_with_retry(
            prompt,
            client,
            model=eval_settings.llm_model,
            temperature=eval_settings.temperature,
            generation_budget=generation_budget,
            rate_limit_counter=rate_limit_counter,
        )
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000

        if generation is None:
            row.update(
                {
                    "generated_answer": "",
                    "generation_success": False,
                    "generation_error": gen_error,
                    "generation_latency_ms": round(gen_latency_ms, 2),
                    "retries": retries,
                    "groundedness_heuristic": "unable_to_answer_refusal",
                }
            )
            per_question.append(row)
            continue

        refusal = REFUSAL_PHRASE in generation.text
        groundedness = tag_groundedness_heuristic(
            generation.text,
            sources,
            qa.ground_truth,
            refusal=refusal,
        )

        row.update(
            {
                "generated_answer": generation.text,
                "generation_success": True,
                "generation_model": generation.model,
                "generation_latency_ms": round(gen_latency_ms, 2),
                "retries": retries,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.total_tokens,
                "refusal": refusal,
                "groundedness_heuristic": groundedness,
            }
        )
        per_question.append(row)

    total_runtime = time.perf_counter() - experiment_start
    successful = [r for r in per_question if r.get("generation_success")]
    failed = [r for r in per_question if not r.get("generation_success")]

    prompt_tokens = [r["prompt_tokens"] for r in successful if r.get("prompt_tokens") is not None]
    completion_tokens = [
        r["completion_tokens"] for r in successful if r.get("completion_tokens") is not None
    ]
    latencies = [r["generation_latency_ms"] for r in per_question if "generation_latency_ms" in r]

    grounded_counts = {
        "clearly_supported": 0,
        "partially_supported": 0,
        "unsupported_hallucinated": 0,
        "unable_to_answer_refusal": 0,
    }
    for row in per_question:
        key = row.get("groundedness_heuristic", "unable_to_answer_refusal")
        grounded_counts[key] = grounded_counts.get(key, 0) + 1

    summary = {
        "questions_target": PILOT_SIZE,
        "questions_processed": len(per_question),
        "successful_generations": len(successful),
        "failed_generations": len(failed),
        "rate_limit_errors": rate_limit_counter["count"],
        "generation_api_calls_used": generation_budget["used"],
        "average_generation_latency_ms": round(sum(latencies) / len(latencies), 2)
        if latencies
        else 0.0,
        "total_runtime_seconds": round(total_runtime, 2),
        "average_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 2)
        if prompt_tokens
        else None,
        "average_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 2)
        if completion_tokens
        else None,
        "groundedness_heuristic_counts": grounded_counts,
    }

    payload = {
        "experiment": "phase5_generation_pilot_10",
        "description": "Bounded end-to-end RAG generation pilot on 10 holdout questions",
        "eval_only": True,
        "production_code_modified": False,
        "stopped_reason": stopped_reason,
        "constraints": {
            "pilot_questions": PILOT_SIZE,
            "max_generation_api_calls": MAX_GENERATION_CALLS,
            "max_retries_per_call": MAX_RETRIES_PER_CALL,
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "generation_model": eval_settings.llm_model,
            "rewrite_model": settings.rewrite_model,
            "embed_model": settings.embed_model,
            "top_k": settings.top_k,
            "no_llm_judge": True,
        },
        "selection": {
            "method": "deterministic spread from normalization_experiment_baseline.json",
            "success_indices": SUCCESS_INDICES,
            "failure_indices": FAILURE_INDICES,
            "selected_question_ids": pilot_ids,
        },
        "summary": summary,
        "per_question": per_question,
    }
    save_json(OUTPUT_PATH, payload)
    return payload


def main() -> None:
    try:
        payload = run_pilot()
    except RuntimeError as exc:
        print(f"PILOT STOPPED: {exc}")
        raise SystemExit(1) from exc

    s = payload["summary"]
    print("\n" + "=" * 60)
    print("Phase 5 generation pilot — COMPLETE")
    print("=" * 60)
    print(f"Successful generations: {s['successful_generations']}")
    print(f"Failed generations: {s['failed_generations']}")
    print(f"Rate-limit errors: {s['rate_limit_errors']}")
    print(f"Generation API calls used: {s['generation_api_calls_used']}/{MAX_GENERATION_CALLS}")
    print(f"Average generation latency: {s['average_generation_latency_ms']:.1f} ms")
    print(f"Total runtime: {s['total_runtime_seconds']:.1f} s")
    if s["average_prompt_tokens"] is not None:
        print(f"Average prompt tokens: {s['average_prompt_tokens']:.1f}")
    if s["average_completion_tokens"] is not None:
        print(f"Average completion tokens: {s['average_completion_tokens']:.1f}")

    counts = s["groundedness_heuristic_counts"]
    print(f"Clearly grounded (heuristic): {counts.get('clearly_supported', 0)}")
    print(f"Partially grounded (heuristic): {counts.get('partially_supported', 0)}")
    print(f"Unsupported (heuristic): {counts.get('unsupported_hallucinated', 0)}")
    print(f"Refusals (heuristic): {counts.get('unable_to_answer_refusal', 0)}")

    print("\nGenerated answers for manual inspection:")
    for row in payload["per_question"]:
        print("-" * 60)
        print(f"question_id={row['question_id']}")
        print(f"question: {row['original_question']}")
        if row.get("rewritten_query"):
            print(f"rewritten_query: {row['rewritten_query']}")
        print(f"retrieved_document_ids: {row.get('retrieved_document_ids', [])}")
        print(f"gold_document_rank: {row.get('gold_document_rank')}")
        print(f"groundedness_heuristic: {row.get('groundedness_heuristic')}")
        print(f"generated_answer: {row.get('generated_answer', '')}")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
