"""Phase 5B: bounded 30-question end-to-end RAG evaluation (eval only)."""

from __future__ import annotations

import json
import re
import statistics
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
from src.eval_utils import EvalQA, load_eval_qas, save_json
from src.history_manager import ConversationState, HistoryManager, format_history_for_prompt
from src.input_validation import validate_query
from src.llm import REFUSAL_PHRASE, GenerationResult, generate_answer
from src.logging_config import setup_logging
from src.pipeline import load_vector_collection
from src.prompt_builder import build_conversational_prompt
from src.query_rewriter import rewrite_query
from src.retriever import retrieve
from src.token_counter import TokenCounter

GROUP_SIZE = 10
TOTAL_QUESTIONS = 30
MAX_GENERATION_CALLS = 30
MAX_RETRIES_PER_CALL = 3
MAX_RUNTIME_SECONDS = 1200
MAX_RATE_LIMIT_ERRORS = 2
BASELINE_PATH = "results/normalization_experiment_baseline.json"
OUTPUT_JSON = "results/phase5_end_to_end_30.json"
OUTPUT_MD = "results/phase5_end_to_end_30_manual_review.md"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def baseline_gold_rank(row: dict[str, Any]) -> int | None:
    gold = row["gold_document_id"]
    retrieved = row["retrieved_document_ids"]
    if row.get("hit_at_1"):
        return 1
    if gold in retrieved:
        return retrieved.index(gold) + 1
    return None


def classify_baseline_group(rank: int | None) -> str:
    if rank == 1:
        return "A_gold_rank_1"
    if rank in (2, 3):
        return "B_gold_rank_2_3"
    return "C_gold_outside_top_3"


def pick_spread(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) < count:
        raise RuntimeError(f"Need {count} items, only {len(items)} available.")
    if len(items) == count:
        return list(items)
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def select_questions(
    baseline_rows: list[dict[str, Any]],
    question_to_qa: dict[str, EvalQA],
) -> list[dict[str, Any]]:
    group_a: list[dict[str, Any]] = []
    group_b: list[dict[str, Any]] = []
    group_c: list[dict[str, Any]] = []

    for row in baseline_rows:
        qa = question_to_qa.get(row["question"])
        if qa is None or qa.question_id is None:
            continue
        rank = baseline_gold_rank(row)
        entry = {
            "question_id": qa.question_id,
            "baseline_gold_rank": rank,
            "sampling_group": classify_baseline_group(rank),
            "gold_document_id": row["gold_document_id"],
            "qa": qa,
        }
        if rank == 1:
            group_a.append(entry)
        elif rank in (2, 3):
            group_b.append(entry)
        else:
            group_c.append(entry)

    selected = (
        pick_spread(group_a, GROUP_SIZE)
        + pick_spread(group_b, GROUP_SIZE)
        + pick_spread(group_c, GROUP_SIZE)
    )
    if len(selected) != TOTAL_QUESTIONS:
        raise RuntimeError(f"Expected {TOTAL_QUESTIONS} selected questions, got {len(selected)}.")
    return selected


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or ("rate" in message and "limit" in message)


def parse_citations(answer: str) -> list[int]:
    return [int(x) for x in CITATION_PATTERN.findall(answer or "")]


def citation_mapping(citations: list[int], retrieved_ids: list[str]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for rank in citations:
        index = rank - 1
        doc_id = retrieved_ids[index] if 0 <= index < len(retrieved_ids) else None
        mapped.append({"citation": rank, "document_id": doc_id})
    return mapped


def gold_rank_live(sources: list[dict], gold_id: str) -> int | None:
    for doc in sources:
        if doc["document_id"] == gold_id:
            return doc["rank"]
    return None


def serialize_sources(sources: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "rank": s["rank"],
            "document_id": s["document_id"],
            "distance": s.get("distance"),
            "text": s["text"],
        }
        for s in sources
    ]


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
            return None, retries, "Generation call budget exhausted (30 max)."
        generation_budget["used"] += 1
        try:
            return (
                generate_answer(prompt, client, model=model, temperature=temperature),
                retries,
                None,
            )
        except GenerationError as exc:
            last_error = str(exc)
            if is_rate_limit_error(exc):
                rate_limit_counter["count"] += 1
                if rate_limit_counter["count"] >= MAX_RATE_LIMIT_ERRORS:
                    raise RuntimeError(
                        "Repeated rate-limit errors from generation API; stopping experiment."
                    ) from exc
            if retries >= MAX_RETRIES_PER_CALL:
                return None, retries, last_error
            if generation_budget["used"] >= MAX_GENERATION_CALLS:
                return None, retries, last_error
            retries += 1
            time.sleep(min(2.0 * retries, 6.0))


def build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Phase 5B — End-to-End RAG Evaluation (30 Questions)",
        "",
        f"**Source JSON:** `{OUTPUT_JSON}`",
        f"**Stopped reason:** {payload.get('stopped_reason', 'completed')}",
        "",
        "---",
        "",
    ]
    for case_num, row in enumerate(payload["per_question"], start=1):
        retrieved = row.get("retrieved_passages", [])
        citations = row.get("parsed_citations", [])
        citation_map = row.get("citation_document_mapping", [])
        lines.extend(
            [
                f"## Case {case_num} — question_id {row['question_id']} ({row['sampling_group']})",
                "",
                "### Question",
                row["original_question"],
                "",
                "### Rewritten query",
                row.get("rewritten_query", ""),
                "",
                "### Retrieval",
                f"- **Gold document ID:** `{row['gold_document_id']}`",
                f"- **Baseline gold rank (artifact):** {row.get('baseline_gold_rank')}",
                f"- **Live gold rank:** {row.get('live_gold_rank')}",
                "",
                "**Top retrieved documents:**",
                "",
                "| Rank | Document ID | Distance |",
                "|------|-------------|----------|",
            ]
        )
        for passage in retrieved:
            dist = passage.get("distance")
            dist_s = f"{dist:.4f}" if isinstance(dist, (int, float)) else "—"
            lines.append(f"| {passage['rank']} | `{passage['document_id']}` | {dist_s} |")
        lines.append("")
        lines.append("### Retrieved evidence")
        lines.append("")
        for passage in retrieved:
            lines.extend(
                [
                    f"#### Rank {passage['rank']} — `{passage['document_id']}`",
                    "",
                    "```",
                    passage.get("text", ""),
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "### Generated answer",
                "",
                "```",
                row.get("generated_answer", ""),
                "```",
                "",
                "### Citations",
                "",
            ]
        )
        if citations:
            lines.append("Parsed: " + ", ".join(f"[{c}]" for c in citations))
        else:
            lines.append("Parsed: none")
        lines.append("")
        if citation_map:
            lines.extend(["| Citation | Document ID |", "|----------|-------------|"])
            for item in citation_map:
                doc = item.get("document_id")
                lines.append(f"| [{item['citation']}] | `{doc}` |")
        lines.append("")
        lines.extend(
            [
                "### Runtime",
                f"- Latency: {row.get('generation_latency_ms', 'n/a')} ms",
                f"- Retries: {row.get('retries', 0)}",
            ]
        )
        if row.get("prompt_tokens") is not None:
            lines.append(
                f"- Tokens: prompt {row['prompt_tokens']} | "
                f"completion {row.get('completion_tokens')} | total {row.get('total_tokens')}"
            )
        lines.extend(
            [
                "",
                "### Human review",
                "",
                "- **Answer correctness:** [TO REVIEW]",
                "- **Groundedness:** [TO REVIEW]",
                "- **Citation correctness:** [TO REVIEW]",
                "- **Citation completeness:** [TO REVIEW]",
                "- **Relevance:** [TO REVIEW]",
                "- **Refusal correctness:** [TO REVIEW]",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def run_evaluation() -> dict[str, Any]:
    settings = get_settings(require_groq=True)
    eval_settings = replace(settings, llm_model=settings.model_70b)
    setup_logging(settings.log_level)
    experiment_start = time.perf_counter()

    baseline_path = ROOT / BASELINE_PATH
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline_payload = json.load(handle)

    holdout_qas = load_eval_qas(holdout_split_path(settings))
    question_to_qa = {qa.question: qa for qa in holdout_qas}
    qa_by_id = {qa.question_id: qa for qa in holdout_qas if qa.question_id is not None}

    selected = select_questions(baseline_payload["per_question"], question_to_qa)

    print("Selected 30 questions (baseline gold ranks from normalization_experiment_baseline.json):")
    for entry in selected:
        print(
            f"  id={entry['question_id']}  group={entry['sampling_group']}  "
            f"baseline_gold_rank={entry['baseline_gold_rank']}"
        )

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

    for entry in selected:
        elapsed = time.perf_counter() - experiment_start
        if elapsed >= MAX_RUNTIME_SECONDS:
            stopped_reason = "max_runtime_exceeded"
            break

        qid = entry["question_id"]
        qa = qa_by_id[qid]
        row: dict[str, Any] = {
            "question_id": qid,
            "sampling_group": entry["sampling_group"],
            "baseline_gold_rank": entry["baseline_gold_rank"],
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
                    "retrieved_passages": [],
                    "parsed_citations": [],
                    "citation_document_mapping": [],
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
        row["retrieved_passages"] = serialize_sources(sources)
        row["live_gold_rank"] = gold_rank_live(sources, qa.document_id)

        if not sources:
            row.update(
                {
                    "generated_answer": REFUSAL_PHRASE,
                    "generation_success": True,
                    "generation_skipped": True,
                    "generation_latency_ms": 0.0,
                    "retries": 0,
                    "parsed_citations": [],
                    "citation_document_mapping": [],
                    "refusal": True,
                }
            )
            per_question.append(row)
            continue

        history_text = format_history_for_prompt(prompt_history)
        prompt = build_conversational_prompt(cleaned_query, sources, history_text)
        row["prompt_tokens_estimated"] = token_counter.count(prompt)

        gen_start = time.perf_counter()
        try:
            generation, retries, gen_error = generate_with_retry(
                prompt,
                client,
                model=eval_settings.llm_model,
                temperature=eval_settings.temperature,
                generation_budget=generation_budget,
                rate_limit_counter=rate_limit_counter,
            )
        except RuntimeError:
            stopped_reason = "repeated_rate_limit"
            break

        gen_latency_ms = (time.perf_counter() - gen_start) * 1000

        if generation is None:
            row.update(
                {
                    "generated_answer": "",
                    "generation_success": False,
                    "generation_error": gen_error,
                    "generation_latency_ms": round(gen_latency_ms, 2),
                    "retries": retries,
                    "rate_limit_errors_observed": rate_limit_counter["count"],
                    "parsed_citations": [],
                    "citation_document_mapping": [],
                }
            )
            per_question.append(row)
            continue

        answer = generation.text
        citations = parse_citations(answer)
        row.update(
            {
                "generated_answer": answer,
                "generation_success": True,
                "generation_model": generation.model,
                "generation_latency_ms": round(gen_latency_ms, 2),
                "retries": retries,
                "rate_limit_errors_observed": rate_limit_counter["count"],
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.total_tokens,
                "refusal": REFUSAL_PHRASE in answer,
                "parsed_citations": citations,
                "citation_document_mapping": citation_mapping(
                    citations, row["retrieved_document_ids"]
                ),
            }
        )
        per_question.append(row)

    total_runtime = time.perf_counter() - experiment_start
    successful = [r for r in per_question if r.get("generation_success")]
    failed = [r for r in per_question if not r.get("generation_success")]
    latencies = [
        r["generation_latency_ms"]
        for r in per_question
        if r.get("generation_latency_ms") is not None
    ]
    prompt_tokens = [r["prompt_tokens"] for r in successful if r.get("prompt_tokens") is not None]
    completion_tokens = [
        r["completion_tokens"] for r in successful if r.get("completion_tokens") is not None
    ]
    total_tokens_list = [r["total_tokens"] for r in successful if r.get("total_tokens") is not None]

    group_counts = {
        "A_gold_rank_1": sum(1 for r in per_question if r.get("sampling_group") == "A_gold_rank_1"),
        "B_gold_rank_2_3": sum(
            1 for r in per_question if r.get("sampling_group") == "B_gold_rank_2_3"
        ),
        "C_gold_outside_top_3": sum(
            1 for r in per_question if r.get("sampling_group") == "C_gold_outside_top_3"
        ),
    }

    summary = {
        "questions_target": TOTAL_QUESTIONS,
        "questions_processed": len(per_question),
        "successful_generations": len(successful),
        "failed_generations": len(failed),
        "rate_limit_errors": rate_limit_counter["count"],
        "generation_api_calls_used": generation_budget["used"],
        "average_generation_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "median_generation_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "total_runtime_seconds": round(total_runtime, 2),
        "average_prompt_tokens": round(statistics.mean(prompt_tokens), 2) if prompt_tokens else None,
        "average_completion_tokens": round(statistics.mean(completion_tokens), 2)
        if completion_tokens
        else None,
        "total_tokens_sum": sum(total_tokens_list) if total_tokens_list else 0,
        "sampling_group_counts": group_counts,
    }

    payload: dict[str, Any] = {
        "experiment": "phase5b_end_to_end_30",
        "description": "Bounded 30-question end-to-end conversational RAG evaluation",
        "eval_only": True,
        "production_code_modified": False,
        "stopped_reason": stopped_reason,
        "artifact_selection_source": BASELINE_PATH,
        "constraints": {
            "total_questions": TOTAL_QUESTIONS,
            "max_generation_api_calls": MAX_GENERATION_CALLS,
            "max_retries_per_call": MAX_RETRIES_PER_CALL,
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "generation_model": eval_settings.llm_model,
            "rewrite_model": settings.rewrite_model,
            "top_k": settings.top_k,
            "no_llm_judge": True,
        },
        "selection": {
            "group_a_gold_rank_1": GROUP_SIZE,
            "group_b_gold_rank_2_3": GROUP_SIZE,
            "group_c_outside_top_3": GROUP_SIZE,
            "selected": [
                {
                    "question_id": e["question_id"],
                    "sampling_group": e["sampling_group"],
                    "baseline_gold_rank": e["baseline_gold_rank"],
                }
                for e in selected[: len(per_question)]
            ],
        },
        "summary": summary,
        "per_question": per_question,
    }

    save_json(OUTPUT_JSON, payload)
    Path(OUTPUT_MD).write_text(build_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    try:
        payload = run_evaluation()
    except RuntimeError as exc:
        print(f"EXPERIMENT STOPPED: {exc}")
        raise SystemExit(1) from exc

    s = payload["summary"]
    print("\n" + "=" * 60)
    print("Phase 5B end-to-end evaluation — DONE")
    print("=" * 60)
    print(f"Stopped reason: {payload['stopped_reason']}")
    print(f"Processed: {s['questions_processed']}/{s['questions_target']}")
    print(f"Successful generations: {s['successful_generations']}")
    print(f"Failed generations: {s['failed_generations']}")
    print(f"Rate-limit errors: {s['rate_limit_errors']}")
    print(f"Generation API calls: {s['generation_api_calls_used']}/{MAX_GENERATION_CALLS}")
    print(f"Avg latency: {s['average_generation_latency_ms']:.1f} ms")
    print(f"Median latency: {s['median_generation_latency_ms']:.1f} ms")
    print(f"Total runtime: {s['total_runtime_seconds']:.1f} s")
    if s["average_prompt_tokens"] is not None:
        print(f"Avg prompt tokens: {s['average_prompt_tokens']:.1f}")
    if s["average_completion_tokens"] is not None:
        print(f"Avg completion tokens: {s['average_completion_tokens']:.1f}")
    print(f"Total tokens (sum): {s['total_tokens_sum']}")
    print("Sampling groups processed:", s["sampling_group_counts"])
    print(f"\nJSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
