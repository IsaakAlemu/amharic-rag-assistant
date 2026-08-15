"""Lightweight rewrite + retrieval integration eval — no 70B generation."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.eval_api import (
    DEFAULT_MAX_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_ANSWER_SCORED_RATIO,
    DEFAULT_MIN_TURNS_BEFORE_VIABILITY_CHECK,
    call_with_bounded_retry,
    classify_error,
)
from src.eval_utils import save_json
from src.history_manager import ChatMessage
from src.logging_config import setup_logging
from src.pipeline import load_vector_collection
from src.query_rewriter import rewrite_query
from src.retriever import retrieve
from src.rewrite_eval import evaluate_rewrite, summarize_rewrite_eval

PHASE4_FOLLOW_UP_HIT_BASELINE = 0.375
PHASE4_FOLLOW_UP_HIT_CONVERSATIONAL = 0.5


def load_scenarios(path: Path) -> list[dict]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_history_messages(history_turns: list[dict]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for turn in history_turns:
        messages.append(ChatMessage(role="user", content=turn["user"]))
        messages.append(ChatMessage(role="assistant", content="..."))
    return messages


def retrieval_hit_at_k(sources: list[dict], gold_document_id: str, k: int) -> bool | None:
    if not gold_document_id:
        return None
    if not sources:
        return False
    return any(source["document_id"] == gold_document_id for source in sources[:k])


def retrieval_mrr(sources: list[dict], gold_document_id: str) -> float | None:
    if not gold_document_id:
        return None
    if not sources:
        return 0.0
    for rank, source in enumerate(sources, start=1):
        if source["document_id"] == gold_document_id:
            return 1.0 / rank
    return 0.0


def aggregate_retrieval_rows(rows: list[dict]) -> dict:
    scored = [row for row in rows if row.get("retrieval_hit_at_1") is not None]

    def rate(field: str) -> float | None:
        values = [row[field] for row in scored if row.get(field) is not None]
        if not values:
            return None
        return sum(1 for value in values if value) / len(values)

    def mean(field: str) -> float | None:
        values = [row[field] for row in scored if row.get(field) is not None]
        if not values:
            return None
        return round(statistics.mean(values), 4)

    follow_up = [row for row in scored if row.get("turn_index", 0) > 0]

    return {
        "turns_evaluated": len(rows),
        "retrieval_scored_turns": len(scored),
        "follow_up_turns": len(follow_up),
        "retrieval_hit_at_1": rate("retrieval_hit_at_1"),
        "retrieval_hit_at_3": rate("retrieval_hit_at_3"),
        "retrieval_mrr": mean("retrieval_mrr"),
        "follow_up_retrieval_hit_at_1": rate("retrieval_hit_at_1")
        if follow_up
        else None,
        "follow_up_retrieval_hit_at_3": rate("retrieval_hit_at_3")
        if follow_up
        else None,
        "follow_up_retrieval_mrr": mean("retrieval_mrr") if follow_up else None,
        "rewrite_success_rate": rate("rewrite_success"),
        "rewrite_api_errors": sum(1 for row in rows if row.get("rewrite_error_type")),
        "rewrite_rate_limit_errors": sum(
            1 for row in rows if row.get("rewrite_error_type") == "rate_limit"
        ),
        "mean_rewrite_latency_ms": round(
            statistics.mean(
                row["rewrite_latency_ms"]
                for row in rows
                if row.get("rewrite_latency_ms") is not None
            ),
            1,
        )
        if any(row.get("rewrite_latency_ms") is not None for row in rows)
        else None,
    }


def run_retrieval_integration_eval(
    *,
    settings,
    scenarios_path: str = "data/eval/conversation_scenarios.json",
    output: str = "results/conversation_retrieval_eval.json",
    sleep_seconds: float = 2.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    min_answer_scored_ratio: float = 0.7,
) -> dict:
    client = Groq(api_key=settings.groq_api_key)
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    scenarios = load_scenarios(Path(scenarios_path))

    baseline_rows: list[dict] = []
    conversational_rows: list[dict] = []
    rewrite_eval_results = []
    scenario_results: list[dict] = []

    status = "complete"
    termination_reason: str | None = None
    turns_attempted = 0
    rewrite_attempts = 0
    rewrite_rate_limits = 0

    for scenario in scenarios:
        history_turns: list[dict] = []
        scenario_row = {"id": scenario["id"], "category": scenario["category"], "turns": []}

        for turn_index, turn in enumerate(scenario["turns"]):
            turns_attempted += 1
            gold_id = turn.get("gold_document_id", "")

            baseline_sources = retrieve(
                turn["user"],
                collection,
                embed_model,
                top_k=settings.top_k,
            )
            baseline_row = {
                "scenario_id": scenario["id"],
                "category": scenario["category"],
                "turn_index": turn_index,
                "mode": "baseline_retrieval",
                "query": turn["user"],
                "sources": baseline_sources,
                "retrieval_hit_at_1": retrieval_hit_at_k(baseline_sources, gold_id, 1),
                "retrieval_hit_at_3": retrieval_hit_at_k(baseline_sources, gold_id, 3),
                "retrieval_mrr": retrieval_mrr(baseline_sources, gold_id),
                "rewrite_success": None,
                "rewrite_error_type": None,
                "rewrite_latency_ms": None,
            }
            baseline_rows.append(baseline_row)

            conv_row: dict
            if turn_index == 0:
                conv_row = {
                    **baseline_row,
                    "mode": "conversational_retrieval",
                    "rewritten_query": turn["user"],
                }
            else:
                history_messages = build_history_messages(history_turns)
                rewrite_attempts += 1

                def do_rewrite():
                    t0 = time.perf_counter()
                    try:
                        result = rewrite_query(
                            turn["user"],
                            history_messages,
                            client=client,
                            model=settings.rewrite_model,
                            temperature=0.0,
                        )
                        return {
                            "rewritten_query": result.rewritten_query,
                            "error": None,
                            "latency_ms": (time.perf_counter() - t0) * 1000,
                            "retried": result.retried,
                        }
                    except Exception as exc:
                        return {
                            "rewritten_query": turn["user"],
                            "error": str(exc),
                            "latency_ms": (time.perf_counter() - t0) * 1000,
                            "retried": False,
                        }

                rewrite_payload, retries, error_type = call_with_bounded_retry(
                    do_rewrite,
                    max_retries=max_retries,
                    max_backoff_seconds=max_backoff_seconds,
                    get_error=lambda payload: payload.get("error"),
                    label="Rewrite",
                )
                rewrite_error_type = error_type or classify_error(rewrite_payload.get("error"))

                rewrite_eval = evaluate_rewrite(
                    scenario_id=scenario["id"],
                    category=scenario["category"],
                    turn_index=turn_index,
                    history=history_turns,
                    turn=turn,
                    original=turn["user"],
                    rewrite=rewrite_payload["rewritten_query"],
                    retried=rewrite_payload.get("retried", False),
                    error=rewrite_payload.get("error"),
                )
                rewrite_eval_results.append(rewrite_eval)
                if rewrite_error_type == "rate_limit":
                    rewrite_rate_limits += 1

                conv_sources = retrieve(
                    rewrite_payload["rewritten_query"],
                    collection,
                    embed_model,
                    top_k=settings.top_k,
                )
                conv_row = {
                    "scenario_id": scenario["id"],
                    "category": scenario["category"],
                    "turn_index": turn_index,
                    "mode": "conversational_retrieval",
                    "query": turn["user"],
                    "rewritten_query": rewrite_payload["rewritten_query"],
                    "sources": conv_sources,
                    "retrieval_hit_at_1": retrieval_hit_at_k(conv_sources, gold_id, 1),
                    "retrieval_hit_at_3": retrieval_hit_at_k(conv_sources, gold_id, 3),
                    "retrieval_mrr": retrieval_mrr(conv_sources, gold_id),
                    "rewrite_success": rewrite_eval.success,
                    "rewrite_error_type": rewrite_error_type,
                    "rewrite_latency_ms": rewrite_payload.get("latency_ms"),
                    "rewrite_retries": retries,
                }

            conversational_rows.append(conv_row)
            scenario_row["turns"].append(
                {
                    "turn_index": turn_index,
                    "baseline": baseline_row,
                    "conversational": conv_row,
                }
            )
            history_turns.append(turn)

            print(
                f"[{scenario['id']}] turn {turn_index} — "
                f"baseline hit@1={baseline_row['retrieval_hit_at_1']} | "
                f"conv hit@1={conv_row['retrieval_hit_at_1']} | "
                f"rewrite_ok={conv_row.get('rewrite_success')}"
            )

            if (
                rewrite_attempts >= DEFAULT_MIN_TURNS_BEFORE_VIABILITY_CHECK
                and rewrite_attempts > 0
                and (rewrite_attempts - rewrite_rate_limits) / rewrite_attempts
                < min_answer_scored_ratio
            ):
                status = "inconclusive"
                termination_reason = (
                    "Rewrite API rate limits exceeded viability threshold "
                    f"(<{min_answer_scored_ratio:.0%} successful rewrite calls)."
                )
                print(f"\n*** TERMINATING EARLY: {termination_reason}")
                break

            time.sleep(sleep_seconds)

        scenario_results.append(scenario_row)
        if status == "inconclusive":
            break

    baseline_summary = aggregate_retrieval_rows(baseline_rows)
    conv_summary = aggregate_retrieval_rows(conversational_rows)
    follow_up_baseline = aggregate_retrieval_rows(
        [row for row in baseline_rows if row["turn_index"] > 0]
    )
    follow_up_conv = aggregate_retrieval_rows(
        [row for row in conversational_rows if row["turn_index"] > 0]
    )

    payload = {
        "description": (
            "Lightweight integration eval: rewrite (8B) + retrieval only. "
            "No 70B generation calls."
        ),
        "status": status,
        "termination_reason": termination_reason,
        "settings": {
            "rewrite_model": settings.rewrite_model,
            "top_k": settings.top_k,
            "sleep_seconds": sleep_seconds,
            "max_retries": max_retries,
            "max_backoff_seconds": max_backoff_seconds,
        },
        "phase4_baseline_comparison": {
            "follow_up_hit_at_1_baseline": PHASE4_FOLLOW_UP_HIT_BASELINE,
            "follow_up_hit_at_1_conversational": PHASE4_FOLLOW_UP_HIT_CONVERSATIONAL,
        },
        "summary": {
            "baseline": baseline_summary,
            "conversational": conv_summary,
            "follow_up_comparison": {
                "baseline": follow_up_baseline,
                "conversational": follow_up_conv,
            },
            "rewrite_eval": summarize_rewrite_eval(rewrite_eval_results),
        },
        "scenarios": scenario_results,
    }
    save_json(output, payload)
    return payload


def print_summary(payload: dict, output_path: str) -> None:
    status = payload.get("status", "complete")
    print("=" * 60)
    print(f"Retrieval integration eval — status: {status.upper()}")
    if payload.get("termination_reason"):
        print(f"Reason: {payload['termination_reason']}")
    print("=" * 60)

    for mode in ("baseline", "conversational"):
        summary = payload["summary"][mode]
        print(f"\n{mode.upper()}")
        print(f"  Retrieval Hit@1: {summary['retrieval_hit_at_1']:.1%}" if summary.get("retrieval_hit_at_1") is not None else "  Retrieval Hit@1: n/a")
        print(f"  Retrieval Hit@3: {summary['retrieval_hit_at_3']:.1%}" if summary.get("retrieval_hit_at_3") is not None else "  Retrieval Hit@3: n/a")
        print(f"  MRR:             {summary['retrieval_mrr']:.3f}" if summary.get("retrieval_mrr") is not None else "  MRR:             n/a")

    fu = payload["summary"]["follow_up_comparison"]
    b = fu["baseline"]
    c = fu["conversational"]
    print("\nFOLLOW-UP RETRIEVAL (primary integration signal)")
    if b.get("follow_up_retrieval_hit_at_1") is not None:
        print(
            f"  Baseline Hit@1:        {b['follow_up_retrieval_hit_at_1']:.1%} "
            f"(Phase 4: {PHASE4_FOLLOW_UP_HIT_BASELINE:.1%})"
        )
        print(
            f"  Conversational Hit@1:  {c['follow_up_retrieval_hit_at_1']:.1%} "
            f"(Phase 4: {PHASE4_FOLLOW_UP_HIT_CONVERSATIONAL:.1%})"
        )
        delta = c["follow_up_retrieval_hit_at_1"] - b["follow_up_retrieval_hit_at_1"]
        print(f"  Delta (conv - baseline): {delta:+.1%}")
    else:
        print("  Insufficient follow-up data")

    rewrite = payload["summary"].get("rewrite_eval", {})
    if rewrite.get("rewrite_success_rate") is not None:
        print(f"\nRewrite success: {rewrite['rewrite_success_rate']:.1%}")
    print(f"\nRewrite rate-limit errors: {c.get('rewrite_rate_limit_errors', 0)}")
    print(f"Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight rewrite + retrieval integration eval (no 70B)."
    )
    parser.add_argument("--input", default="data/eval/conversation_scenarios.json")
    parser.add_argument("--output", default="results/conversation_retrieval_eval.json")
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-backoff-seconds", type=float, default=DEFAULT_MAX_BACKOFF_SECONDS)
    parser.add_argument(
        "--min-rewrite-success-ratio",
        type=float,
        default=DEFAULT_MIN_ANSWER_SCORED_RATIO,
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    payload = run_retrieval_integration_eval(
        settings=settings,
        scenarios_path=args.input,
        output=args.output,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        max_backoff_seconds=args.max_backoff_seconds,
        min_answer_scored_ratio=args.min_rewrite_success_ratio,
    )
    print_summary(payload, args.output)


if __name__ == "__main__":
    main()
