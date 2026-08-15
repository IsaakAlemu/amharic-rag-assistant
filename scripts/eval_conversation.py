"""Evaluate conversational RAG against single-turn baseline."""

from __future__ import annotations

import argparse
import json
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
    evaluation_is_viable,
)
from src.eval_utils import normalize_text, save_json
from src.history_manager import ConversationState
from src.llm import REFUSAL_PHRASE
from src.logging_config import setup_logging
from src.pipeline import answer_conversation, answer_question, load_vector_collection
from src.rewrite_eval import evaluate_rewrite

PHASE4_BASELINE = {
    "source": "results/phase4_summary.json (pre-rewrite-tuning, rate-limit contaminated)",
    "baseline": {
        "retrieval_hit_at_1": 0.5,
        "follow_up_retrieval_hit_at_1": 0.375,
        "ground_truth_match_rate": 0.27,
        "refusal_correct_rate": 0.9,
        "mean_latency_ms": 5188,
        "errors": 15,
    },
    "conversational": {
        "retrieval_hit_at_1": 0.556,
        "follow_up_retrieval_hit_at_1": 0.5,
        "ground_truth_match_rate": 0.2,
        "refusal_correct_rate": 0.85,
        "rewrite_match_rate": 0.25,
        "mean_prompt_tokens_estimated": 1125,
        "mean_latency_ms": 4570,
        "errors": 16,
    },
}


def load_scenarios(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def retrieval_hit_at_k(sources: list[dict], gold_document_id: str, k: int) -> bool | None:
    if not gold_document_id:
        return None
    if not sources:
        return False
    for source in sources[:k]:
        if source["document_id"] == gold_document_id:
            return True
    return False


def retrieval_hit_at_1(sources: list[dict], gold_document_id: str) -> bool | None:
    return retrieval_hit_at_k(sources, gold_document_id, 1)


def retrieval_mrr(sources: list[dict], gold_document_id: str) -> float | None:
    if not gold_document_id:
        return None
    if not sources:
        return 0.0
    for rank, source in enumerate(sources, start=1):
        if source["document_id"] == gold_document_id:
            return 1.0 / rank
    return 0.0


def ground_truth_match(answer: str, ground_truth: str) -> bool | None:
    if not ground_truth:
        return None
    return normalize_text(ground_truth) in normalize_text(answer)


def rewrite_match(rewritten: str, required_terms: list[str]) -> bool | None:
    if not required_terms:
        return None
    normalized = normalize_text(rewritten)
    return all(normalize_text(term) in normalized for term in required_terms)


def refusal_correct(answer: str, expects_refusal: bool) -> bool:
    refused = REFUSAL_PHRASE in answer
    return refused if expects_refusal else not refused


def summarize(values: list[float]) -> float:
    return round(statistics.mean(values), 1) if values else 0.0


def annotate_turn_result(
    row: dict,
    *,
    turn: dict,
    scenario: dict,
    turn_index: int,
    history_turns: list[dict],
) -> dict:
    row["error_type"] = classify_error(row.get("error"))
    row["excluded_from_answer_scoring"] = row["error_type"] == "rate_limit"
    row["excluded_from_scoring"] = row["excluded_from_answer_scoring"]
    row["retrieval_hit_at_3"] = retrieval_hit_at_k(
        row.get("sources", []),
        turn.get("gold_document_id", ""),
        3,
    )
    row["retrieval_mrr"] = retrieval_mrr(
        row.get("sources", []),
        turn.get("gold_document_id", ""),
    )

    if row.get("mode") == "conversational" and turn_index > 0:
        rewrite_eval = evaluate_rewrite(
            scenario_id=scenario["id"],
            category=scenario["category"],
            turn_index=turn_index,
            history=history_turns,
            turn=turn,
            original=turn["user"],
            rewrite=row.get("rewritten_query") or turn["user"],
            retried=False,
            error=row.get("error") if row["error_type"] == "api_error" else None,
        )
        row["rewrite_success"] = rewrite_eval.success
        row["rewrite_eval_failures"] = rewrite_eval.failure_categories
    else:
        row["rewrite_success"] = None
        row["rewrite_eval_failures"] = []

    return row


def run_turn_baseline(
    turn: dict,
    *,
    client,
    embed_model,
    collection,
    settings,
):
    t0 = time.perf_counter()
    result = answer_question(
        turn["user"],
        client=client,
        embed_model=embed_model,
        collection=collection,
        settings=settings,
    )
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "mode": "single_turn_baseline",
        "query_used_for_retrieval": turn["user"],
        "rewritten_query": None,
        "answer": result.answer,
        "sources": result.sources,
        "error": result.error,
        "refusal": result.refusal or REFUSAL_PHRASE in result.answer,
        "retrieval_hit_at_1": retrieval_hit_at_1(result.sources, turn.get("gold_document_id", "")),
        "ground_truth_match": ground_truth_match(result.answer, turn.get("ground_truth", "")),
        "refusal_correct": refusal_correct(result.answer, turn.get("expects_refusal", False)),
        "rewrite_match": None,
        "timings_ms": result.timings_ms | {"total": total_ms},
        "prompt_tokens_estimated": None,
    }


def run_turn_conversational(
    turn: dict,
    conversation: ConversationState,
    *,
    client,
    embed_model,
    collection,
    settings,
):
    t0 = time.perf_counter()
    result = answer_conversation(
        turn["user"],
        conversation,
        client=client,
        embed_model=embed_model,
        collection=collection,
        settings=settings,
    )
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "mode": "conversational",
        "query_used_for_retrieval": result.retrieval_query,
        "rewritten_query": result.rewritten_query,
        "answer": result.answer,
        "sources": result.sources,
        "error": result.error,
        "refusal": result.refusal or REFUSAL_PHRASE in result.answer,
        "retrieval_hit_at_1": retrieval_hit_at_1(result.sources, turn.get("gold_document_id", "")),
        "ground_truth_match": ground_truth_match(result.answer, turn.get("ground_truth", "")),
        "refusal_correct": refusal_correct(result.answer, turn.get("expects_refusal", False)),
        "rewrite_match": rewrite_match(
            result.rewritten_query,
            turn.get("rewrite_should_contain", []),
        ),
        "timings_ms": result.timings_ms | {"total": total_ms},
        "prompt_tokens_estimated": result.prompt_tokens_estimated,
        "history_turns_used": result.history_turns_used,
    }


def run_with_rate_limit_retry(
    run_fn,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    **kwargs,
) -> dict:
    row, retries, error_type = call_with_bounded_retry(
        lambda: run_fn(*args, **kwargs),
        max_retries=max_retries,
        max_backoff_seconds=max_backoff_seconds,
        get_error=lambda result: result.get("error"),
        label="Generation",
    )
    row["api_retries"] = retries
    if error_type and not row.get("error"):
        row["error"] = "The language model is temporarily rate-limited. Please retry shortly."
    return row


def aggregate_results(rows: list[dict]) -> dict:
    answer_scored_rows = [
        row for row in rows if not row.get("excluded_from_answer_scoring")
    ]
    retrieval_rows = [
        row for row in rows if row.get("retrieval_hit_at_1") is not None
    ]

    def rate(field: str, subset: list[dict] | None = None) -> float | None:
        source = subset if subset is not None else answer_scored_rows
        values = [row[field] for row in source if row.get(field) is not None]
        if not values:
            return None
        return sum(1 for value in values if value) / len(values)

    def mean(field: str, subset: list[dict] | None = None) -> float:
        source = subset if subset is not None else answer_scored_rows
        values = [row[field] for row in source if row.get(field) is not None]
        if not values:
            return 0.0
        return round(statistics.mean(values), 4)

    follow_up_answer_rows = [
        row
        for row in answer_scored_rows
        if row.get("turn_index", 0) > 0
    ]
    follow_up_retrieval_rows = [
        row
        for row in retrieval_rows
        if row.get("turn_index", 0) > 0
    ]
    follow_up_all = [row for row in rows if row.get("turn_index", 0) > 0]

    return {
        "turns_evaluated": len(rows),
        "turns_scored_for_answers": len(answer_scored_rows),
        "turns_scored_for_retrieval": len(retrieval_rows),
        "turns_excluded_rate_limit": sum(
            1 for row in rows if row.get("error_type") == "rate_limit"
        ),
        "follow_up_turns": len(follow_up_all),
        "follow_up_turns_scored_for_answers": len(follow_up_answer_rows),
        "follow_up_turns_scored_for_retrieval": len(follow_up_retrieval_rows),
        "retrieval_hit_at_1": rate("retrieval_hit_at_1", retrieval_rows),
        "retrieval_hit_at_3": rate("retrieval_hit_at_3", retrieval_rows),
        "retrieval_mrr": mean("retrieval_mrr", retrieval_rows),
        "follow_up_retrieval_hit_at_1": rate(
            "retrieval_hit_at_1", follow_up_retrieval_rows
        ),
        "follow_up_retrieval_hit_at_3": rate(
            "retrieval_hit_at_3", follow_up_retrieval_rows
        ),
        "follow_up_retrieval_mrr": mean("retrieval_mrr", follow_up_retrieval_rows),
        "ground_truth_match_rate": rate("ground_truth_match"),
        "refusal_correct_rate": rate("refusal_correct"),
        "rewrite_match_rate": rate("rewrite_match", follow_up_all),
        "rewrite_success_rate": rate("rewrite_success", follow_up_all),
        "mean_total_latency_ms": summarize(
            [
                row["timings_ms"].get("total", 0)
                for row in answer_scored_rows
                if row.get("timings_ms")
            ]
        ),
        "mean_rewrite_latency_ms": summarize(
            [
                row["timings_ms"].get("rewrite", 0)
                for row in answer_scored_rows
                if row.get("timings_ms") and row["timings_ms"].get("rewrite") is not None
            ]
        ),
        "mean_prompt_tokens_estimated": summarize(
            [
                float(row["prompt_tokens_estimated"])
                for row in answer_scored_rows
                if row.get("prompt_tokens_estimated")
            ]
        ),
        "errors_total": sum(1 for row in rows if row.get("error")),
        "errors_rate_limit": sum(
            1 for row in rows if row.get("error_type") == "rate_limit"
        ),
        "errors_api_other": sum(1 for row in rows if row.get("error_type") == "api_error"),
    }


def _restore_conversation_on_error(
    conversation: ConversationState,
    turn: dict,
    conv_row: dict,
) -> None:
    """Keep eval history intact when generation fails and pipeline pops the user turn."""
    if not conv_row.get("error"):
        return
    if conversation.messages and conversation.messages[-1].role == "user":
        conversation.add_assistant("(unavailable due to API error)")
        return
    conversation.add_user(turn["user"])
    conversation.add_assistant("(unavailable due to API error)")


def run_conversation_eval(
    *,
    settings,
    scenarios_path: str = "data/eval/conversation_scenarios.json",
    output: str = "results/conversation_eval.json",
    sleep_seconds: float = 8.0,
    inter_call_sleep_seconds: float = 6.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    min_answer_scored_ratio: float = DEFAULT_MIN_ANSWER_SCORED_RATIO,
    min_turns_before_viability_check: int = DEFAULT_MIN_TURNS_BEFORE_VIABILITY_CHECK,
) -> dict:
    groq_client = Groq(api_key=settings.groq_api_key)
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    scenarios = load_scenarios(Path(scenarios_path))

    scenario_results = []
    baseline_rows: list[dict] = []
    conversational_rows: list[dict] = []
    status = "complete"
    termination_reason: str | None = None
    turns_attempted = 0

    for scenario in scenarios:
        conv_state = ConversationState()
        history_turns: list[dict] = []

        scenario_row = {
            "id": scenario["id"],
            "category": scenario["category"],
            "turns": [],
        }

        for turn_index, turn in enumerate(scenario["turns"]):
            turns_attempted += 1
            baseline_row = run_with_rate_limit_retry(
                run_turn_baseline,
                turn,
                client=groq_client,
                embed_model=embed_model,
                collection=collection,
                settings=settings,
                max_retries=max_retries,
                max_backoff_seconds=max_backoff_seconds,
            )
            baseline_row["turn_index"] = turn_index
            baseline_row["user"] = turn["user"]
            baseline_row = annotate_turn_result(
                baseline_row,
                turn=turn,
                scenario=scenario,
                turn_index=turn_index,
                history_turns=history_turns,
            )
            baseline_rows.append(baseline_row)

            time.sleep(inter_call_sleep_seconds)

            conv_row = run_with_rate_limit_retry(
                run_turn_conversational,
                turn,
                conv_state,
                client=groq_client,
                embed_model=embed_model,
                collection=collection,
                settings=settings,
                max_retries=max_retries,
                max_backoff_seconds=max_backoff_seconds,
            )
            conv_row["turn_index"] = turn_index
            conv_row["user"] = turn["user"]
            conv_row = annotate_turn_result(
                conv_row,
                turn=turn,
                scenario=scenario,
                turn_index=turn_index,
                history_turns=history_turns,
            )
            _restore_conversation_on_error(conv_state, turn, conv_row)
            conversational_rows.append(conv_row)

            scenario_row["turns"].append(
                {
                    "turn_index": turn_index,
                    "user": turn["user"],
                    "baseline": baseline_row,
                    "conversational": conv_row,
                }
            )

            history_turns.append(turn)
            time.sleep(sleep_seconds)

            b_hit = baseline_row.get("retrieval_hit_at_1")
            c_hit = conv_row.get("retrieval_hit_at_1")
            b_err = baseline_row.get("error_type") or "ok"
            c_err = conv_row.get("error_type") or "ok"
            print(
                f"[{scenario['category']}] {scenario['id']} turn {turn_index} — "
                f"baseline hit@1={b_hit} ({b_err}) | "
                f"conv hit@1={c_hit} ({c_err})"
            )

            answer_scored = sum(
                1
                for row in baseline_rows + conversational_rows
                if not row.get("excluded_from_answer_scoring")
            )
            total_api_slots = len(baseline_rows) + len(conversational_rows)
            if not evaluation_is_viable(
                turns_attempted=turns_attempted,
                answer_scored=answer_scored,
                total_api_slots=total_api_slots,
                min_answer_scored_ratio=min_answer_scored_ratio,
                min_turns_before_check=min_turns_before_viability_check,
            ):
                status = "inconclusive"
                termination_reason = (
                    "Groq API rate limits prevented a trustworthy full evaluation "
                    f"(<{min_answer_scored_ratio:.0%} of API calls produced answers "
                    f"after {turns_attempted} turns)."
                )
                print(f"\n*** TERMINATING EARLY: {termination_reason}")
                break

        scenario_results.append(scenario_row)
        if status == "inconclusive":
            break

    payload = {
        "description": (
            "Compares single-turn baseline (each question alone) against conversational "
            "RAG (history-aware rewrite + separated prompt sections). "
            "Rate-limit failures are excluded from answer metrics. "
            "Run terminates early when API quota is exhausted."
        ),
        "status": status,
        "termination_reason": termination_reason,
        "settings": {
            "llm_model": settings.llm_model,
            "rewrite_model": settings.rewrite_model,
            "top_k": settings.top_k,
            "max_history_turns": settings.max_history_turns,
            "max_history_tokens": settings.max_history_tokens,
            "sleep_seconds": sleep_seconds,
            "inter_call_sleep_seconds": inter_call_sleep_seconds,
            "max_retries": max_retries,
            "max_backoff_seconds": max_backoff_seconds,
            "min_answer_scored_ratio": min_answer_scored_ratio,
        },
        "phase4_baseline_comparison": PHASE4_BASELINE,
        "summary": {
            "baseline": aggregate_results(baseline_rows),
            "conversational": aggregate_results(conversational_rows),
            "follow_up_comparison": {
                "baseline": aggregate_results(
                    [row for row in baseline_rows if row["turn_index"] > 0]
                ),
                "conversational": aggregate_results(
                    [row for row in conversational_rows if row["turn_index"] > 0]
                ),
            },
        },
        "scenarios": scenario_results,
    }
    save_json(output, payload)
    return payload


def print_summary(payload: dict, output_path: str) -> None:
    print("=" * 60)
    status = payload.get("status", "complete")
    print(f"Conversation evaluation — status: {status.upper()}")
    if payload.get("termination_reason"):
        print(f"Reason: {payload['termination_reason']}")
    print("=" * 60)
    for mode in ("baseline", "conversational"):
        summary = payload["summary"][mode]
        print(f"\n{mode.upper()}")
        print(
            f"  Answer-scored turns:    {summary['turns_scored_for_answers']}/{summary['turns_evaluated']}"
        )
        print(
            f"  Retrieval-scored turns: {summary['turns_scored_for_retrieval']}/{summary['turns_evaluated']}"
        )
        print(f"  Rate-limit exclusions:  {summary['turns_excluded_rate_limit']}")
        print(f"  Retrieval Hit@1:        {summary['retrieval_hit_at_1']:.1%}" if summary['retrieval_hit_at_1'] is not None else "  Retrieval Hit@1:        n/a")
        print(f"  Retrieval Hit@3:        {summary['retrieval_hit_at_3']:.1%}" if summary['retrieval_hit_at_3'] is not None else "  Retrieval Hit@3:        n/a")
        print(f"  MRR:                    {summary['retrieval_mrr']:.3f}" if summary['retrieval_mrr'] else "  MRR:                    n/a")
        gt = summary["ground_truth_match_rate"]
        print(f"  Ground-truth match:     {gt:.1%}" if gt is not None else "  Ground-truth match:     n/a")
        print(f"  Refusal correct:        {summary['refusal_correct_rate']:.1%}" if summary['refusal_correct_rate'] is not None else "  Refusal correct:        n/a")
        rw = summary.get("rewrite_match_rate")
        if rw is not None:
            print(f"  Rewrite match (legacy): {rw:.1%}")
        rs = summary.get("rewrite_success_rate")
        if rs is not None:
            print(f"  Rewrite success:        {rs:.1%}")
        print(f"  Mean latency ms:        {summary['mean_total_latency_ms']}")
        if summary.get("mean_prompt_tokens_estimated"):
            print(f"  Mean prompt tokens:     {summary['mean_prompt_tokens_estimated']}")
        print(f"  API errors (rate limit): {summary['errors_rate_limit']}")
        print(f"  API errors (other):      {summary['errors_api_other']}")

    print("\nFOLLOW-UP ONLY (turn_index > 0)")
    for mode in ("baseline", "conversational"):
        summary = payload["summary"]["follow_up_comparison"][mode]
        hit1 = summary["follow_up_retrieval_hit_at_1"]
        hit3 = summary["follow_up_retrieval_hit_at_3"]
        print(
            f"  {mode}: Hit@1={hit1:.1%} Hit@3={hit3:.1%} MRR={summary['follow_up_retrieval_mrr']:.3f}"
            if hit1 is not None
            else f"  {mode}: insufficient scored follow-ups"
        )
    print(f"\nSaved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conversational RAG.")
    parser.add_argument(
        "--input",
        default="data/eval/conversation_scenarios.json",
        help="Scenario JSON path.",
    )
    parser.add_argument("--output", default="results/conversation_eval_v3.json")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=8.0,
        help="Pause between turns to reduce Groq rate limits.",
    )
    parser.add_argument(
        "--inter-call-sleep-seconds",
        type=float,
        default=6.0,
        help="Pause between baseline and conversational calls within a turn.",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--min-answer-scored-ratio",
        type=float,
        default=DEFAULT_MIN_ANSWER_SCORED_RATIO,
        help="Terminate early if scored answer ratio falls below this threshold.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    payload = run_conversation_eval(
        settings=settings,
        scenarios_path=args.input,
        output=args.output,
        sleep_seconds=args.sleep_seconds,
        inter_call_sleep_seconds=args.inter_call_sleep_seconds,
        max_retries=args.max_retries,
        max_backoff_seconds=args.max_backoff_seconds,
        min_answer_scored_ratio=args.min_answer_scored_ratio,
    )
    print_summary(payload, args.output)


if __name__ == "__main__":
    main()
