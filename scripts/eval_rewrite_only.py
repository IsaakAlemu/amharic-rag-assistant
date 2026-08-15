"""Lightweight rewrite-only evaluation — no retrieval or 70B generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from groq import Groq

from src.eval_utils import save_json
from src.history_manager import ChatMessage
from src.logging_config import setup_logging
from src.query_rewriter import rewrite_query
from src.rewrite_eval import (
    RewriteEvalResult,
    evaluate_rewrite,
    summarize_rewrite_eval,
)

BASELINE_LEGACY_MATCH_RATE = 0.25
BASELINE_SOURCE = "results/conversation_eval.json (Phase 4, pre-tuning)"


def load_scenarios(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_history_messages(history_turns: list[dict]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for turn in history_turns:
        messages.append(ChatMessage(role="user", content=turn["user"]))
        messages.append(ChatMessage(role="assistant", content="..."))
    return messages


def run_rewrite_eval(
    *,
    settings,
    scenarios_path: str = "data/eval/conversation_scenarios.json",
    output: str = "results/rewrite_eval.json",
    sleep_seconds: float = 1.0,
) -> dict:
    client = Groq(api_key=settings.groq_api_key)
    scenarios = load_scenarios(Path(scenarios_path))
    results: list[RewriteEvalResult] = []

    for scenario in scenarios:
        history_turns: list[dict] = []

        for turn_index, turn in enumerate(scenario["turns"]):
            if turn_index == 0:
                history_turns.append(turn)
                continue

            history_messages = build_history_messages(history_turns)

            error = None
            rewritten = turn["user"]
            retried = False

            try:
                rewrite_result = rewrite_query(
                    turn["user"],
                    history_messages,
                    client=client,
                    model=settings.rewrite_model,
                    temperature=0.0,
                )
                rewritten = rewrite_result.rewritten_query
                retried = rewrite_result.retried
            except Exception as exc:
                error = str(exc)

            eval_result = evaluate_rewrite(
                scenario_id=scenario["id"],
                category=scenario["category"],
                turn_index=turn_index,
                history=history_turns,
                turn=turn,
                original=turn["user"],
                rewrite=rewritten,
                retried=retried,
                error=error,
            )
            results.append(eval_result)

            status = "PASS" if eval_result.success else "FAIL"
            print(
                f"[{status}] {scenario['id']} turn {turn_index} "
                f"({scenario['category']})"
            )
            if not eval_result.success:
                print(f"         original:  {turn['user']}")
                print(f"         rewrite:   {rewritten}")
                print(f"         failures:  {eval_result.failure_categories}")

            history_turns.append(turn)
            time.sleep(sleep_seconds)

    summary = summarize_rewrite_eval(results)
    failures = [row.to_dict() for row in results if not row.success]

    payload = {
        "description": (
            "Rewrite-only evaluation using the 8B rewrite model. "
            "No retrieval, no 70B generation."
        ),
        "settings": {
            "rewrite_model": settings.rewrite_model,
        },
        "baseline_comparison": {
            "legacy_rewrite_match_rate": BASELINE_LEGACY_MATCH_RATE,
            "legacy_source": BASELINE_SOURCE,
            "current_legacy_rewrite_match_rate": summary["legacy_rewrite_match_rate"],
            "current_rewrite_success_rate": summary["rewrite_success_rate"],
            "delta_vs_baseline": (
                None
                if summary["legacy_rewrite_match_rate"] is None
                else round(
                    summary["legacy_rewrite_match_rate"] - BASELINE_LEGACY_MATCH_RATE,
                    4,
                )
            ),
        },
        "summary": summary,
        "failures": failures,
        "results": [row.to_dict() for row in results],
    }
    save_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate query rewriting only.")
    parser.add_argument(
        "--input",
        default="data/eval/conversation_scenarios.json",
        help="Conversation scenario JSON.",
    )
    parser.add_argument("--output", default="results/rewrite_eval.json")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    payload = run_rewrite_eval(
        settings=settings,
        scenarios_path=args.input,
        output=args.output,
        sleep_seconds=args.sleep_seconds,
    )
    summary = payload["summary"]
    baseline = payload["baseline_comparison"]

    print("=" * 50)
    print("Rewrite-only evaluation summary")
    print("=" * 50)
    print(f"Follow-up turns evaluated: {summary['follow_up_turns']}")
    print(f"Rewrite success rate:        {summary['rewrite_success_rate']:.1%}")
    print(f"Entity/topic preservation: {summary['entity_topic_preservation_rate']:.1%}")
    print(f"Pronoun resolution:          {summary['pronoun_resolution_rate']:.1%}")
    print(f"Standalone rate:             {summary['standalone_rate']:.1%}")
    print(f"History preserved:           {summary['history_preserved_rate']:.1%}")
    print(f"No irrelevant history:       {summary['no_irrelevant_history_rate']:.1%}")
    print(f"Meaning preserved:           {summary['meaning_preserved_rate']:.1%}")
    print(
        "Standalone pass-through:     "
        f"{summary['standalone_pass_through_accuracy']:.1%}"
        if summary["standalone_pass_through_accuracy"] is not None
        else "Standalone pass-through:     n/a"
    )
    print(
        "Out-of-scope pass-through:   "
        f"{summary['out_of_scope_pass_through_accuracy']:.1%}"
        if summary["out_of_scope_pass_through_accuracy"] is not None
        else "Out-of-scope pass-through:   n/a"
    )
    print(
        "Correct pass-through overall:"
        f" {summary['correct_pass_through_rate']:.1%}"
        if summary["correct_pass_through_rate"] is not None
        else "Correct pass-through overall: n/a"
    )
    print(f"Legacy rewrite match:        {summary['legacy_rewrite_match_rate']:.1%}")
    print(
        "Baseline legacy match:       "
        f"{baseline['legacy_rewrite_match_rate']:.1%} "
        f"(delta {baseline['delta_vs_baseline']:+.1%})"
    )
    print(f"Failures logged:             {len(payload['failures'])}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
