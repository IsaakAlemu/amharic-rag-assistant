"""Controlled 8B vs 70B benchmark on representative synthesis benchmark questions."""

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
from sentence_transformers import SentenceTransformer

from src.eval_utils import normalize_text, save_json
from src.llm import REFUSAL_PHRASE, generate_answer
from src.logging_config import setup_logging
from src.pipeline import load_vector_collection
from src.prompt_builder import build_prompt
from src.retriever import retrieve
from src.token_counter import TokenCounter


def load_benchmark(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ground_truth_match(answer: str, ground_truth: str) -> bool | None:
    if not ground_truth:
        return None
    return normalize_text(ground_truth) in normalize_text(answer)


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE in answer


def run_model_benchmark(
    *,
    settings,
    benchmark_path: str = "data/eval/synthesis_benchmark.json",
    output: str = "results/model_benchmark.json",
    sleep_seconds: float = 2.0,
) -> dict:
    token_counter = TokenCounter()
    groq_client = Groq(api_key=settings.groq_api_key)
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    cases = load_benchmark(Path(benchmark_path))

    models = {
        "8b": settings.model_8b,
        "70b": settings.model_70b,
    }

    results = []
    summary_by_model: dict[str, dict] = {
        name: {
            "successes": 0,
            "failures": 0,
            "token_limit_failures": 0,
            "refusals": 0,
            "correct_refusals": 0,
            "ground_truth_hits": 0,
            "ground_truth_scored": 0,
            "latencies_ms": [],
            "prompt_tokens": [],
            "completion_tokens": [],
            "estimated_prompt_tokens": [],
        }
        for name in models
    }

    for case in cases:
        question = case["question"]
        retrieved = retrieve(question, collection, embed_model, top_k=settings.top_k)
        prompt = build_prompt(question, retrieved)
        estimated_prompt_tokens = token_counter.count(prompt)

        row = {
            "id": case["id"],
            "category": case["category"],
            "question": question,
            "expects_refusal": case.get("expects_refusal", False),
            "ground_truth": case.get("ground_truth", ""),
            "retrieved_document_ids": [doc["document_id"] for doc in retrieved],
            "models": {},
        }

        for model_name, model_id in models.items():
            model_stats = summary_by_model[model_name]
            t0 = time.perf_counter()
            error = None
            answer = ""
            prompt_tokens = None
            completion_tokens = None

            try:
                generation = generate_answer(
                    prompt,
                    groq_client,
                    model=model_id,
                    temperature=settings.temperature,
                )
                answer = generation.text
                prompt_tokens = generation.prompt_tokens
                completion_tokens = generation.completion_tokens
                model_stats["successes"] += 1
            except Exception as exc:
                error = str(exc)
                model_stats["failures"] += 1
                if "token" in error.lower():
                    model_stats["token_limit_failures"] += 1

            latency_ms = (time.perf_counter() - t0) * 1000
            model_stats["latencies_ms"].append(latency_ms)
            model_stats["estimated_prompt_tokens"].append(estimated_prompt_tokens)
            if prompt_tokens is not None:
                model_stats["prompt_tokens"].append(prompt_tokens)

            refused = is_refusal(answer)
            if refused:
                model_stats["refusals"] += 1
            if case.get("expects_refusal") and refused:
                model_stats["correct_refusals"] += 1

            gt_match = ground_truth_match(answer, case.get("ground_truth", ""))
            if gt_match is not None:
                model_stats["ground_truth_scored"] += 1
                if gt_match:
                    model_stats["ground_truth_hits"] += 1

            row["models"][model_name] = {
                "model_id": model_id,
                "answer": answer,
                "error": error,
                "refusal": refused,
                "ground_truth_match": gt_match,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_prompt_tokens": estimated_prompt_tokens,
            }

            time.sleep(sleep_seconds)

        results.append(row)
        print(
            f"[{case['category']}] {case['id']} — "
            f"8B: {'OK' if not row['models']['8b']['error'] else 'ERR'} | "
            f"70B: {'OK' if not row['models']['70b']['error'] else 'ERR'}"
        )

    final_summary = {}
    for model_name, stats in summary_by_model.items():
        out_of_scope_total = sum(
            1 for case in cases if case.get("expects_refusal") and case["category"] == "out_of_scope"
        )
        final_summary[model_name] = {
            "model_id": models[model_name],
            "successes": stats["successes"],
            "failures": stats["failures"],
            "token_limit_failures": stats["token_limit_failures"],
            "refusal_rate": stats["refusals"] / len(cases) if cases else 0,
            "out_of_scope_refusal_accuracy": (
                stats["correct_refusals"] / out_of_scope_total if out_of_scope_total else None
            ),
            "ground_truth_match_rate": (
                stats["ground_truth_hits"] / stats["ground_truth_scored"]
                if stats["ground_truth_scored"]
                else None
            ),
            "latency_ms_mean": round(
                sum(stats["latencies_ms"]) / len(stats["latencies_ms"]), 1
            )
            if stats["latencies_ms"]
            else 0,
            "prompt_tokens_mean": round(
                sum(stats["prompt_tokens"]) / len(stats["prompt_tokens"]), 1
            )
            if stats["prompt_tokens"]
            else None,
            "estimated_prompt_tokens_mean": round(
                sum(stats["estimated_prompt_tokens"]) / len(stats["estimated_prompt_tokens"]),
                1,
            ),
            "completion_tokens_mean": round(
                sum(
                    row["models"][model_name]["completion_tokens"]
                    for row in results
                    if row["models"][model_name]["completion_tokens"] is not None
                )
                / max(
                    1,
                    sum(
                        1
                        for row in results
                        if row["models"][model_name]["completion_tokens"] is not None
                    ),
                ),
                1,
            ),
        }

    payload = {
        "description": (
            "Controlled comparison of 8B and 70B models on the same retrieved prompts. "
            "Ground-truth matching is exact substring only; use generation eval for "
            "semantic correctness and faithfulness scoring."
        ),
        "benchmark_path": benchmark_path,
        "top_k": settings.top_k,
        "summary": final_summary,
        "results": results,
    }
    save_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 8B vs 70B models.")
    parser.add_argument(
        "--benchmark",
        default="data/eval/synthesis_benchmark.json",
        help="Benchmark JSON path.",
    )
    parser.add_argument("--output", default="results/model_benchmark.json")
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    payload = run_model_benchmark(
        settings=settings,
        benchmark_path=args.benchmark,
        output=args.output,
        sleep_seconds=args.sleep_seconds,
    )

    print("=" * 50)
    print("Model benchmark summary")
    print("=" * 50)
    for model_name, stats in payload["summary"].items():
        print(f"{model_name.upper()} ({stats['model_id']})")
        print(f"  Successes: {stats['successes']}  Failures: {stats['failures']}")
        print(f"  Token-limit failures: {stats['token_limit_failures']}")
        print(f"  Ground-truth match rate: {stats['ground_truth_match_rate']}")
        print(f"  Out-of-scope refusal accuracy: {stats['out_of_scope_refusal_accuracy']}")
        print(f"  Mean latency (ms): {stats['latency_ms_mean']}")
        print(f"  Mean prompt tokens (API): {stats['prompt_tokens_mean']}")
        print(f"  Mean prompt tokens (estimated): {stats['estimated_prompt_tokens_mean']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
