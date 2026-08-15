"""Controlled experiment: holdout retrieval with vs without Amharic query normalization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path
from sentence_transformers import SentenceTransformer

from src.eval_utils import compute_retrieval_metrics, load_eval_qas, save_json
from src.logging_config import setup_logging
from src.pipeline import load_vector_collection
from src.query_normalization import normalize_amharic_query
from src.retriever import retrieve


def hit_at_1(retrieved: list[dict], gold_id: str) -> bool:
    return bool(retrieved) and retrieved[0]["document_id"] == gold_id


def retrieved_ids(retrieved: list[dict]) -> list[str]:
    return [doc["document_id"] for doc in retrieved]


def run_condition(
    eval_qas,
    collection,
    embed_model,
    top_k: int,
    *,
    normalize: bool,
) -> tuple[dict[str, Any], list[list[dict]], list[dict[str, Any]]]:
    per_question: list[dict[str, Any]] = []
    retrieved_results: list[list[dict]] = []

    for qa in eval_qas:
        raw_query = qa.question
        query = normalize_amharic_query(raw_query) if normalize else raw_query
        retrieved = retrieve(query, collection, embed_model, top_k=top_k)
        retrieved_results.append(retrieved)
        per_question.append(
            {
                "question": raw_query,
                "query_used": query,
                "query_changed": query != raw_query,
                "gold_document_id": qa.document_id,
                "ground_truth": qa.ground_truth,
                "retrieved_document_ids": retrieved_ids(retrieved),
                "hit_at_1": hit_at_1(retrieved, qa.document_id),
            }
        )

    metrics, failures = compute_retrieval_metrics(eval_qas, retrieved_results, top_k=top_k)
    return metrics.to_dict(), retrieved_results, per_question


def compare_conditions(
    baseline_rows: list[dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    changed = []
    improvements = []
    regressions = []

    for base, norm in zip(baseline_rows, normalized_rows):
        base_ids = base["retrieved_document_ids"]
        norm_ids = norm["retrieved_document_ids"]
        if base_ids != norm_ids:
            changed.append(
                {
                    "question": base["question"],
                    "normalized_query": norm["query_used"],
                    "baseline_top3": base_ids,
                    "normalized_top3": norm_ids,
                    "baseline_hit_at_1": base["hit_at_1"],
                    "normalized_hit_at_1": norm["hit_at_1"],
                }
            )

        if not base["hit_at_1"] and norm["hit_at_1"]:
            improvements.append(
                {
                    "question": base["question"],
                    "normalized_query": norm["query_used"],
                    "gold_document_id": base["gold_document_id"],
                    "baseline_top1": base_ids[0] if base_ids else None,
                    "normalized_top1": norm_ids[0] if norm_ids else None,
                }
            )
        elif base["hit_at_1"] and not norm["hit_at_1"]:
            regressions.append(
                {
                    "question": base["question"],
                    "normalized_query": norm["query_used"],
                    "gold_document_id": base["gold_document_id"],
                    "baseline_top1": base_ids[0] if base_ids else None,
                    "normalized_top1": norm_ids[0] if norm_ids else None,
                }
            )

    base_hits = sum(1 for row in baseline_rows if row["hit_at_1"])
    norm_hits = sum(1 for row in normalized_rows if row["hit_at_1"])
    count = len(baseline_rows)

    return {
        "questions_with_changed_retrieval": len(changed),
        "previously_failed_now_correct": len(improvements),
        "previously_correct_now_incorrect": len(regressions),
        "net_hit_at_1_change": norm_hits - base_hits,
        "net_hit_at_1_change_pp": ((norm_hits - base_hits) / count * 100) if count else 0.0,
        "queries_modified_by_normalization": sum(1 for row in normalized_rows if row["query_changed"]),
        "changed_retrieval_examples": changed[:10],
        "improvements": improvements,
        "regressions": regressions,
    }


def run_experiment(*, output: str = "results/normalization_experiment.json") -> dict:
    settings = get_settings(require_groq=False)
    setup_logging(settings.log_level)

    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    eval_qas = load_eval_qas(holdout_split_path(settings))

    baseline_metrics, _, baseline_rows = run_condition(
        eval_qas,
        collection,
        embed_model,
        settings.top_k,
        normalize=False,
    )
    normalized_metrics, _, normalized_rows = run_condition(
        eval_qas,
        collection,
        embed_model,
        settings.top_k,
        normalize=True,
    )

    comparison = compare_conditions(baseline_rows, normalized_rows)
    net = comparison["net_hit_at_1_change"]
    recommendation = (
        "enable"
        if net > 0 and comparison["previously_correct_now_incorrect"] == 0
        else "disable"
        if net <= 0
        else "review"
    )

    payload = {
        "description": "Controlled holdout experiment: query normalization vs baseline",
        "holdout_path": holdout_split_path(settings),
        "question_count": len(eval_qas),
        "top_k": settings.top_k,
        "embed_model": settings.embed_model,
        "normalization_rules": [
            "ሠ → ሰ (U+1220 → U+1230)",
            "Ethiopic and ASCII punctuation → space",
            "ዓ.ም / እ.ኤ.አ label spacing",
            "date slash spacing (YYYY/MM, MM/YYYY)",
            "NFKC + whitespace collapse",
        ],
        "baseline": {
            "metrics": baseline_metrics,
            "output_path": "results/normalization_experiment_baseline.json",
        },
        "normalized": {
            "metrics": normalized_metrics,
        },
        "comparison": comparison,
        "recommendation": recommendation,
        "production_enabled": False,
    }

    save_json(output, payload)
    save_json(
        "results/normalization_experiment_baseline.json",
        {
            "metrics": baseline_metrics,
            "per_question": baseline_rows,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run normalization A/B on holdout retrieval.")
    parser.add_argument(
        "--output",
        default="results/normalization_experiment.json",
        help="Combined experiment report path.",
    )
    args = parser.parse_args()

    payload = run_experiment(output=args.output)
    base = payload["baseline"]["metrics"]
    norm = payload["normalized"]["metrics"]
    cmp_ = payload["comparison"]

    print("=" * 60)
    print("Normalization controlled experiment")
    print("=" * 60)
    print(f"Questions: {payload['question_count']}")
    print("\nBASELINE (no normalization)")
    print(f"  Hit@1: {base['hit_at_1']:.2%}")
    print(f"  Hit@3: {base['hit_at_3']:.2%}")
    print(f"  MRR:   {base['mrr']:.4f}")
    print("\nNORMALIZED")
    print(f"  Hit@1: {norm['hit_at_1']:.2%}")
    print(f"  Hit@3: {norm['hit_at_3']:.2%}")
    print(f"  MRR:   {norm['mrr']:.4f}")
    print("\nCOMPARISON")
    print(f"  Queries modified:              {cmp_['queries_modified_by_normalization']}")
    print(f"  Retrieval results changed:     {cmp_['questions_with_changed_retrieval']}")
    print(f"  Failed → correct (Hit@1):      {cmp_['previously_failed_now_correct']}")
    print(f"  Correct → failed (Hit@1):      {cmp_['previously_correct_now_incorrect']}")
    print(f"  Net Hit@1 change:              {cmp_['net_hit_at_1_change']:+d} ({cmp_['net_hit_at_1_change_pp']:+.2f} pp)")
    print(f"\nRecommendation: {payload['recommendation'].upper()}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
