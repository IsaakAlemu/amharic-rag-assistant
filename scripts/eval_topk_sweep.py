"""Experiment 1: controlled top-k retrieval sweep on holdout (eval only)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
from src.retriever import retrieve

TOP_K_VALUES = [1, 3, 5, 10]
HIT_K_LEVELS = [1, 3, 5, 10]


def gold_rank(retrieved: list[dict], gold_id: str) -> int | None:
    for rank, doc in enumerate(retrieved, start=1):
        if doc["document_id"] == gold_id:
            return rank
    return None


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_in_top_10"
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    if rank == 4:
        return "rank_4"
    if rank == 5:
        return "rank_5"
    if 6 <= rank <= 10:
        return "rank_6_to_10"
    return "not_in_top_10"


def hit_at_k_from_ranks(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mrr_from_ranks(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r if r else 0.0 for r in ranks) / len(ranks)


def evaluate_top_k(
    eval_qas,
    collection,
    embed_model,
    top_k: int,
) -> dict[str, Any]:
    retrieved_results = [
        retrieve(qa.question, collection, embed_model, top_k=top_k) for qa in eval_qas
    ]
    metrics, failures = compute_retrieval_metrics(
        eval_qas, retrieved_results, top_k=top_k
    )

    ranks = [
        gold_rank(retrieved, qa.document_id)
        for qa, retrieved in zip(eval_qas, retrieved_results)
    ]

    hit_ks = {f"hit_at_{k}": hit_at_k_from_ranks(ranks, k) for k in HIT_K_LEVELS}

    return {
        "top_k": top_k,
        "question_count": len(eval_qas),
        "model_errors": 0,
        "metrics": {
            **metrics.to_dict(),
            **hit_ks,
            "mrr": mrr_from_ranks(ranks),
        },
        "hit_at_1_count": sum(1 for r in ranks if r == 1),
        "failures_logged": len(failures),
        "ranks": ranks,
    }


def failure_recovery_analysis(
    ranks_top10: list[int | None],
    baseline_top_k: int = 3,
) -> dict[str, Any]:
    """Analyze Hit@1 failures (rank != 1) using top-10 rank lookup."""
    total = len(ranks_top10)
    hit1_failures = [i for i, r in enumerate(ranks_top10) if r != 1]

    def count_in_top(n: int) -> int:
        return sum(1 for i in hit1_failures if ranks_top10[i] is not None and ranks_top10[i] <= n)

    not_in_top10 = sum(1 for i in hit1_failures if ranks_top10[i] is None)

    return {
        "baseline_top_k": baseline_top_k,
        "baseline_hit_at_1_failures": len(hit1_failures),
        "total_questions": total,
        "failures_with_gold_in_top_3": count_in_top(3),
        "failures_with_gold_in_top_5": count_in_top(5),
        "failures_with_gold_in_top_10": count_in_top(10),
        "failures_not_in_top_10": not_in_top10,
        "failures_at_rank_2": sum(1 for i in hit1_failures if ranks_top10[i] == 2),
        "failures_at_rank_3": sum(1 for i in hit1_failures if ranks_top10[i] == 3),
        "failures_at_rank_4_or_5": sum(
            1 for i in hit1_failures if ranks_top10[i] in (4, 5)
        ),
        "failures_at_rank_6_to_10": sum(
            1 for i in hit1_failures if ranks_top10[i] is not None and 6 <= ranks_top10[i] <= 10
        ),
    }


def run_sweep(*, output: str = "results/topk_sweep.json") -> dict[str, Any]:
    settings = get_settings(require_groq=False)
    setup_logging(settings.log_level)

    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    eval_qas = load_eval_qas(holdout_split_path(settings))

    sweep_results = []
    rank_info = None
    ranks_top10: list[int | None] = []
    for top_k in TOP_K_VALUES:
        print(f"Evaluating top_k={top_k} ...")
        result = evaluate_top_k(eval_qas, collection, embed_model, top_k)
        if top_k == 10:
            ranks_top10 = result["ranks"]
            buckets = Counter(rank_bucket(r) for r in ranks_top10)
            rank_info = {
                "top_k_used": 10,
                "question_count": len(eval_qas),
                "distribution": dict(buckets),
                "distribution_counts": {
                    "rank_1": buckets.get("rank_1", 0),
                    "rank_2": buckets.get("rank_2", 0),
                    "rank_3": buckets.get("rank_3", 0),
                    "rank_4": buckets.get("rank_4", 0),
                    "rank_5": buckets.get("rank_5", 0),
                    "rank_6_to_10": buckets.get("rank_6_to_10", 0),
                    "not_in_top_10": buckets.get("not_in_top_10", 0),
                },
            }
        result_save = {k: v for k, v in result.items() if k != "ranks"}
        sweep_results.append(result_save)

    assert rank_info is not None
    recovery = failure_recovery_analysis(ranks_top10, baseline_top_k=3)
    rank_info_save = rank_info

    payload = {
        "experiment": "top_k_sweep",
        "description": "Controlled holdout retrieval sweep; only top_k varies",
        "constraints": {
            "embedding_model": settings.embed_model,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "chunking": "whole paragraph",
            "normalize_embeddings": True,
            "similarity": "chroma cosine",
            "holdout_path": holdout_split_path(settings),
            "chroma_path": settings.chroma_path,
            "chunk_count": collection.count(),
        },
        "baseline_reference": {
            "top_k": 3,
            "hit_at_1": 0.7295,
            "hit_at_3": 0.8419,
            "mrr": 0.7812,
            "note": "Prior normalization experiment baseline",
        },
        "top_k_settings": sweep_results,
        "gold_rank_distribution_top10": rank_info_save,
        "hit_at_1_failure_recovery": recovery,
        "reranking_ceiling_analysis": {
            "current_hit_at_1_top3": sweep_results[TOP_K_VALUES.index(3)]["metrics"]["hit_at_1"]
            if 3 in TOP_K_VALUES
            else None,
            "max_hit_at_1_if_rank1_always_selected_from_top10": rank_info_save["distribution_counts"]["rank_1"]
            / rank_info_save["question_count"],
            "max_hit_at_1_if_perfect_rerank_within_top10": (
                rank_info_save["question_count"] - rank_info_save["distribution_counts"]["not_in_top_10"]
            )
            / rank_info_save["question_count"],
            "gap_to_90pct": 0.90
            - (
                (rank_info_save["question_count"] - rank_info_save["distribution_counts"]["not_in_top_10"])
                / rank_info_save["question_count"]
            ),
        },
    }

    save_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-k retrieval sweep on holdout.")
    parser.add_argument("--output", default="results/topk_sweep.json")
    args = parser.parse_args()

    payload = run_sweep(output=args.output)

    print("\n" + "=" * 60)
    print("Top-k sweep complete")
    print("=" * 60)
    for row in payload["top_k_settings"]:
        m = row["metrics"]
        print(
            f"top_k={row['top_k']:2d}  "
            f"Hit@1={m['hit_at_1']:.2%}  Hit@3={m['hit_at_3']:.2%}  "
            f"Hit@5={m['hit_at_5']:.2%}  Hit@10={m['hit_at_10']:.2%}  "
            f"MRR={m['mrr']:.4f}"
        )
    dist = payload["gold_rank_distribution_top10"]["distribution_counts"]
    print("\nGold rank distribution (top_k=10 retrieval):")
    for key in ("rank_1", "rank_2", "rank_3", "rank_4", "rank_5", "rank_6_to_10", "not_in_top_10"):
        print(f"  {key}: {dist[key]}")
    rec = payload["hit_at_1_failure_recovery"]
    print(f"\nHit@1 failures (n={rec['baseline_hit_at_1_failures']}):")
    print(f"  gold in top 3: {rec['failures_with_gold_in_top_3']}")
    print(f"  gold in top 5: {rec['failures_with_gold_in_top_5']}")
    print(f"  gold in top 10: {rec['failures_with_gold_in_top_10']}")
    print(f"  not in top 10: {rec['failures_not_in_top_10']}")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
