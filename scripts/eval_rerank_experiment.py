"""Experiment 2: bounded multilingual cross-encoder reranking evaluation.

Eval-only. Does not modify production retrieval code.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.eval_utils import EvalQA, load_eval_qas, save_json
from src.logging_config import setup_logging
from src.pipeline import load_vector_collection
from src.retriever import retrieve

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
RETRIEVE_K = 10
HIT_K_LEVELS = [1, 3, 5, 10]
DEFAULT_MAX_RUNTIME_SECONDS = 600
DEFAULT_CHECKPOINT_EVERY = 25
DEFAULT_RERANK_BATCH_SIZE = 32


def gold_rank(retrieved: list[dict], gold_id: str) -> int | None:
    for rank, doc in enumerate(retrieved, start=1):
        if doc["document_id"] == gold_id:
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r if r else 0.0 for r in ranks) / len(ranks)


def strip_docs(docs: list[dict]) -> list[dict]:
    return [
        {
            "document_id": d["document_id"],
            "distance": d.get("distance"),
            "rank": d.get("rank"),
            "rerank_score": d.get("rerank_score"),
        }
        for d in docs
    ]


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def analyze_question(
    qa: EvalQA,
    before: list[dict],
    after: list[dict],
) -> dict[str, Any]:
    rank_before = gold_rank(before, qa.document_id)
    rank_after = gold_rank(after, qa.document_id)
    hit1_before = rank_before == 1
    hit1_after = rank_after == 1

    row: dict[str, Any] = {
        "question": qa.question,
        "gold_document_id": qa.document_id,
        "rank_before": rank_before,
        "rank_after": rank_after,
        "hit_at_1_before": hit1_before,
        "hit_at_1_after": hit1_after,
        "order_changed": [d["document_id"] for d in before] != [d["document_id"] for d in after],
        "before_top10": strip_docs(before),
        "after_top10": strip_docs(after),
    }

    if rank_before is None:
        row["case"] = "A_retrieval_recall"
    elif not hit1_before and hit1_after:
        row["case"] = "B_solved"
        if rank_before == 2:
            row["promoted_from"] = "rank_2"
        elif rank_before == 3:
            row["promoted_from"] = "rank_3"
        elif rank_before in (4, 5):
            row["promoted_from"] = "rank_4_5"
        else:
            row["promoted_from"] = "rank_6_10"
    elif hit1_before and not hit1_after:
        row["case"] = "regression"
    elif not hit1_after:
        row["case"] = "B_still_failing"
    else:
        row["case"] = "unchanged_correct"

    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks_before = [r["rank_before"] for r in rows]
    ranks_after = [r["rank_after"] for r in rows]

    metrics_before = {f"hit_at_{k}": hit_at_k(ranks_before, k) for k in HIT_K_LEVELS}
    metrics_before["mrr"] = mrr(ranks_before)
    metrics_before["count"] = len(rows)

    metrics_after = {f"hit_at_{k}": hit_at_k(ranks_after, k) for k in HIT_K_LEVELS}
    metrics_after["mrr"] = mrr(ranks_after)
    metrics_after["count"] = len(rows)

    promoted = {"rank_2": 0, "rank_3": 0, "rank_4_5": 0, "rank_6_10": 0}
    for r in rows:
        if r.get("case") == "B_solved":
            key = r.get("promoted_from", "rank_6_10")
            if key in promoted:
                promoted[key] += 1
            else:
                promoted["rank_6_10"] += 1

    return {
        "before": metrics_before,
        "after": metrics_after,
        "delta_hit_at_1_pp": (metrics_after["hit_at_1"] - metrics_before["hit_at_1"]) * 100,
        "recoveries": sum(1 for r in rows if r.get("case") == "B_solved"),
        "regressions": sum(1 for r in rows if r.get("case") == "regression"),
        "rank_order_changed": sum(1 for r in rows if r["order_changed"]),
        "case_A_gold_not_in_top10": sum(1 for r in rows if r.get("case") == "A_retrieval_recall"),
        "case_B_still_failing": sum(1 for r in rows if r.get("case") == "B_still_failing"),
        "promoted_to_rank1_from": promoted,
        "recovery_cases": [r for r in rows if r.get("case") == "B_solved"],
        "regression_cases": [r for r in rows if r.get("case") == "regression"],
        "ranking_failure_cases": [r for r in rows if r.get("case") == "B_still_failing"],
    }


def run_bounded_experiment(
    *,
    reranker_model: str = DEFAULT_RERANKER,
    limit: int | None = None,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    output: str = "results/rerank_experiment.json",
    checkpoint_path: str = "results/rerank_experiment_checkpoint.json",
) -> dict[str, Any]:
    settings = get_settings(require_groq=False)
    setup_logging(settings.log_level)

    all_qas = load_eval_qas(holdout_split_path(settings))
    total_available = len(all_qas)
    eval_qas = all_qas[:limit] if limit is not None else all_qas
    target_count = len(eval_qas)

    print(f"Loading embedder: {settings.embed_model}")
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)

    print(f"Loading reranker: {reranker_model}")
    reranker_load_start = time.perf_counter()
    reranker = CrossEncoder(reranker_model)
    reranker_load_seconds = time.perf_counter() - reranker_load_start
    print(f"Reranker loaded in {reranker_load_seconds:.1f}s")

    experiment_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    question_times: list[float] = []
    stopped_reason = "completed"
    processed = 0

    pending_pairs: list[tuple[str, str]] = []
    pending_meta: list[tuple[EvalQA, list[dict]]] = []

    def flush_batch() -> bool:
        nonlocal processed, stopped_reason, rows

        if not pending_pairs:
            return True

        batch_start = time.perf_counter()
        scores = list(
            reranker.predict(pending_pairs, batch_size=rerank_batch_size, show_progress_bar=False)
        )
        batch_elapsed = time.perf_counter() - batch_start
        per_q = batch_elapsed / max(len(pending_meta), 1)

        offset = 0
        for qa, candidates in pending_meta:
            elapsed = time.perf_counter() - experiment_start
            if elapsed >= max_runtime_seconds:
                stopped_reason = "max_runtime_exceeded"
                pending_pairs.clear()
                pending_meta.clear()
                return False

            n = len(candidates)
            score_chunk = scores[offset : offset + n]
            offset += n

            before = list(candidates)
            reranked = []
            for doc, score in zip(candidates, score_chunk):
                reranked.append({**doc, "rerank_score": float(score)})
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            for i, doc in enumerate(reranked, start=1):
                doc["rank"] = i

            rows.append(analyze_question(qa, before, reranked))
            processed += 1
            question_times.append(per_q)

            avg = sum(question_times) / len(question_times)
            remaining = target_count - processed
            eta = avg * remaining
            hit1_after = sum(1 for r in rows if r["hit_at_1_after"])
            print(
                f"[{processed}/{target_count}] reranked  "
                f"batch={batch_elapsed:.1f}s  avg={avg:.2f}s/q  "
                f"elapsed={elapsed:.0f}s  ETA={format_eta(eta)}  "
                f"Hit@1 after rerank={hit1_after}/{processed}",
                flush=True,
            )

            if processed % checkpoint_every == 0:
                _write_checkpoint(
                    checkpoint_path,
                    rows,
                    processed,
                    target_count,
                    question_times,
                    stopped_reason,
                    reranker_model,
                    max_runtime_seconds,
                )

        pending_pairs.clear()
        pending_meta.clear()
        return True

    for idx, qa in enumerate(eval_qas):
        elapsed = time.perf_counter() - experiment_start
        if elapsed >= max_runtime_seconds:
            stopped_reason = "max_runtime_exceeded"
            break

        candidates = retrieve(qa.question, collection, embed_model, top_k=RETRIEVE_K)
        pending_meta.append((qa, candidates))
        pending_pairs.extend((qa.question, doc["text"]) for doc in candidates)

        # Process in mini-batches of up to 5 questions (50 pairs) for progress visibility
        if len(pending_meta) >= 5:
            if not flush_batch():
                break

    if stopped_reason != "max_runtime_exceeded":
        flush_batch()

    total_elapsed = time.perf_counter() - experiment_start
    avg_per_question = sum(question_times) / len(question_times) if question_times else 0.0
    estimated_full_runtime_seconds = avg_per_question * total_available + reranker_load_seconds

    summary = aggregate_rows(rows)
    payload = {
        "experiment": "multilingual_cross_encoder_reranking_bounded",
        "status": stopped_reason,
        "reranker_model": reranker_model,
        "reranker_load_seconds": round(reranker_load_seconds, 2),
        "constraints": {
            "max_runtime_seconds": max_runtime_seconds,
            "checkpoint_every": checkpoint_every,
            "rerank_batch_size": rerank_batch_size,
            "retrieve_top_k": RETRIEVE_K,
            "eval_only": True,
        },
        "scope": {
            "questions_processed": processed,
            "questions_target": target_count,
            "questions_available": total_available,
            "limit": limit,
        },
        "timing": {
            "total_elapsed_seconds": round(total_elapsed, 2),
            "avg_seconds_per_question": round(avg_per_question, 3),
            "estimated_full_329_runtime_seconds": round(estimated_full_runtime_seconds, 1),
            "estimated_full_329_runtime_minutes": round(estimated_full_runtime_seconds / 60, 1),
            "question_times_seconds": [round(t, 3) for t in question_times],
        },
        "baseline_before_rerank": {"metrics": summary["before"]},
        "after_rerank": {"metrics": summary["after"]},
        "delta": {
            "hit_at_1_pp": summary["delta_hit_at_1_pp"],
            "recoveries": summary["recoveries"],
            "regressions": summary["regressions"],
        },
        "problem_decomposition": {
            "case_A_gold_not_in_e5_top10": summary["case_A_gold_not_in_top10"],
            "case_B_solved_by_reranker": summary["recoveries"],
            "case_B_still_failing_after_rerank": summary["case_B_still_failing"],
            "promoted_to_rank1_from": summary["promoted_to_rank1_from"],
        },
        "summary_counts": {
            "rank_order_changed": summary["rank_order_changed"],
            "hit_at_1_failures_recovered": summary["recoveries"],
            "previously_correct_made_worse": summary["regressions"],
        },
        "recovery_cases": summary["recovery_cases"],
        "regression_cases": summary["regression_cases"],
        "ranking_failure_cases": summary["ranking_failure_cases"],
    }

    save_json(output, payload)
    _write_checkpoint(
        checkpoint_path,
        rows,
        processed,
        target_count,
        question_times,
        stopped_reason,
        reranker_model,
        max_runtime_seconds,
    )
    return payload


def _write_checkpoint(
    path: str,
    rows: list[dict],
    processed: int,
    target: int,
    question_times: list[float],
    status: str,
    reranker_model: str,
    max_runtime_seconds: int,
) -> None:
    avg = sum(question_times) / len(question_times) if question_times else 0.0
    save_json(
        path,
        {
            "status": status,
            "reranker_model": reranker_model,
            "processed": processed,
            "target": target,
            "max_runtime_seconds": max_runtime_seconds,
            "avg_seconds_per_question": round(avg, 3),
            "rows": rows,
        },
    )
    print(f"  >> checkpoint saved ({processed}/{target}) -> {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded reranking experiment.")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--limit", type=int, default=20, help="Questions to evaluate (pilot default 20).")
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--rerank-batch-size", type=int, default=DEFAULT_RERANK_BATCH_SIZE)
    parser.add_argument("--output", default="results/rerank_experiment.json")
    parser.add_argument("--checkpoint-path", default="results/rerank_experiment_checkpoint.json")
    args = parser.parse_args()

    payload = run_bounded_experiment(
        reranker_model=args.reranker,
        limit=args.limit,
        max_runtime_seconds=args.max_runtime_seconds,
        checkpoint_every=args.checkpoint_every,
        rerank_batch_size=args.rerank_batch_size,
        output=args.output,
        checkpoint_path=args.checkpoint_path,
    )

    t = payload["timing"]
    b = payload["baseline_before_rerank"]["metrics"]
    a = payload["after_rerank"]["metrics"]

    print("\n" + "=" * 60)
    print("Bounded reranking experiment — PILOT COMPLETE")
    print("=" * 60)
    print(f"Status: {payload['status']}")
    print(f"Processed: {payload['scope']['questions_processed']}/{payload['scope']['questions_target']}")
    print(f"Avg time/question: {t['avg_seconds_per_question']:.3f}s")
    print(f"Estimated full 329 runtime: {t['estimated_full_329_runtime_minutes']:.1f} min "
          f"({t['estimated_full_329_runtime_seconds']:.0f}s incl. model load)")
    print(f"\nBEFORE  Hit@1={b['hit_at_1']:.2%}  MRR={b['mrr']:.4f}")
    print(f"AFTER   Hit@1={a['hit_at_1']:.2%}  MRR={a['mrr']:.4f}")
    print(f"Delta Hit@1: {payload['delta']['hit_at_1_pp']:+.2f} pp  "
          f"recovered={payload['delta']['recoveries']}  "
          f"regressed={payload['delta']['regressions']}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
