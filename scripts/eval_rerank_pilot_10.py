"""Bounded 10-question reranking pilot (eval only; does not modify production code)."""

from __future__ import annotations

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

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RETRIEVE_K = 10
PILOT_SIZE = 10
MAX_PAIRS = 100
MAX_RUNTIME_SECONDS = 300
OUTPUT_PATH = "results/reranking_pilot_10_results.json"

# Diverse pilot set (gold ranks 2-10) from holdout scouting with E5 top-10 retrieval.
PILOT_QUESTION_IDS = [
    282052,
    281983,
    275102,
    282719,
    280808,
    281771,
    280809,
    280639,
    280683,
    281770,
]


def gold_rank(retrieved: list[dict], gold_id: str) -> int | None:
    for rank, doc in enumerate(retrieved, start=1):
        if doc["document_id"] == gold_id:
            return rank
    return None


def classify_result(summary: dict[str, Any]) -> str:
    promoted = summary["promoted_to_rank1"]
    demoted = summary["demoted"]
    avg_latency_ms = summary["mean_reranking_latency_ms"]

    if promoted >= 3 and demoted == 0 and avg_latency_ms <= 2000:
        return "PROMISING"
    if promoted == 0 and demoted >= 2:
        return "NOT WORTH ADDING"
    if promoted >= 1 and demoted <= 1 and avg_latency_ms <= 5000:
        return "PROMISING" if promoted >= 2 else "WEAK"
    if avg_latency_ms > 5000 or (promoted <= 1 and demoted >= promoted):
        return "NOT WORTH ADDING" if demoted > promoted else "WEAK"
    return "WEAK"


def run_pilot() -> dict[str, Any]:
    settings = get_settings(require_groq=False)
    setup_logging(settings.log_level)
    experiment_start = time.perf_counter()

    all_qas = load_eval_qas(holdout_split_path(settings))
    qa_by_id = {qa.question_id: qa for qa in all_qas if qa.question_id is not None}
    missing = [qid for qid in PILOT_QUESTION_IDS if qid not in qa_by_id]
    if missing:
        raise RuntimeError(f"Pilot question IDs not found in holdout: {missing}")
    if len(PILOT_QUESTION_IDS) != PILOT_SIZE:
        raise RuntimeError(f"Expected {PILOT_SIZE} pilot IDs, got {len(PILOT_QUESTION_IDS)}.")

    print("Loading embedder and Chroma collection...")
    embed_load_start = time.perf_counter()
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    embed_load_seconds = time.perf_counter() - embed_load_start

    print(f"Retrieving top-{RETRIEVE_K} baseline candidates for {PILOT_SIZE} pilot questions...")
    retrieval_start = time.perf_counter()
    selected: list[dict[str, Any]] = []

    for qid in PILOT_QUESTION_IDS:
        qa = qa_by_id[qid]
        holdout_index = next(i for i, q in enumerate(all_qas) if q.question_id == qid)
        retrieved = retrieve(qa.question, collection, embed_model, top_k=RETRIEVE_K)
        rank = gold_rank(retrieved, qa.document_id)
        if rank is None or rank < 2 or rank > 10:
            raise RuntimeError(
                f"Pilot question {qid} has gold rank {rank}; expected ranks 2-10."
            )
        selected.append(
            {
                "holdout_index": holdout_index,
                "question_id": qid,
                "question": qa.question,
                "gold_document_id": qa.document_id,
                "baseline_gold_rank": rank,
                "baseline_top10_document_ids": [d["document_id"] for d in retrieved],
                "baseline_top10": retrieved,
                "qa": qa,
            }
        )

    retrieval_seconds = time.perf_counter() - retrieval_start

    print("\nSelected pilot questions (gold in ranks 2-10):")
    for row in selected:
        print(
            f"  question_id={row['question_id']}  baseline_gold_rank={row['baseline_gold_rank']}"
        )

    print(f"\nLoading reranker: {RERANKER_MODEL}")
    reranker_load_start = time.perf_counter()
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    reranker_load_seconds = time.perf_counter() - reranker_load_start
    print(f"Reranker loaded in {reranker_load_seconds:.2f}s")

    all_pairs: list[tuple[str, str]] = []
    pair_counts: list[int] = []
    for row in selected:
        qa: EvalQA = row["qa"]
        before = list(row["baseline_top10"])
        pairs = [(qa.question, doc["text"]) for doc in before]
        if len(pairs) != RETRIEVE_K:
            raise RuntimeError(f"Expected {RETRIEVE_K} candidates, got {len(pairs)}.")
        all_pairs.extend(pairs)
        pair_counts.append(len(pairs))

    if len(all_pairs) > MAX_PAIRS:
        raise RuntimeError(f"Scored {len(all_pairs)} pairs; hard limit is {MAX_PAIRS}.")

    warmup_q, warmup_text = all_pairs[0]
    reranker.predict([(warmup_q, warmup_text)], batch_size=1, show_progress_bar=False)

    rerank_start = time.perf_counter()
    all_scores = list(
        reranker.predict(all_pairs, batch_size=32, show_progress_bar=False)
    )
    total_rerank_seconds = time.perf_counter() - rerank_start
    per_query_rerank_s = total_rerank_seconds / len(selected)
    rerank_latencies = [per_query_rerank_s] * len(selected)

    per_question: list[dict[str, Any]] = []
    offset = 0
    for row, n in zip(selected, pair_counts):
        qa: EvalQA = row["qa"]
        before = list(row["baseline_top10"])
        scores = all_scores[offset : offset + n]
        offset += n

        reranked = []
        for doc, score in zip(before, scores):
            reranked.append({**doc, "rerank_score": float(score)})
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        rank_before = row["baseline_gold_rank"]
        rank_after = gold_rank(reranked, qa.document_id)
        gold_score = next(
            (float(s) for d, s in zip(before, scores) if d["document_id"] == qa.document_id),
            None,
        )

        hit1_before = rank_before == 1
        hit1_after = rank_after == 1
        moved_up = (
            rank_before is not None
            and rank_after is not None
            and rank_after < rank_before
        )
        moved_down = (
            rank_before is not None
            and rank_after is not None
            and rank_after > rank_before
        )

        per_question.append(
            {
                "question_id": row["question_id"],
                "holdout_index": row["holdout_index"],
                "question": qa.question,
                "gold_document_id": qa.document_id,
                "baseline_gold_rank": rank_before,
                "reranked_gold_rank": rank_after,
                "gold_reranker_score": gold_score,
                "baseline_hit_at_1": hit1_before,
                "reranked_hit_at_1": hit1_after,
                "gold_moved_upward": moved_up,
                "gold_moved_downward": moved_down,
                "gold_reached_rank_1": hit1_after,
                "reranking_latency_ms": round(rerank_latency_s * 1000, 2),
                "baseline_top10_document_ids": row["baseline_top10_document_ids"],
                "reranked_top10_document_ids": [d["document_id"] for d in reranked],
                "reranked_top10_scores": [
                    {"document_id": d["document_id"], "rerank_score": d["rerank_score"]}
                    for d in reranked
                ],
            }
        )

    baseline_hit1 = sum(1 for r in per_question if r["baseline_hit_at_1"])
    reranked_hit1 = sum(1 for r in per_question if r["reranked_hit_at_1"])
    promoted = sum(
        1 for r in per_question if r["gold_reached_rank_1"] and not r["baseline_hit_at_1"]
    )
    improved = sum(1 for r in per_question if r["gold_moved_upward"])
    unchanged = sum(
        1 for r in per_question if r["baseline_gold_rank"] == r["reranked_gold_rank"]
    )
    demoted = sum(1 for r in per_question if r["gold_moved_downward"])

    mean_baseline_rank = sum(r["baseline_gold_rank"] for r in per_question) / len(per_question)
    mean_reranked_rank = sum(
        r["reranked_gold_rank"] if r["reranked_gold_rank"] is not None else 11
        for r in per_question
    ) / len(per_question)
    mean_rerank_latency_ms = (sum(rerank_latencies) / len(rerank_latencies)) * 1000
    total_runtime = time.perf_counter() - experiment_start

    summary = {
        "baseline_hit_at_1": baseline_hit1,
        "reranked_hit_at_1": reranked_hit1,
        "promoted_to_rank1": promoted,
        "improved_rank": improved,
        "unchanged": unchanged,
        "demoted": demoted,
        "mean_baseline_gold_rank": round(mean_baseline_rank, 3),
        "mean_reranked_gold_rank": round(mean_reranked_rank, 3),
        "mean_reranking_latency_ms": round(mean_rerank_latency_ms, 2),
        "total_runtime_seconds": round(total_runtime, 2),
    }

    decision = classify_result(summary)

    payload = {
        "experiment": "reranking_pilot_10",
        "description": "Bounded pilot: rerank existing top-10 for 10 holdout questions with gold at ranks 2-10",
        "eval_only": True,
        "production_code_modified": False,
        "reranker_model": RERANKER_MODEL,
        "constraints": {
            "pilot_questions": PILOT_SIZE,
            "candidates_per_query": RETRIEVE_K,
            "max_scoring_pairs": MAX_PAIRS,
            "max_reranking_phase_runtime_seconds": MAX_RUNTIME_SECONDS,
            "no_llm_apis": True,
            "reranker_reorders_top10_only": True,
        },
        "timing": {
            "embedder_and_collection_load_seconds": round(embed_load_seconds, 2),
            "retrieval_baseline_seconds": round(retrieval_seconds, 2),
            "reranker_model_load_seconds": round(reranker_load_seconds, 2),
            "total_reranking_seconds": round(total_rerank_seconds, 2),
            "mean_reranking_latency_ms_per_query": summary["mean_reranking_latency_ms"],
            "total_runtime_seconds": summary["total_runtime_seconds"],
            "reranking_latencies_ms": [round(s * 1000, 2) for s in rerank_latencies],
            "note": "Per-query reranking latency is total batched rerank time / 10.",
        },
        "selection": {
            "method": "pre-scouted diverse holdout IDs (gold ranks 2-10)",
            "selected_question_ids": [r["question_id"] for r in selected],
            "selected_baseline_ranks": [r["baseline_gold_rank"] for r in selected],
        },
        "aggregate": summary,
        "decision": decision,
        "per_question": per_question,
    }

    save_json(OUTPUT_PATH, payload)
    return payload


def main() -> None:
    payload = run_pilot()
    s = payload["aggregate"]

    print("\n" + "=" * 60)
    print("Reranking pilot (10 questions) — COMPLETE")
    print("=" * 60)
    print(f"Baseline Hit@1: {s['baseline_hit_at_1']}/10")
    print(f"Reranked Hit@1: {s['reranked_hit_at_1']}/10")
    print(f"Promoted to #1: {s['promoted_to_rank1']}/10")
    print(f"Improved rank: {s['improved_rank']}/10")
    print(f"Unchanged: {s['unchanged']}/10")
    print(f"Demoted: {s['demoted']}/10")
    print(f"Average baseline rank: {s['mean_baseline_gold_rank']:.2f}")
    print(f"Average reranked rank: {s['mean_reranked_gold_rank']:.2f}")
    print(f"Average reranking latency: {s['mean_reranking_latency_ms']:.1f} ms/query")
    print(f"Total runtime: {s['total_runtime_seconds']:.1f} seconds")
    print(f"Decision: {payload['decision']}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
