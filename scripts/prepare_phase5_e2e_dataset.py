"""Script to prepare a representative, balanced 35-question evaluation dataset for Phase 5 E2E evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path
from src.eval_utils import load_eval_qas, save_json
from sentence_transformers import SentenceTransformer
from src.pipeline import load_vector_collection, get_bm25_index
from src.retriever import retrieve
from src.hybrid_retriever import reciprocal_rank_fusion


def main():
    settings = get_settings(require_groq=False)
    out_path = Path("data/phase5_e2e_questions.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline models for holdout retrieval verification...")
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    bm25_index = get_bm25_index(settings)

    # 1. Load holdout QAs
    holdout_qas = load_eval_qas(holdout_split_path(settings))
    print(f"Loaded {len(holdout_qas)} holdout QAs.")

    # 2. Map existing questions from prior pilots/evals
    prior_pilot_ids = {
        282023, 281768, 282123, 282578, 282848,
        282052, 275102, 282553, 282363, 207400,
        280806, 281781
    }

    # Verify retrieval outcome for holdout questions using current production retriever
    selected_items = []
    seen_ids = set()

    # Process prior items first
    for qa in holdout_qas:
        if qa.question_id in prior_pilot_ids and qa.question_id not in seen_ids:
            dense_sources = retrieve(qa.question, collection, embed_model, top_k=settings.top_k * 2)
            lexical_sources = bm25_index.search(qa.question, top_k=settings.top_k * 2)
            sources = reciprocal_rank_fusion(dense_sources, lexical_sources, top_k=settings.top_k)
            retrieved_ids = [str(s["document_id"]) for s in sources]
            outcome = "hit" if str(qa.document_id) in retrieved_ids else "miss"

            selected_items.append({
                "question_id": qa.question_id,
                "question": qa.question,
                "gold_document_id": str(qa.document_id),
                "reference_answer": qa.ground_truth,
                "retrieval_outcome": outcome,
                "source": "prior_eval_recovery"
            })
            seen_ids.add(qa.question_id)

    # Now select additional hits and misses to reach a balanced ~35 questions (e.g. ~24 hits, ~11 misses)
    hits_needed = 24 - sum(1 for item in selected_items if item["retrieval_outcome"] == "hit")
    misses_needed = 11 - sum(1 for item in selected_items if item["retrieval_outcome"] == "miss")

    print(f"Recovered {len(selected_items)} prior items. Collecting {hits_needed} hits and {misses_needed} misses...")

    for qa in holdout_qas:
        if qa.question_id in seen_ids:
            continue
        
        dense_sources = retrieve(qa.question, collection, embed_model, top_k=settings.top_k * 2)
        lexical_sources = bm25_index.search(qa.question, top_k=settings.top_k * 2)
        sources = reciprocal_rank_fusion(dense_sources, lexical_sources, top_k=settings.top_k)
        retrieved_ids = [str(s["document_id"]) for s in sources]
        outcome = "hit" if str(qa.document_id) in retrieved_ids else "miss"

        if outcome == "hit" and hits_needed > 0:
            selected_items.append({
                "question_id": qa.question_id,
                "question": qa.question,
                "gold_document_id": str(qa.document_id),
                "reference_answer": qa.ground_truth,
                "retrieval_outcome": outcome,
                "source": "holdout_sample"
            })
            seen_ids.add(qa.question_id)
            hits_needed -= 1
        elif outcome == "miss" and misses_needed > 0:
            selected_items.append({
                "question_id": qa.question_id,
                "question": qa.question,
                "gold_document_id": str(qa.document_id),
                "reference_answer": qa.ground_truth,
                "retrieval_outcome": outcome,
                "source": "holdout_sample"
            })
            seen_ids.add(qa.question_id)
            misses_needed -= 1

        if hits_needed <= 0 and misses_needed <= 0:
            break

    print(f"Total dataset prepared: {len(selected_items)} questions.")
    hit_count = sum(1 for x in selected_items if x["retrieval_outcome"] == "hit")
    miss_count = sum(1 for x in selected_items if x["retrieval_outcome"] == "miss")
    print(f"Breakdown: {hit_count} Hits, {miss_count} Misses.")

    save_json(out_path, {"questions": selected_items, "total": len(selected_items), "hits": hit_count, "misses": miss_count})
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
