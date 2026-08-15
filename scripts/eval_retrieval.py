"""Evaluate retrieval quality on the hold-out split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path, split_manifest_path, train_split_path
from sentence_transformers import SentenceTransformer

from src.eval_utils import (
    compute_retrieval_metrics,
    load_eval_qas,
    save_json,
)
from src.logging_config import setup_logging
from src.pipeline import load_eval_collection, load_vector_collection
from src.retriever import retrieve


def ensure_splits_exist(settings) -> None:
    if not Path(train_split_path(settings)).exists() or not Path(
        holdout_split_path(settings)
    ).exists():
        from scripts.split_dataset import main as split_main

        print("Splits not found — creating train/hold-out splits...")
        split_main()


def run_retrieval_eval(
    *,
    settings,
    limit: int | None = None,
    output: str = "results/retrieval_eval.json",
    index_mode: str = "full",
) -> dict:
    ensure_splits_exist(settings)

    embed_model = SentenceTransformer(settings.embed_model)
    if index_mode == "train_only":
        collection = load_eval_collection(settings, embed_model)
        index_path = settings.eval_chroma_path
    elif index_mode == "full":
        collection = load_vector_collection(settings, embed_model)
        index_path = settings.chroma_path
    else:
        raise ValueError("index_mode must be 'full' or 'train_only'")

    eval_qas = load_eval_qas(holdout_split_path(settings))
    if limit is not None:
        eval_qas = eval_qas[:limit]

    retrieved_results = [
        retrieve(qa.question, collection, embed_model, top_k=settings.top_k)
        for qa in eval_qas
    ]

    metrics, failures = compute_retrieval_metrics(
        eval_qas,
        retrieved_results,
        top_k=settings.top_k,
    )

    payload = {
        "index_mode": index_mode,
        "split_manifest": split_manifest_path(settings),
        "holdout_path": holdout_split_path(settings),
        "train_index_path": index_path,
        "top_k": settings.top_k,
        "metrics": metrics.to_dict(),
        "failures": failures,
        "notes": (
            "Use index_mode='full' to measure retrieval on hold-out questions "
            "with all paragraphs indexed (standard RAG eval). "
            "Use index_mode='train_only' for strict document generalization "
            "where gold paragraphs are excluded from the index."
        ),
    }
    save_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on hold-out QAs.")
    parser.add_argument("--limit", type=int, default=None, help="Limit hold-out QAs.")
    parser.add_argument(
        "--index-mode",
        choices=["full", "train_only"],
        default="full",
        help="full = all paragraphs indexed; train_only = train paragraphs only.",
    )
    parser.add_argument(
        "--output",
        default="results/retrieval_eval.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    settings = get_settings(require_groq=False)
    setup_logging(settings.log_level)

    payload = run_retrieval_eval(
        settings=settings,
        limit=args.limit,
        output=args.output,
        index_mode=args.index_mode,
    )
    metrics = payload["metrics"]

    print("Retrieval evaluation complete")
    print(f"  Index mode: {payload['index_mode']}")
    print(f"  Questions evaluated: {metrics['count']}")
    print(f"  Hit@1:  {metrics['hit_at_1']:.2%}")
    print(f"  Hit@{settings.top_k}: {metrics['hit_at_k']:.2%}")
    print(f"  MRR:    {metrics['mrr']:.4f}")
    print(f"  Context Precision@{settings.top_k}: {metrics['context_precision_at_k']:.2%}")
    print(f"  Context Recall: {metrics['context_recall']:.2%}")
    print(f"  Failures logged: {len(payload['failures'])}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
