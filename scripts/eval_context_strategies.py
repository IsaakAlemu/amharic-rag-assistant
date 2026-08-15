"""Measure how context strategies affect answer-preserving context and token usage."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path
from sentence_transformers import SentenceTransformer

from src.context_manager import STRATEGY_BUDGETS, apply_context_strategy
from src.eval_utils import answer_in_text, load_eval_qas, save_json
from src.logging_config import setup_logging
from src.pipeline import load_vector_collection
from src.retriever import retrieve
from src.token_counter import TokenCounter

STRATEGIES = [
    "baseline",
    "compressed_system",
    "budget_6000",
    "budget_4500",
    "budget_3500",
]


def run_context_strategy_eval(
    *,
    settings,
    sample_size: int = 50,
    output: str = "results/context_strategy_eval.json",
) -> dict:
    token_counter = TokenCounter()
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    eval_qas = load_eval_qas(holdout_split_path(settings))[:sample_size]

    strategy_results: dict[str, dict] = {}

    for strategy in STRATEGIES:
        answer_preserved = 0
        gold_doc_preserved = 0
        rank1_preserved = 0
        prompt_tokens = []
        dropped_docs = 0
        truncated_docs = 0
        rows = []

        for qa in eval_qas:
            retrieved = retrieve(qa.question, collection, embed_model, top_k=settings.top_k)
            prepared = apply_context_strategy(
                qa.question,
                retrieved,
                strategy=strategy,
                token_counter=token_counter,
            )

            preserved = any(answer_in_text(qa, doc["text"]) for doc in prepared.docs)
            if preserved:
                answer_preserved += 1

            if any(doc["document_id"] == qa.document_id for doc in prepared.docs):
                gold_doc_preserved += 1

            if prepared.docs and prepared.docs[0]["document_id"] == qa.document_id:
                rank1_preserved += 1

            prompt_tokens.append(prepared.decision.prompt_tokens)
            dropped_docs += len(prepared.decision.dropped_docs)
            truncated_docs += len(prepared.decision.truncated_docs)

            rows.append(
                {
                    "question": qa.question,
                    "gold_document_id": qa.document_id,
                    "answer_preserved": preserved,
                    "gold_doc_in_prompt": any(
                        doc["document_id"] == qa.document_id for doc in prepared.docs
                    ),
                    "prompt_tokens": prepared.decision.prompt_tokens,
                    "final_doc_count": prepared.decision.final_doc_count,
                    "dropped_docs": prepared.decision.dropped_docs,
                    "truncated_docs": prepared.decision.truncated_docs,
                }
            )

        count = len(eval_qas)
        strategy_results[strategy] = {
            "budget": STRATEGY_BUDGETS[strategy],
            "answer_preservation_rate": answer_preserved / count if count else 0.0,
            "gold_document_preservation_rate": gold_doc_preserved / count if count else 0.0,
            "rank1_gold_rate": rank1_preserved / count if count else 0.0,
            "prompt_tokens": {
                "mean": round(statistics.mean(prompt_tokens), 1) if prompt_tokens else 0,
                "p95": sorted(prompt_tokens)[max(0, int(len(prompt_tokens) * 0.95) - 1)]
                if prompt_tokens
                else 0,
                "max": max(prompt_tokens) if prompt_tokens else 0,
            },
            "avg_dropped_docs": round(dropped_docs / count, 2) if count else 0,
            "avg_truncated_docs": round(truncated_docs / count, 2) if count else 0,
            "samples": rows,
        }

    payload = {
        "description": (
            "Compare context strategies by measuring whether gold answers remain "
            "in the prompt context and how many tokens are used."
        ),
        "sample_size": len(eval_qas),
        "strategies": strategy_results,
    }
    save_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate context compression strategies.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--output", default="results/context_strategy_eval.json")
    args = parser.parse_args()

    settings = get_settings(require_groq=False)
    setup_logging(settings.log_level)

    payload = run_context_strategy_eval(
        settings=settings,
        sample_size=args.sample_size,
        output=args.output,
    )

    print("Context strategy evaluation complete")
    for strategy, result in payload["strategies"].items():
        print(
            f"  {strategy:18s} "
            f"tokens p95={result['prompt_tokens']['p95']:4d}  "
            f"answer_preserved={result['answer_preservation_rate']:.1%}  "
            f"gold_doc_preserved={result['gold_document_preservation_rate']:.1%}"
        )
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
