"""Measure baseline prompt/token breakdown before any optimization."""

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

from src.eval_utils import load_eval_qas, save_json
from src.logging_config import setup_logging
from src.pipeline import load_vector_collection
from src.prompt_builder import SYSTEM_RULES, build_prompt_parts
from src.retriever import retrieve
from src.token_counter import TokenCounter


def summarize(values: list[int | float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 1),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def run_token_eval(
    *,
    settings,
    sample_size: int = 50,
    output: str = "results/token_baseline.json",
) -> dict:
    token_counter = TokenCounter()
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)

    eval_qas = load_eval_qas(holdout_split_path(settings))[:sample_size]

    system_tokens = []
    context_tokens = []
    query_tokens = []
    total_tokens = []
    per_doc_tokens = []
    rows = []

    for qa in eval_qas:
        retrieved = retrieve(qa.question, collection, embed_model, top_k=settings.top_k)
        parts = build_prompt_parts(qa.question, retrieved, system_rules=SYSTEM_RULES)

        sys_count = token_counter.count(parts["system_rules"])
        ctx_count = token_counter.count(parts["context"])
        q_count = token_counter.count(parts["question_suffix"])
        full_count = token_counter.count(parts["full"])

        system_tokens.append(sys_count)
        context_tokens.append(ctx_count)
        query_tokens.append(q_count)
        total_tokens.append(full_count)
        per_doc_tokens.extend(token_counter.count(doc["text"]) for doc in retrieved)

        rows.append(
            {
                "question": qa.question,
                "document_id": qa.document_id,
                "retrieved_count": len(retrieved),
                "tokens": {
                    "system_rules": sys_count,
                    "context": ctx_count,
                    "question_suffix": q_count,
                    "total_prompt": full_count,
                    "per_doc": [
                        {
                            "rank": doc["rank"],
                            "document_id": doc["document_id"],
                            "tokens": token_counter.count(doc["text"]),
                            "chars": len(doc["text"]),
                        }
                        for doc in retrieved
                    ],
                },
            }
        )

    payload = {
        "description": "Baseline prompt token breakdown with no context compression.",
        "tokenizer": "meta-llama/Meta-Llama-3-8B-Instruct (fallback: chars/3)",
        "top_k": settings.top_k,
        "sample_size": len(rows),
        "summary": {
            "system_rules": summarize(system_tokens),
            "context": summarize(context_tokens),
            "question_suffix": summarize(query_tokens),
            "total_prompt": summarize(total_tokens),
            "per_retrieved_doc": summarize(per_doc_tokens),
        },
        "notes": {
            "groq_8b_tpm_limit": 6000,
            "comparison": "Compare total_prompt.p95 against model TPM limits before choosing 8B.",
        },
        "samples": rows,
    }
    save_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure baseline prompt token usage.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--output", default="results/token_baseline.json")
    args = parser.parse_args()

    settings = get_settings(require_groq=False)
    setup_logging(settings.log_level)

    payload = run_token_eval(
        settings=settings,
        sample_size=args.sample_size,
        output=args.output,
    )
    summary = payload["summary"]

    print("Baseline token measurement complete")
    print(f"  Samples: {payload['sample_size']}")
    print(f"  System rules  mean/p95/max: {summary['system_rules']['mean']}/"
          f"{summary['system_rules']['p95']}/{summary['system_rules']['max']}")
    print(f"  Context       mean/p95/max: {summary['context']['mean']}/"
          f"{summary['context']['p95']}/{summary['context']['max']}")
    print(f"  Question      mean/p95/max: {summary['question_suffix']['mean']}/"
          f"{summary['question_suffix']['p95']}/{summary['question_suffix']['max']}")
    print(f"  TOTAL PROMPT  mean/p95/max: {summary['total_prompt']['mean']}/"
          f"{summary['total_prompt']['p95']}/{summary['total_prompt']['max']}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
