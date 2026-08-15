"""Evaluate refusal behavior on out-of-scope questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.eval_utils import save_json
from src.llm import REFUSAL_PHRASE
from src.logging_config import setup_logging
from src.pipeline import answer_question, load_vector_collection


def load_out_of_scope_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_out_of_scope_eval(
    *,
    settings,
    input_path: str = "data/eval/out_of_scope_questions.json",
    output: str = "results/out_of_scope_eval.json",
) -> dict:
    groq_client = Groq(api_key=settings.groq_api_key)
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)

    questions = load_out_of_scope_questions(Path(input_path))
    results = []
    correct_refusals = 0

    for item in questions:
        question = item["question"]
        result = answer_question(
            question,
            client=groq_client,
            embed_model=embed_model,
            collection=collection,
            settings=settings,
        )

        refused = (
            result.refusal
            or REFUSAL_PHRASE in result.answer
            or result.skipped_generation
        )
        if refused:
            correct_refusals += 1

        row = {
            "question": question,
            "category": item.get("category", "unknown"),
            "answer": result.answer,
            "refusal": refused,
            "error": result.error,
        }
        results.append(row)

        status = "PASS" if refused else "FAIL"
        print(f"[{status}] {question}")
        if not refused:
            print(f"        Answer: {result.answer}")

    accuracy = correct_refusals / len(results) if results else 0.0
    payload = {
        "total": len(results),
        "correct_refusals": correct_refusals,
        "refusal_accuracy": accuracy,
        "results": results,
    }
    save_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate out-of-scope refusal behavior.")
    parser.add_argument(
        "--input",
        default="data/eval/out_of_scope_questions.json",
        help="JSON file with out-of-scope questions.",
    )
    parser.add_argument(
        "--output",
        default="results/out_of_scope_eval.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    payload = run_out_of_scope_eval(
        settings=settings,
        input_path=args.input,
        output=args.output,
    )

    print("=" * 50)
    print(
        "Refusal accuracy: "
        f"{payload['refusal_accuracy']:.2%} "
        f"({payload['correct_refusals']}/{payload['total']})"
    )
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
