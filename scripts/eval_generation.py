"""Evaluate end-to-end RAG generation with an LLM judge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path
from google import genai
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.eval_utils import average_generation_scores, load_eval_qas, save_json
from src.logging_config import setup_logging
from src.pipeline import answer_question, load_vector_collection


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"completed": [], "results": []}


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def judge_answer(gemini_client, settings, question, context, answer, ground_truth) -> str:
    judge_prompt = f"""You are evaluating a RAG system's answer to an Amharic question.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}
Ground Truth Answer: {ground_truth}

Rate the following on a scale of 1-5 (5 = best):
1. Faithfulness: Is the generated answer fully supported by the retrieved context (no hallucination)?
2. Relevance: Does the generated answer actually address the question asked?
3. Correctness: Does the generated answer match the ground truth in meaning?
4. Amharic Fluency: Is the answer natural, grammatical, and clear in Amharic?

Respond ONLY in this exact format, no other text:
Faithfulness: <score>
Relevance: <score>
Correctness: <score>
Amharic Fluency: <score>
"""
    response = gemini_client.models.generate_content(
        model=settings.judge_model,
        contents=judge_prompt,
    )
    return response.text


def parse_scores(text: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for line in text.strip().split("\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            scores[key.strip()] = int(value.strip())
        except ValueError:
            continue
    return scores


def ensure_splits_exist(settings) -> None:
    train_path = Path(settings.splits_dir) / "train.json"
    holdout_path = Path(settings.splits_dir) / "holdout.json"
    if not train_path.exists() or not holdout_path.exists():
        from scripts.split_dataset import main as split_main

        print("Splits not found — creating train/hold-out splits...")
        split_main()


def run_generation_eval(
    *,
    settings,
    limit: int = 30,
    checkpoint: str = "results/generation_checkpoint.json",
    output: str = "results/generation_eval.json",
    sleep_seconds: int = 15,
) -> dict:
    ensure_splits_exist(settings)

    groq_client = Groq(api_key=settings.groq_api_key)
    gemini_client = genai.Client(api_key=settings.gemini_api_key)
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)

    eval_qas = load_eval_qas(holdout_split_path(settings))[:limit]
    checkpoint_path = Path(checkpoint)
    checkpoint_data = load_checkpoint(checkpoint_path)
    completed = set(checkpoint_data["completed"])

    for qa in eval_qas:
        if qa.question in completed:
            continue

        result = answer_question(
            qa.question,
            client=groq_client,
            embed_model=embed_model,
            collection=collection,
            settings=settings,
        )

        row = {
            "question": qa.question,
            "ground_truth": qa.ground_truth,
            "document_id": qa.document_id,
            "answer": result.answer,
            "sources": result.sources,
            "refusal": result.refusal,
            "error": result.error,
            "scores": {},
        }

        if result.error:
            print(f"Q: {qa.question}\n   Error: {result.error}\n")
        else:
            context = " ".join(source["text"] for source in result.sources)
            judge_text = judge_answer(
                gemini_client,
                settings,
                qa.question,
                context,
                result.answer,
                qa.ground_truth,
            )
            row["scores"] = parse_scores(judge_text)
            print(f"Q: {qa.question}")
            print(f"   Generated: {result.answer}")
            print(f"   Ground truth: {qa.ground_truth}")
            print(f"   Scores: {row['scores']}\n")

        checkpoint_data["completed"].append(qa.question)
        checkpoint_data["results"].append(row)
        save_checkpoint(checkpoint_path, checkpoint_data)

        if not result.error:
            time.sleep(sleep_seconds)

    averages = average_generation_scores(checkpoint_data["results"])
    payload = {
        "checkpoint": str(checkpoint_path),
        "questions_evaluated": len(checkpoint_data["results"]),
        "averages": averages,
        "results": checkpoint_data["results"],
    }
    save_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG generation on hold-out QAs.")
    parser.add_argument("--limit", type=int, default=30, help="Max questions to evaluate.")
    parser.add_argument(
        "--checkpoint",
        default="results/generation_checkpoint.json",
        help="Checkpoint path for resumable runs.",
    )
    parser.add_argument(
        "--output",
        default="results/generation_eval.json",
        help="Final output JSON path.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=15,
        help="Delay between judge calls to reduce rate limits.",
    )
    args = parser.parse_args()

    settings = get_settings(require_gemini=True)
    setup_logging(settings.log_level)

    payload = run_generation_eval(
        settings=settings,
        limit=args.limit,
        checkpoint=args.checkpoint,
        output=args.output,
        sleep_seconds=args.sleep_seconds,
    )

    print("=" * 50)
    print("Generation evaluation summary")
    print("=" * 50)
    for key, value in payload["averages"].items():
        print(f"{key}: {value:.2f}/5")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
