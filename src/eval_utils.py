"""Shared utilities for dataset splitting and evaluation metrics."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.errors import RetrievalError


@dataclass(frozen=True)
class EvalQA:
    question: str
    ground_truth: str
    document_id: str
    context: str
    answer_start: int | None = None
    answer_end: int | None = None
    is_impossible: bool = False
    question_id: int | None = None


@dataclass
class RetrievalMetrics:
    count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_k: float
    mrr: float
    context_precision_at_k: float
    context_recall: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_amqa_raw(json_path: str | Path) -> dict[str, Any]:
    path = Path(json_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise RetrievalError(f"Data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RetrievalError(f"Invalid JSON in data file: {path}") from exc

    if "data" not in raw:
        raise RetrievalError(f"Data file missing required 'data' field: {path}")
    return raw


def iter_paragraphs(raw: dict[str, Any]) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    for article in raw["data"]:
        paragraphs.extend(article.get("paragraphs", []))
    return paragraphs


def paragraphs_to_amqa(paragraphs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"data": [{"paragraphs": [paragraph]} for paragraph in paragraphs]}


def split_paragraphs_by_document_id(
    paragraphs: list[dict[str, Any]],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    doc_ids = sorted({paragraph["document_id"] for paragraph in paragraphs})
    rng = random.Random(seed)
    shuffled = doc_ids.copy()
    rng.shuffle(shuffled)

    holdout_count = max(1, round(len(shuffled) * holdout_ratio))
    holdout_ids = set(shuffled[:holdout_count])
    train_ids = set(shuffled[holdout_count:])

    train_paragraphs = [p for p in paragraphs if p["document_id"] in train_ids]
    holdout_paragraphs = [p for p in paragraphs if p["document_id"] in holdout_ids]

    manifest = {
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "total_documents": len(doc_ids),
        "train_documents": len(train_ids),
        "holdout_documents": len(holdout_ids),
        "train_document_ids": sorted(train_ids),
        "holdout_document_ids": sorted(holdout_ids),
    }
    return train_paragraphs, holdout_paragraphs, manifest


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_eval_qas(json_path: str | Path, *, include_impossible: bool = False) -> list[EvalQA]:
    raw = load_amqa_raw(json_path)
    samples: list[EvalQA] = []

    for paragraph in iter_paragraphs(raw):
        context = paragraph.get("context", "")
        document_id = str(paragraph.get("document_id", ""))
        for qa in paragraph.get("qas", []):
            if qa.get("is_impossible") and not include_impossible:
                continue
            answers = qa.get("answers", [])
            if not answers:
                continue
            answer = answers[0]
            samples.append(
                EvalQA(
                    question=qa["question"],
                    ground_truth=answer.get("text", ""),
                    document_id=document_id,
                    context=context,
                    answer_start=answer.get("answer_start"),
                    answer_end=answer.get("answer_end"),
                    is_impossible=bool(qa.get("is_impossible")),
                    question_id=qa.get("id"),
                )
            )
    return samples


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def answer_in_text(qa: EvalQA, text: str) -> bool:
    if qa.answer_start is not None and qa.answer_end is not None:
        start = qa.answer_start
        end = qa.answer_end
        if 0 <= start < end <= len(qa.context):
            span = qa.context[start:end]
            if span and span in text:
                return True

    ground_truth = normalize_text(qa.ground_truth)
    if not ground_truth:
        return False
    return ground_truth in normalize_text(text)


def compute_retrieval_metrics(
    eval_qas: list[EvalQA],
    retrieved_results: list[list[dict[str, Any]]],
    *,
    top_k: int,
) -> tuple[RetrievalMetrics, list[dict[str, Any]]]:
    if len(eval_qas) != len(retrieved_results):
        raise ValueError("eval_qas and retrieved_results must have the same length")

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    failures: list[dict[str, Any]] = []

    for qa, retrieved in zip(eval_qas, retrieved_results):
        retrieved_ids = [doc["document_id"] for doc in retrieved]
        first_correct_rank = None
        for rank, doc in enumerate(retrieved, start=1):
            if doc["document_id"] == qa.document_id:
                first_correct_rank = rank
                break

        if first_correct_rank == 1:
            hits_at_1 += 1
        if first_correct_rank is not None and first_correct_rank <= 3:
            hits_at_3 += 1
        if first_correct_rank is not None and first_correct_rank <= top_k:
            hits_at_k += 1

        if first_correct_rank is not None:
            reciprocal_ranks.append(1.0 / first_correct_rank)
        else:
            reciprocal_ranks.append(0.0)

        relevant_docs = sum(1 for doc in retrieved if answer_in_text(qa, doc["text"]))
        precision_scores.append(relevant_docs / len(retrieved) if retrieved else 0.0)
        recall_scores.append(
            1.0 if any(answer_in_text(qa, doc["text"]) for doc in retrieved) else 0.0
        )

        if first_correct_rank != 1:
            failures.append(
                {
                    "question": qa.question,
                    "gold_document_id": qa.document_id,
                    "retrieved_document_ids": retrieved_ids,
                    "first_correct_rank": first_correct_rank,
                    "ground_truth": qa.ground_truth,
                }
            )

    count = len(eval_qas)
    return (
        RetrievalMetrics(
            count=count,
            hit_at_1=hits_at_1 / count if count else 0.0,
            hit_at_3=hits_at_3 / count if count else 0.0,
            hit_at_k=hits_at_k / count if count else 0.0,
            mrr=sum(reciprocal_ranks) / count if count else 0.0,
            context_precision_at_k=sum(precision_scores) / count if count else 0.0,
            context_recall=sum(recall_scores) / count if count else 0.0,
        ),
        failures,
    )


def average_generation_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["Faithfulness", "Relevance", "Correctness", "Amharic Fluency"]
    averages: dict[str, float] = {}
    for key in keys:
        values = [row["scores"].get(key) for row in results if row.get("scores")]
        values = [value for value in values if isinstance(value, int | float)]
        if values:
            averages[key] = sum(values) / len(values)
    return averages
