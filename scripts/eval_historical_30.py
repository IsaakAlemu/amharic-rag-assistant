"""Experiment 0: Reproduce historical 30-question eval and compare methodologies.

Historical source: archive/early_experiments/test_retrieval_accuracy.py
- Corpus: data/raw/train_data.json (all paragraphs → chroma_db)
- First 30 paragraphs with qas, qas[0] only
- Model: intfloat/multilingual-e5-small
- Query prefix: "query: ", passage prefix: "passage: "
- normalize_embeddings=True, Chroma cosine distance
- top_k=1 (n_results=1) in original script

Does NOT modify production retrieval code.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path
from sentence_transformers import SentenceTransformer

from src.document_loader import load_documents
from src.embedding_generator import generate_embeddings
from src.eval_utils import EvalQA, compute_retrieval_metrics, load_eval_qas, save_json
from src.retriever import retrieve

HISTORICAL_SCRIPT = "archive/early_experiments/test_retrieval_accuracy.py"
HISTORICAL_CORPUS = "data/raw/train_data.json"
HISTORICAL_MODEL = "intfloat/multilingual-e5-small"
HISTORICAL_TOP_K = 1


@dataclass
class HistoricalParagraph:
    document_id: str
    context: str
    qas: list[dict]
    selection_index: int  # 1-based order in original 30-paragraph sample


def load_historical_30_paragraphs(json_path: str | Path) -> list[HistoricalParagraph]:
    """Match test_retrieval_accuracy.load_sample_questions paragraph selection."""
    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    paragraphs: list[HistoricalParagraph] = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            if para.get("qas") and len(paragraphs) < 30:
                paragraphs.append(
                    HistoricalParagraph(
                        document_id=str(para["document_id"]),
                        context=para.get("context", ""),
                        qas=para["qas"],
                        selection_index=len(paragraphs) + 1,
                    )
                )
    return paragraphs


def load_qas_first_only(paragraphs: list[HistoricalParagraph]) -> list[EvalQA]:
    samples: list[EvalQA] = []
    for para in paragraphs:
        qa = para.qas[0]
        answers = qa.get("answers", [])
        if not answers:
            continue
        answer = answers[0]
        samples.append(
            EvalQA(
                question=qa["question"],
                ground_truth=answer.get("text", ""),
                document_id=para.document_id,
                context=para.context,
                answer_start=answer.get("answer_start"),
                answer_end=answer.get("answer_end"),
                is_impossible=bool(qa.get("is_impossible")),
                question_id=qa.get("id"),
            )
        )
    return samples


def load_qas_all(paragraphs: list[HistoricalParagraph]) -> list[EvalQA]:
    samples: list[EvalQA] = []
    for para in paragraphs:
        for qa in para.qas:
            if qa.get("is_impossible"):
                continue
            answers = qa.get("answers", [])
            if not answers:
                continue
            answer = answers[0]
            samples.append(
                EvalQA(
                    question=qa["question"],
                    ground_truth=answer.get("text", ""),
                    document_id=para.document_id,
                    context=para.context,
                    answer_start=answer.get("answer_start"),
                    answer_end=answer.get("answer_end"),
                    is_impossible=bool(qa.get("is_impossible")),
                    question_id=qa.get("id"),
                )
            )
    return samples


def run_retrieval_eval_qas(
    eval_qas: list[EvalQA],
    collection,
    embed_model,
    top_k: int,
) -> tuple[dict, list[dict[str, Any]], list[list[dict]]]:
    retrieved_results = [
        retrieve(qa.question, collection, embed_model, top_k=top_k) for qa in eval_qas
    ]
    metrics, failures = compute_retrieval_metrics(eval_qas, retrieved_results, top_k=top_k)

    rows = []
    for qa, retrieved in zip(eval_qas, retrieved_results):
        top_ids = [d["document_id"] for d in retrieved]
        rank = next((i + 1 for i, d in enumerate(retrieved) if d["document_id"] == qa.document_id), None)
        rows.append(
            {
                "question": qa.question,
                "gold_document_id": qa.document_id,
                "ground_truth": qa.ground_truth,
                "retrieved_document_ids": top_ids,
                "first_correct_rank": rank,
                "hit_at_1": rank == 1,
            }
        )
    return metrics.to_dict(), rows, failures


def reproduce_original_top1(
    eval_qas: list[EvalQA],
    collection,
    embed_model,
) -> dict[str, Any]:
    """Faithful to test_retrieval_accuracy.py: n_results=1, direct Chroma query."""
    hits = 0
    rows = []
    for qa in eval_qas:
        query_embedding = embed_model.encode(
            "query: " + qa.question,
            normalize_embeddings=True,
        )
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=1,
        )
        retrieved_id = results["metadatas"][0][0]["filename"]
        correct = str(retrieved_id) == str(qa.document_id)
        hits += int(correct)
        rows.append(
            {
                "question": qa.question,
                "gold_document_id": qa.document_id,
                "retrieved_document_id": str(retrieved_id),
                "hit_at_1": correct,
            }
        )
    count = len(eval_qas)
    return {
        "method": "direct_chroma_query_n_results_1",
        "source_script": HISTORICAL_SCRIPT,
        "count": count,
        "hits": hits,
        "hit_at_1": hits / count if count else 0.0,
        "rows": rows,
        "failures": [r for r in rows if not r["hit_at_1"]],
    }


def audit_holdout_failures(
    failures: list[dict],
    docs: dict[str, str],
    *,
    sample_size: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    sample = rng.sample(failures, min(sample_size, len(failures)))

    categories: dict[str, int] = {}
    audited = []

    for f in sample:
        gold_id = f["gold_document_id"]
        tops = f.get("retrieved_document_ids", [])
        top1_id = tops[0] if tops else None
        gt = (f.get("ground_truth") or "").strip()
        gold_text = docs.get(gold_id, "")
        top1_text = docs.get(top1_id, "") if top1_id else ""

        gt_in_gold = bool(gt and gt in gold_text)
        gt_in_top1 = bool(gt and gt in top1_text)

        if gt_in_top1 and not gt_in_gold:
            category = "possible_incorrect_gold_label"
        elif not gt_in_gold and not gt_in_top1:
            category = "ground_truth_not_in_gold_or_top1"
        elif gt_in_gold and gt_in_top1 and top1_id != gold_id:
            category = "both_contain_answer_different_id"
        elif gt_in_gold and not gt_in_top1:
            category = "genuine_retrieval_failure"
        else:
            category = "other"

        categories[category] = categories.get(category, 0) + 1
        audited.append(
            {
                "question": f["question"],
                "gold_document_id": gold_id,
                "retrieved_top1": top1_id,
                "ground_truth": gt,
                "category": category,
                "ground_truth_in_gold": gt_in_gold,
                "ground_truth_in_top1": gt_in_top1,
                "gold_snippet": gold_text[:200],
                "top1_snippet": top1_text[:200],
            }
        )

    return {
        "sample_size": len(audited),
        "seed": seed,
        "source_failures_total": len(failures),
        "category_counts": categories,
        "cases": audited,
    }


def load_corpus_texts() -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in (ROOT / "data/raw/train_data.json", ROOT / "data/splits/holdout.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for article in raw.get("data", []):
            for para in article.get("paragraphs", []):
                docs[str(para["document_id"])] = para.get("context", "")
    return docs


def run_experiment() -> dict[str, Any]:
    settings = get_settings(require_groq=False)
    embed_model = SentenceTransformer(HISTORICAL_MODEL)
    documents = load_documents(HISTORICAL_CORPUS)
    collection = generate_embeddings(documents, embed_model, persist_path=settings.chroma_path)

    paragraphs = load_historical_30_paragraphs(HISTORICAL_CORPUS)
    qas_first = load_qas_first_only(paragraphs)
    qas_all = load_qas_all(paragraphs)

    reproduction = reproduce_original_top1(qas_first, collection, embed_model)

    # All QAs from same 30 paragraphs — top_k=3 for Hit@3/MRR (same corpus & embedder)
    all_qas_metrics, all_qas_rows, all_qas_failures = run_retrieval_eval_qas(
        qas_all, collection, embed_model, top_k=settings.top_k
    )

    # Holdout 329 — same embedder/collection as current pipeline
    holdout_qas = load_eval_qas(holdout_split_path(settings))
    holdout_metrics, holdout_rows, holdout_failures = run_retrieval_eval_qas(
        holdout_qas, collection, embed_model, top_k=settings.top_k
    )

    docs = load_corpus_texts()
    audit = audit_holdout_failures(holdout_failures, docs, sample_size=20, seed=42)

    paragraph_manifest = [
        {
            "selection_index": p.selection_index,
            "document_id": p.document_id,
            "qas_count": len(p.qas),
            "first_question": p.qas[0]["question"] if p.qas else "",
        }
        for p in paragraphs
    ]

    payload_a = {
        "label": "A_historical_30_qas0_top1",
        "description": "Reproduction of test_retrieval_accuracy.py",
        "methodology": {
            "source_script": HISTORICAL_SCRIPT,
            "corpus": HISTORICAL_CORPUS,
            "chroma_path": settings.chroma_path,
            "chunk_count": collection.count(),
            "embed_model": HISTORICAL_MODEL,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "normalize_embeddings": True,
            "similarity": "cosine (Chroma default on normalized vectors)",
            "top_k": HISTORICAL_TOP_K,
            "question_selection": "first 30 paragraphs with qas in train_data.json order; qas[0] only",
            "document_ids": [p.document_id for p in paragraphs],
        },
        "reproduction": reproduction,
    }

    payload_b = {
        "label": "B_same_30_paragraphs_all_qas",
        "description": "All questions from the same 30 paragraphs as A",
        "methodology": {
            "same_paragraphs_as_A": True,
            "document_ids": [p.document_id for p in paragraphs],
            "question_count": len(qas_all),
            "top_k": settings.top_k,
            "note": "Uses src.retriever.retrieve with top_k=3 for Hit@3/MRR; corpus identical to A",
        },
        "metrics": all_qas_metrics,
        "failures_count": len(all_qas_failures),
        "failures": all_qas_failures,
        "per_question": all_qas_rows,
    }

    payload_c = {
        "label": "C_holdout_329",
        "description": "Current holdout evaluation on same chroma_db index",
        "methodology": {
            "holdout_path": holdout_split_path(settings),
            "question_selection": "all non-impossible qas from holdout.json split (document-level holdout)",
            "top_k": settings.top_k,
            "note": "Questions from held-out document IDs; gold paragraphs ARE in chroma_db (full index mode)",
        },
        "metrics": holdout_metrics,
        "failures_count": len(holdout_failures),
    }

    comparison = {
        "A_historical_qas0_hit_at_1": reproduction["hit_at_1"],
        "A_historical_qas0_hits": f"{reproduction['hits']}/{reproduction['count']}",
        "B_all_qas_hit_at_1": all_qas_metrics["hit_at_1"],
        "B_all_qas_hit_at_3": all_qas_metrics["hit_at_3"],
        "B_all_qas_mrr": all_qas_metrics["mrr"],
        "B_all_qas_count": all_qas_metrics["count"],
        "C_holdout_hit_at_1": holdout_metrics["hit_at_1"],
        "C_holdout_hit_at_3": holdout_metrics["hit_at_3"],
        "C_holdout_mrr": holdout_metrics["mrr"],
        "C_holdout_count": holdout_metrics["count"],
    }

    return {
        "experiment": "historical_30_methodology_investigation",
        "paragraph_manifest": paragraph_manifest,
        "A": payload_a,
        "B": payload_b,
        "C": payload_c,
        "comparison": comparison,
        "holdout_failure_audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 0: historical 30 vs holdout 329.")
    parser.add_argument(
        "--output-reproduction",
        default="results/historical_30_reproduction.json",
    )
    parser.add_argument(
        "--output-all-qas",
        default="results/historical_30_all_qas.json",
    )
    parser.add_argument(
        "--output-audit",
        default="results/holdout_failure_audit.json",
    )
    args = parser.parse_args()

    result = run_experiment()

    save_json(args.output_reproduction, {"experiment": result["experiment"], **result["A"], "paragraph_manifest": result["paragraph_manifest"], "comparison": result["comparison"]})
    save_json(args.output_all_qas, {"experiment": result["experiment"], **result["B"], "comparison": result["comparison"]})
    save_json(args.output_audit, result["holdout_failure_audit"])

    combined_path = ROOT / "results/historical_30_investigation.json"
    save_json(str(combined_path), result)

    rep = result["A"]["reproduction"]
    b = result["B"]["metrics"]
    c = result["C"]["metrics"]
    print("=" * 60)
    print("Experiment 0 — Historical vs Holdout Methodology")
    print("=" * 60)
    print(f"A) Historical qas[0], top_k=1: {rep['hits']}/{rep['count']} = {rep['hit_at_1']:.1%}")
    print(f"B) All QAs from same 30 paras (n={b['count']}):")
    print(f"   Hit@1={b['hit_at_1']:.1%}  Hit@3={b['hit_at_3']:.1%}  MRR={b['mrr']:.4f}")
    print(f"C) Holdout 329 (n={c['count']}):")
    print(f"   Hit@1={c['hit_at_1']:.1%}  Hit@3={c['hit_at_3']:.1%}  MRR={c['mrr']:.4f}")
    print(f"\nSaved: {args.output_reproduction}")
    print(f"Saved: {args.output_all_qas}")
    print(f"Saved: {args.output_audit}")
    print(f"Saved: {combined_path}")


if __name__ == "__main__":
    main()
