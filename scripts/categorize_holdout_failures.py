"""Detailed holdout retrieval failure categorization."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_corpus() -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in (ROOT / "data/raw/train_data.json", ROOT / "data/splits/holdout.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for article in raw["data"]:
            for para in article.get("paragraphs", []):
                docs[str(para["document_id"])] = para["context"]
    return docs


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def categorize(question: str, gold_id: str, retrieved: list[str], rank: int | None, docs: dict[str, str]) -> tuple[str, str]:
    gold = docs.get(gold_id, "")
    top1 = docs.get(retrieved[0], "") if retrieved else ""
    gt_words = set(re.findall(r"[\u1200-\u137F\w]+", question))

    if rank is not None and rank > 1:
        top_r = docs.get(retrieved[rank - 1], "")
        return "top_k_limitation", f"Gold at rank {rank}; distance gap likely small"

    # Gold not in top 3
    # Check if top1 contains distinctive entities from gold
    gold_entities = [w for w in gold.split() if len(w) > 4][:8]
    ent_in_top1 = sum(1 for e in gold_entities if e in top1)
    ent_in_gold = len(gold_entities)

    if ent_in_top1 >= 2 and SequenceMatcher(None, gold[:400], top1[:400]).ratio() > 0.35:
        return "similar_competing_documents", "Top1 shares entities/topic with gold but different paragraph"

    # Generic short questions
    generic_patterns = ("ማን ", "ስንት ", "የት ", "እንዴት ", "የትኛ", "ዕቅዱ", "ባለፉት")
    if any(p in question for p in generic_patterns) and len(question.split()) <= 10:
        return "query_formulation", "Short/generic query matches multiple corpus topics"

    # Long gold chunk - answer may need finer granularity
    if len(gold) > 700:
        return "chunk_granularity", f"Gold paragraph length {len(gold)} chars; query targets subset"

    # Morphology / variant: query uses different form than gold
    q_roots = {w[:4] for w in gt_words if len(w) >= 5}
    gold_roots = {w[:4] for w in gold.split() if len(w) >= 5}
    overlap = len(q_roots & gold_roots) / max(len(q_roots), 1)
    if overlap < 0.3:
        return "amharic_morphology_or_variant", f"Low query-gold token overlap ({overlap:.2f})"

    # Olympics / sports / specific named events often missed
    if any(k in question for k in ("ኦሎምፒክ", "ቴኳንዶ", "ወርልድ", "UN", "ተ.መ.ድ")):
        return "embedding_semantic_mismatch", "Event/entity-heavy query; embedding conflates sports/news"

    return "embedding_semantic_mismatch", "Gold not in top 3; no near-duplicate top1"


def main() -> None:
    docs = load_corpus()
    re_data = json.loads((ROOT / "results/retrieval_eval.json").read_text(encoding="utf-8"))
    failures = re_data["failures"]

    categories: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)

    for f in failures:
        cat, note = categorize(
            f["question"],
            f["gold_document_id"],
            f["retrieved_document_ids"],
            f.get("first_correct_rank"),
            docs,
        )
        categories[cat] += 1
        if len(examples[cat]) < 3:
            examples[cat].append({
                "question": f["question"],
                "gold_document_id": f["gold_document_id"],
                "retrieved_document_ids": f["retrieved_document_ids"][:3],
                "first_correct_rank": f.get("first_correct_rank"),
                "ground_truth": f.get("ground_truth"),
                "notes": note,
                "gold_snippet": docs.get(f["gold_document_id"], "")[:150],
                "top1_snippet": docs.get(f["retrieved_document_ids"][0], "")[:150] if f["retrieved_document_ids"] else "",
            })

    report = {
        "holdout_failures_total": len(failures),
        "holdout_hit_at_1": re_data["metrics"]["hit_at_1"],
        "category_counts": dict(categories),
        "category_percentages": {k: round(v / len(failures) * 100, 1) for k, v in categories.items()},
        "examples_by_category": dict(examples),
    }

    out = ROOT / "results/holdout_failure_categories.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["category_counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
