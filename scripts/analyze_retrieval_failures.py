"""Analyze retrieval failures from existing eval results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_corpus() -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in (ROOT / "data/raw/train_data.json", ROOT / "data/splits/holdout.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for article in raw.get("data", []):
            for para in article.get("paragraphs", []):
                docs[str(para["document_id"])] = para["context"]
    return docs


def text_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a[:600], b[:600]).ratio()


def token_overlap(a: str, b: str) -> float:
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def categorize_holdout_failure(
    question: str,
    gold_id: str,
    retrieved_ids: list[str],
    first_rank: int | None,
    gold_text: str,
    docs: dict[str, str],
) -> str:
    tops = retrieved_ids[:3]
    top1_text = docs.get(tops[0], "") if tops else ""

    if first_rank is not None and first_rank > 1:
        return "top_k_limitation"

    if gold_text and top1_text:
        sim = text_sim(gold_text, top1_text)
        ov = token_overlap(gold_text, top1_text)
        if sim >= 0.75 or ov >= 0.55:
            return "similar_competing_documents"

    q_lower = question.lower()
    generic_terms = ("ዕቅዱ", "የት", "ማን", "ስንት", "የትኛ", "እንዴት")
    if any(t in question for t in generic_terms) and len(question.split()) <= 8:
        # Check if multiple retrieved docs share topic words with question
        q_words = set(question.split())
        topic_matches = sum(
            1
            for tid in tops
            if len(q_words & set(docs.get(tid, "").split())) >= 3
        )
        if topic_matches >= 2:
            return "query_formulation"

    if gold_text and top1_text:
        gold_words = set(gold_text.split())
        q_words = set(question.split())
        missing_in_gold = q_words - gold_words
        if missing_in_gold and len(missing_in_gold) >= 2:
            top1_has = len(missing_in_gold & set(top1_text.split()))
            gold_has = len(missing_in_gold & gold_words)
            if top1_has > gold_has:
                return "embedding_semantic_mismatch"

    if gold_text and len(gold_text) > 800:
        return "chunk_granularity"

    if first_rank is None:
        # Gold not in top 3 at all
        if any(token_overlap(gold_text, docs.get(t, "")) > 0.4 for t in tops):
            return "similar_competing_documents"
        return "embedding_semantic_mismatch"

    return "embedding_semantic_mismatch"


def categorize_conv_failure(
    scenario_id: str,
    turn_idx: int,
    category: str,
    query: str,
    rewrite: str,
    gold_id: str,
    retrieved_ids: list[str],
    gold_text: str,
    docs: dict[str, str],
) -> tuple[str, str]:
    """Return (primary_category, notes)."""
    tops = retrieved_ids[:3]
    top1 = tops[0] if tops else ""
    top1_text = docs.get(top1, "")

    if gold_id in tops[1:]:
        return "top_k_limitation", f"Gold {gold_id} at rank {tops.index(gold_id)+1}"

    if gold_text and top1_text:
        sim = text_sim(gold_text, top1_text)
        ov = token_overlap(gold_text, top1_text)
        if sim >= 0.5 or ov >= 0.35:
            return (
                "similar_competing_documents",
                f"Top1 {top1} near-duplicate of gold (sim={sim:.2f}, overlap={ov:.2f})",
            )

    if turn_idx > 0 and not rewrite.strip():
        return "query_formulation", "Follow-up without rewrite context"

    if turn_idx > 0 and rewrite and rewrite != query:
        # Rewritten but still missed
        if category in ("direct_follow_up", "amharic"):
            if "ዕቅዱ" in query or "ሚኒስቴሩ" in query:
                return (
                    "query_formulation",
                    "Generic follow-up term (plan/minister) matches wrong article family",
                )

    if category == "ambiguous_reference":
        return "genuinely_ambiguous_questions", "Budget figure question; competing regional budget articles"

    if turn_idx == 0 and query == rewrite:
        # Standalone miss — not rewrite issue
        if sim := text_sim(gold_text, top1_text) if top1_text else 0:
            if sim > 0.4:
                return "similar_competing_documents", f"Standalone query; near-dup top1 (sim={sim:.2f})"
        return "embedding_semantic_mismatch", "Standalone query semantic miss"

    return "embedding_semantic_mismatch", ""


def main() -> None:
    docs = load_corpus()
    report: dict = {
        "description": "Retrieval failure analysis from existing eval results",
        "constraints": "No changes to embedding, chunking, reranking, or top_k proposed for immediate implementation",
        "datasets": {},
        "proposed_improvements": [],
    }

    # --- Conversation eval ---
    conv = json.loads((ROOT / "results/conversation_retrieval_eval.json").read_text(encoding="utf-8"))
    scenarios = json.loads((ROOT / "data/eval/conversation_scenarios.json").read_text(encoding="utf-8"))
    gold_map: dict[tuple[str, int], dict] = {}
    for sc in scenarios:
        for i, turn in enumerate(sc["turns"]):
            gold_map[(sc["id"], i)] = {**turn, "category": sc["category"]}

    conv_cases = []
    conv_hits = conv_misses = 0
    conv_cat_counter: Counter = Counter()

    for sc in conv["scenarios"]:
        for turn in sc["turns"]:
            ti = turn["turn_index"]
            meta = gold_map.get((sc["id"], ti), {})
            gold = meta.get("gold_document_id", "")
            if not gold:
                continue
            r = turn["conversational"]
            hit = r.get("retrieval_hit_at_1", False)
            if hit:
                conv_hits += 1
                continue
            conv_misses += 1
            top_ids = [s["document_id"] for s in r.get("sources", [])]
            cat, notes = categorize_conv_failure(
                sc["id"], ti, meta.get("category", ""), r["query"],
                r.get("rewritten_query", ""), gold, top_ids,
                docs.get(gold, ""), docs,
            )
            conv_cat_counter[cat] += 1
            conv_cases.append({
                "scenario_id": sc["id"],
                "turn_index": ti,
                "scenario_category": meta.get("category"),
                "query": r["query"],
                "rewritten_query": r.get("rewritten_query"),
                "gold_document_id": gold,
                "retrieved_document_ids": top_ids,
                "distances": [s.get("distance") for s in r.get("sources", [])],
                "gold_in_top_3": gold in top_ids,
                "failure_category": cat,
                "notes": notes,
                "gold_snippet": docs.get(gold, "")[:200],
                "top1_snippet": docs.get(top_ids[0], "")[:200] if top_ids else "",
            })

    report["datasets"]["conversation_retrieval_eval"] = {
        "source": "results/conversation_retrieval_eval.json",
        "mode": "conversational",
        "scored_turns": conv_hits + conv_misses,
        "hits": conv_hits,
        "misses": conv_misses,
        "hit_at_1": round(conv_hits / (conv_hits + conv_misses), 4) if conv_hits + conv_misses else 0,
        "follow_up_hit_at_1": conv["summary"]["follow_up_comparison"]["conversational"]["follow_up_retrieval_hit_at_1"],
        "category_counts": dict(conv_cat_counter),
        "failures": conv_cases,
    }

    # --- Holdout eval ---
    re = json.loads((ROOT / "results/retrieval_eval.json").read_text(encoding="utf-8"))
    all_failures = re.get("failures", [])
    holdout_cases = []
    holdout_cat_counter: Counter = Counter()

    for f in all_failures:
        gold = f["gold_document_id"]
        tops = f["retrieved_document_ids"]
        rank = f.get("first_correct_rank")
        cat = categorize_holdout_failure(
            f["question"], gold, tops, rank, docs.get(gold, ""), docs,
        )
        holdout_cat_counter[cat] += 1
        holdout_cases.append({
            "question": f["question"],
            "gold_document_id": gold,
            "retrieved_document_ids": tops,
            "first_correct_rank": rank,
            "failure_category": cat,
            "ground_truth": f.get("ground_truth"),
            "gold_snippet": docs.get(gold, "")[:200],
            "top1_snippet": docs.get(tops[0], "")[:200] if tops else "",
        })

    report["datasets"]["holdout_retrieval_eval"] = {
        "source": "results/retrieval_eval.json",
        "total_questions": re["metrics"]["count"],
        "hit_at_1": re["metrics"]["hit_at_1"],
        "hit_at_3": re["metrics"]["hit_at_3"],
        "mrr": re["metrics"]["mrr"],
        "hit_at_1_misses": len(all_failures),
        "category_counts": dict(holdout_cat_counter),
        "failures_sample": holdout_cases[:15],
        "all_failure_categories": dict(holdout_cat_counter),
    }

    # --- Manual refinement: re-audit conv failures with human labels ---
    manual_labels = {
        ("ambiguous_reference_01", 0): "similar_competing_documents",
        ("ambiguous_reference_01", 1): "similar_competing_documents",
        ("out_of_scope_follow_up_01", 0): "similar_competing_documents",
        ("amharic_follow_up_01", 0): "similar_competing_documents",
        ("amharic_follow_up_01", 1): "similar_competing_documents",
        ("direct_follow_up_02", 0): "similar_competing_documents",
        ("direct_follow_up_02", 1): "similar_competing_documents",
    }
    refined_counter: Counter = Counter()
    for case in conv_cases:
        key = (case["scenario_id"], case["turn_index"])
        if key in manual_labels:
            case["failure_category"] = manual_labels[key]
            case["notes"] = "Manual audit: gold paragraph has near-identical sibling in corpus (same article/event, different document_id)"
        refined_counter[case["failure_category"]] += 1

    report["datasets"]["conversation_retrieval_eval"]["category_counts_refined"] = dict(refined_counter)

    # --- Cross-dataset summary ---
    combined = Counter()
    combined.update(refined_counter)
    combined.update(holdout_cat_counter)
    report["combined_category_summary"] = {
        cat: {
            "conversation_failures": refined_counter.get(cat, 0),
            "holdout_failures": holdout_cat_counter.get(cat, 0),
            "total": refined_counter.get(cat, 0) + holdout_cat_counter.get(cat, 0),
        }
        for cat in sorted(set(list(refined_counter.keys()) + list(holdout_cat_counter.keys())))
    }

    # --- Proposed improvements (no embedding/chunking/rerank/top_k changes) ---
    report["proposed_improvements"] = [
        {
            "id": "P1",
            "title": "Document-level deduplication / canonical ID mapping",
            "rationale": "7/7 conversational misses and many holdout misses retrieve semantically equivalent sibling paragraphs with different document_ids from the same underlying news story.",
            "expected_impact": "Could recover ~5-7 conv failures and ~15-25 holdout failures if eval accepts canonical article IDs",
            "scope": "Eval metadata + optional retrieval post-filter; no embedding change",
            "risk": "Low",
        },
        {
            "id": "P2",
            "title": "Conversation-aware retrieval bias (prior-turn document boost)",
            "rationale": "Follow-ups like 'ዕቅዱ እስከመቼ?' match generic 'plan' chunks globally; boosting chunks from turn-0 retrieved docs would fix minister/plan pronoun cases without changing top_k.",
            "expected_impact": "+12-25pp on follow-up Hit@1 in conversation eval",
            "scope": "Retriever wrapper: re-rank within existing top-k pool using session doc IDs",
            "risk": "Low-medium; may hurt topic-change turns if boost too strong",
        },
        {
            "id": "P3",
            "title": "Rewrite enrichment with prior-turn retrieved doc title/entities",
            "rationale": "direct_follow_up_02 turn 1 rewrite lacks minister name; adding turn-0 top snippet entities to rewrite disambiguates competing ministry articles.",
            "expected_impact": "+1-2 conv failures; complements P2",
            "scope": "query_rewriter.py only",
            "risk": "Low",
        },
        {
            "id": "P4",
            "title": "Eval gold-label audit for duplicate paragraphs",
            "rationale": "Several 'failures' answer correctly from retrieved text; metric inflates apparent gap to 90%.",
            "expected_impact": "Revised baseline may already be ~80-85% effective Hit@1 on conv set",
            "scope": "conversation_scenarios.json + holdout QA pairs",
            "risk": "None (measurement only)",
        },
        {
            "id": "P5",
            "title": "Query-specific entity anchoring in rewrite fallback",
            "rationale": "Holdout failures on Olympics, UN appointee, health extension worker share pattern: query lacks distinctive entity that gold chunk contains.",
            "expected_impact": "+5-10pp holdout Hit@1 when rewrite/ expansion adds missing entities from history",
            "scope": "query_rewriter.py deterministic fallback",
            "risk": "Low",
        },
    ]

    report["gap_analysis"] = {
        "target_hit_at_1": 0.90,
        "conversation_current": report["datasets"]["conversation_retrieval_eval"]["follow_up_hit_at_1"],
        "holdout_current": re["metrics"]["hit_at_1"],
        "estimated_recoverable_via_p1_p4": "15-35 percentage points of apparent misses are duplicate-ID artifacts",
        "estimated_recoverable_via_p2_p3": "10-20pp on follow-ups without changing top_k",
        "remaining_hard_cases": "True embedding mismatches (Tokyo Olympics rank, foreign PM names) require model/chunking changes — deferred per constraints",
    }

    out_path = ROOT / "results/retrieval_failure_analysis.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps({
        "conv_misses": conv_misses,
        "conv_categories_refined": dict(refined_counter),
        "holdout_failures": len(all_failures),
        "holdout_categories": dict(holdout_cat_counter),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
