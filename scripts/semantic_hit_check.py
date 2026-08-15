import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
docs = {}
for path in (ROOT / "data/raw/train_data.json", ROOT / "data/splits/holdout.json"):
    raw = json.loads(path.read_text(encoding="utf-8"))
    for article in raw["data"]:
        for para in article.get("paragraphs", []):
            docs[str(para["document_id"])] = para["context"]

conv = json.loads((ROOT / "results/conversation_retrieval_eval.json").read_text(encoding="utf-8"))
scenarios = {sc["id"]: sc for sc in json.loads((ROOT / "data/eval/conversation_scenarios.json").read_text(encoding="utf-8"))}

print("=== CONV MODE: ID hit vs semantic hit ===")
id_hits = sem_hits = 0
scored = 0
failures_detail = []
for sc in conv["scenarios"]:
    sid = sc["id"]
    for turn in sc["turns"]:
        ti = turn["turn_index"]
        meta = scenarios[sid]["turns"][ti]
        gold = meta.get("gold_document_id", "")
        if not gold:
            continue
        scored += 1
        r = turn["conversational"]
        top_ids = [s["document_id"] for s in r.get("sources", [])]
        id_hit = r.get("retrieval_hit_at_1", False)
        gt = meta.get("ground_truth", "")
        top1 = docs.get(top_ids[0], "") if top_ids else ""
        gold_t = docs.get(gold, "")

        sem = id_hit
        if not sem:
            if gt and gt in top1:
                sem = True
            else:
                q = r.get("rewritten_query") or r["query"]
                q_words = [w for w in q.split() if len(w) > 3][:6]
                top_match = sum(1 for w in q_words if w in top1)
                gold_match = sum(1 for w in q_words if w in gold_t)
                if top_match > gold_match:
                    sem = True

        if id_hit:
            id_hits += 1
        if sem:
            sem_hits += 1
        if not id_hit:
            failures_detail.append({
                "scenario": sid,
                "turn": ti,
                "gold": gold,
                "top": top_ids[0] if top_ids else None,
                "gt": gt,
                "gt_in_top1": bool(gt and gt in top1),
                "semantic_correct": sem,
                "gold_label_wrong": sem and not id_hit,
            })

for f in failures_detail:
    print(f)

print(f"Scored: {scored}, ID Hit@1: {id_hits}/{scored}={id_hits/scored:.1%}")
print(f"Semantic Hit@1: {sem_hits}/{scored}={sem_hits/scored:.1%}")
print(f"Gold-label errors (semantic ok, ID miss): {sum(1 for f in failures_detail if f['gold_label_wrong'])}")

re = json.loads((ROOT / "results/retrieval_eval.json").read_text(encoding="utf-8"))
print("\n=== HOLDOUT failure rank distribution ===")
c = Counter()
for f in re["failures"]:
    r = f.get("first_correct_rank")
    c[r if r else "not_in_top3"] += 1
print(dict(c))

# Holdout: check top1 contains ground truth
print("\n=== HOLDOUT: ground truth in top1 despite ID miss ===")
gt_in_top1 = 0
for f in re["failures"]:
    if f.get("first_correct_rank") == 1:
        continue
    gt = (f.get("ground_truth") or "").strip()
    top1 = docs.get(f["retrieved_document_ids"][0], "") if f["retrieved_document_ids"] else ""
    if gt and gt in top1:
        gt_in_top1 += 1
print(f"Failures where GT substring in top1 (but wrong doc id): {gt_in_top1}")
