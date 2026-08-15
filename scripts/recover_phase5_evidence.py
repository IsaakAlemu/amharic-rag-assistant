"""Read-only recovery of passage text from Chroma by document ID (Phase 5 pilot)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb

from src.retriever import strip_passage_prefix

PILOT_JSON = ROOT / "results" / "phase5_generation_pilot_10.json"
CHROMA_PATH = ROOT / "chroma_db"
OUTPUT_MD = ROOT / "results" / "phase5_generation_manual_review_with_evidence.md"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def lookup_passage(collection, document_id: str) -> str | None:
    result = collection.get(
        where={"filename": document_id},
        include=["documents"],
    )
    documents = result.get("documents") or []
    if not documents:
        return None
    return strip_passage_prefix(documents[0])


def parse_citations(answer: str) -> list[int]:
    return [int(match) for match in CITATION_PATTERN.findall(answer or "")]


def citation_mapping(
    citations: list[int],
    retrieved_ids: list[str],
) -> list[dict[str, str | int]]:
    mapped: list[dict[str, str | int]] = []
    for rank in citations:
        index = rank - 1
        if 0 <= index < len(retrieved_ids):
            mapped.append({"citation": rank, "document_id": retrieved_ids[index]})
        else:
            mapped.append({"citation": rank, "document_id": "UNRESOLVED"})
    return mapped


def main() -> None:
    pilot = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection("amqa")

    required_ids: set[str] = set()
    for row in pilot["per_question"]:
        required_ids.update(row["retrieved_document_ids"])

    cache: dict[str, str | None] = {}
    unresolved: list[str] = []
    for doc_id in sorted(required_ids):
        text = lookup_passage(collection, doc_id)
        cache[doc_id] = text
        if text is None:
            unresolved.append(doc_id)

    if unresolved:
        print("STOP: unresolved document IDs in Chroma:")
        for doc_id in unresolved:
            print(f"  {doc_id}")
        raise SystemExit(1)

    passages_recovered = sum(1 for doc_id in required_ids if cache[doc_id])
    lines: list[str] = [
        "# Phase 5 Generation Pilot — Manual Review with Recovered Evidence",
        "",
        "**Source pilot:** `results/phase5_generation_pilot_10.json`",
        "**Evidence source:** Direct Chroma metadata lookup (`filename` = document ID); no similarity search.",
        "",
        f"**Passages recovered:** {passages_recovered} / {len(required_ids)} required retrieved-document lookups",
        f"**Unresolved IDs:** {len(unresolved)}",
        "",
        "---",
        "",
    ]

    for case_num, row in enumerate(pilot["per_question"], start=1):
        qid = row["question_id"]
        retrieved_ids = row["retrieved_document_ids"]
        retrieved_ranks = row["retrieved_ranks"]
        answer = row.get("generated_answer", "")
        citations = parse_citations(answer)
        citation_map = citation_mapping(citations, retrieved_ids)
        gold_rank = row.get("gold_document_rank")
        gold_rank_display = "not in top-3" if gold_rank is None else str(gold_rank)

        lines.extend(
            [
                f"## Case {case_num} — question_id {qid}",
                "",
                "### 1. Question ID",
                f"`{qid}`",
                "",
                "### 2. Original question",
                row["original_question"],
                "",
                "### 3. Rewritten query",
                row.get("rewritten_query", row["original_question"]),
                "",
                "### 4. Gold document ID and gold rank",
                f"- **Gold document ID:** `{row['gold_document_id']}`",
                f"- **Gold rank:** {gold_rank_display}",
                f"- **Ground truth (pilot record):** {row.get('ground_truth', '')}",
                "",
                "### 5. Retrieved documents and ranks",
                "",
                "| Rank | Document ID |",
                "|------|-------------|",
            ]
        )
        for rank, doc_id in zip(retrieved_ranks, retrieved_ids):
            lines.append(f"| {rank} | `{doc_id}` |")
        lines.append("")

        lines.append("### 6. Retrieved passage text (from Chroma)")
        lines.append("")
        for rank, doc_id in zip(retrieved_ranks, retrieved_ids):
            passage = cache[doc_id]
            lines.extend(
                [
                    f"#### Rank {rank} — Document `{doc_id}`",
                    "",
                    "```",
                    passage or "(not found)",
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "### 7. Generated answer",
                "",
                "```",
                answer,
                "```",
                "",
                "### 8. Parsed citations",
                "",
            ]
        )
        if citations:
            lines.append("Inline citation markers: " + ", ".join(f"[{c}]" for c in citations))
        else:
            lines.append("None (no inline citation markers in answer).")
        lines.append("")

        lines.append("### 9. Citation → document mapping")
        lines.append("")
        if citation_map:
            lines.extend(
                [
                    "| Citation | Document ID |",
                    "|----------|-------------|",
                ]
            )
            for item in citation_map:
                lines.append(f"| [{item['citation']}] | `{item['document_id']}` |")
        else:
            lines.append("No citations to map.")
        lines.append("")

        lines.extend(
            [
                "### Manual review classification",
                "",
                "- **Answer correctness:** [TO REVIEW]",
                "- **Groundedness:** [TO REVIEW]",
                "- **Citation correctness:** [TO REVIEW]",
                "- **Citation completeness:** [TO REVIEW]",
                "- **Relevance:** [TO REVIEW]",
                "- **Refusal correctness:** [TO REVIEW]",
                "",
                f"**Generation latency:** {row.get('generation_latency_ms', 'n/a')} ms",
            ]
        )
        if row.get("prompt_tokens") is not None:
            lines.append(
                f"**Tokens:** prompt {row['prompt_tokens']} | "
                f"completion {row.get('completion_tokens')} | total {row.get('total_tokens')}"
            )
        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## Summary table (manual inspection)",
            "",
            "| question_id | gold_rank | citations | refusal | Answer correctness | Groundedness | Citation correctness | Citation completeness | Relevance | Refusal correctness |",
            "|-------------|-----------|-----------|---------|-------------------|--------------|---------------------|----------------------|-----------|---------------------|",
        ]
    )
    for row in pilot["per_question"]:
        qid = row["question_id"]
        gold_rank = row.get("gold_document_rank")
        gold_rank_display = "not in top-3" if gold_rank is None else str(gold_rank)
        cites = ", ".join(f"[{c}]" for c in parse_citations(row.get("generated_answer", ""))) or "—"
        refusal = "yes" if row.get("refusal") else "no"
        lines.append(
            f"| {qid} | {gold_rank_display} | {cites} | {refusal} | "
            "[TO REVIEW] | [TO REVIEW] | [TO REVIEW] | [TO REVIEW] | [TO REVIEW] | [TO REVIEW] |"
        )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Cases processed: {len(pilot['per_question'])}")
    print(f"Passages successfully recovered: {passages_recovered}")
    print(f"Unresolved document IDs: {len(unresolved)}")
    print(f"Output file: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
