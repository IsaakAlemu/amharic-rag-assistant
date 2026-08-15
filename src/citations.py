"""Parse and validate inline document citations in generated answers."""

from __future__ import annotations

import re
from typing import TypedDict

from src.retriever import RetrievedDocument

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CitationValidation(TypedDict):
    rank: int
    valid: bool
    document_id: str | None


def parse_citation_ranks(text: str) -> list[int]:
    """
    Extract citation ranks from answer text in first-seen order, deduplicated.

    Does not modify the answer text. Ignores malformed/non-positive ranks.
    """
    ranks: list[int] = []
    seen: set[int] = set()
    for match in CITATION_PATTERN.finditer(text):
        try:
            rank = int(match.group(1))
        except ValueError:
            continue
        if rank <= 0 or rank in seen:
            continue
        seen.add(rank)
        ranks.append(rank)
    return ranks


def validate_citations(
    ranks: list[int],
    sources: list[RetrievedDocument],
) -> list[CitationValidation]:
    """Map parsed citation ranks to retrieved documents; mark invalid ranks."""
    rank_to_document_id = {source["rank"]: source["document_id"] for source in sources}
    validated: list[CitationValidation] = []
    for rank in ranks:
        document_id = rank_to_document_id.get(rank)
        validated.append(
            {
                "rank": rank,
                "valid": document_id is not None,
                "document_id": document_id,
            }
        )
    return validated


def build_citation_metadata(
    answer: str,
    sources: list[RetrievedDocument],
) -> list[CitationValidation]:
    """Parse citations from an answer and validate against retrieved sources."""
    return validate_citations(parse_citation_ranks(answer), sources)
