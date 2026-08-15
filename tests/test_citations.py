"""Unit tests for citation parsing and validation (no API calls)."""

from __future__ import annotations

import unittest

from src.citations import (
    build_citation_metadata,
    parse_citation_ranks,
    validate_citations,
)


def _source(rank: int, document_id: str) -> dict:
    return {
        "rank": rank,
        "document_id": document_id,
        "text": f"passage {rank}",
        "distance": 0.1,
    }


class ParseCitationRanksTests(unittest.TestCase):
    def test_no_citations(self) -> None:
        answer = "No citation markers here."
        self.assertEqual(parse_citation_ranks(answer), [])

    def test_one_citation(self) -> None:
        answer = "Answer text [1]."
        self.assertEqual(parse_citation_ranks(answer), [1])

    def test_multiple_citations(self) -> None:
        answer = "First [2] then [1]."
        self.assertEqual(parse_citation_ranks(answer), [2, 1])

    def test_duplicate_citations(self) -> None:
        answer = "X happened in 1969 [2]. It was significant [2][1]."
        self.assertEqual(parse_citation_ranks(answer), [2, 1])

    def test_invalid_citation_rank(self) -> None:
        answer = "Bad [0] and [abc] and good [3]."
        self.assertEqual(parse_citation_ranks(answer), [3])

    def test_answer_text_unchanged(self) -> None:
        answer = "X happened in 1969 [2]. It was significant [2][1]."
        original = answer
        parse_citation_ranks(answer)
        self.assertEqual(answer, original)


class ValidateCitationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = [
            _source(1, "A"),
            _source(2, "B"),
            _source(3, "C"),
        ]

    def test_valid_citation(self) -> None:
        result = validate_citations([2], self.sources)
        self.assertEqual(
            result,
            [{"rank": 2, "valid": True, "document_id": "B"}],
        )

    def test_invalid_citation_rank(self) -> None:
        result = validate_citations([5], self.sources)
        self.assertEqual(
            result,
            [{"rank": 5, "valid": False, "document_id": None}],
        )

    def test_build_metadata_out_of_order(self) -> None:
        answer = "First [2] then [1]."
        result = build_citation_metadata(answer, self.sources)
        self.assertEqual(
            result,
            [
                {"rank": 2, "valid": True, "document_id": "B"},
                {"rank": 1, "valid": True, "document_id": "A"},
            ],
        )
        self.assertEqual(answer, "First [2] then [1].")


if __name__ == "__main__":
    unittest.main()
