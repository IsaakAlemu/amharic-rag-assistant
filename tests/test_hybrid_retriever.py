"""Unit tests for BM25 and Hybrid Retrieval (RRF)."""

from __future__ import annotations

import unittest

from src.hybrid_retriever import BM25Retriever, reciprocal_rank_fusion, tokenize_amharic
from src.retriever import RetrievedDocument


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"id": "doc1", "text": "ብርጋዴር ጄነራል መአሾ ሀጎስ ስዩም ለተባበሩት መንግስታት ተሾሙ።"},
            {"id": "doc2", "text": "ኢትዮጵያ የአስትሮኖሚካል ሲምፖዚየም አዘጋጅታለች።"},
            {"id": "doc3", "text": "አዲስ አበባ የአፍሪካ ህብረት መቀመጫ ናት።"},
        ]
        self.bm25 = BM25Retriever()
        self.bm25.fit(self.docs)

    def test_tokenize_amharic(self):
        tokens = tokenize_amharic("ሰላም ዓለም! 1965 ዓ.ም")
        self.assertIn("ሰላም", tokens)
        self.assertIn("ዓለም", tokens)

    def test_bm25_exact_keyword_match(self):
        results = self.bm25.search("መአሾ ሀጎስ", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_id"], "doc1")

    def test_reciprocal_rank_fusion(self):
        dense_results: list[RetrievedDocument] = [
            {"document_id": "doc2", "text": "text2", "distance": 0.1, "rank": 1},
            {"document_id": "doc1", "text": "text1", "distance": 0.3, "rank": 2},
        ]
        lexical_results = [
            {"document_id": "doc1", "text": "text1", "bm25_score": 5.0, "rank": 1},
            {"document_id": "doc3", "text": "text3", "bm25_score": 2.0, "rank": 2},
        ]

        fused = reciprocal_rank_fusion(dense_results, lexical_results, top_k=2)
        self.assertEqual(len(fused), 2)
        # doc1 was rank 2 in dense and rank 1 in lexical -> should fuse to rank 1!
        self.assertEqual(fused[0]["document_id"], "doc1")


if __name__ == "__main__":
    unittest.main()
