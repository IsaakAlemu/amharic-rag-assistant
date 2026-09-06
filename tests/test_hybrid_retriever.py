"""Unit tests for BM25 and Hybrid Retrieval (RRF)."""

from __future__ import annotations

import unittest

from src.hybrid_retriever import (
    BM25Retriever,
    HybridRetriever,
    Reranker,
    reciprocal_rank_fusion,
    tokenize_amharic,
)
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

    def test_reranker_execution_and_top_k(self):
        candidates: list[RetrievedDocument] = [
            {"document_id": "doc1", "text": "Apple and banana fruits in the market", "distance": 0.5, "rank": 1},
            {"document_id": "doc2", "text": "Fresh red apples are delicious", "distance": 0.6, "rank": 2},
            {"document_id": "doc3", "text": "Vehicle maintenance guide and cars", "distance": 0.7, "rank": 3},
        ]
        reranker = Reranker()
        reranked = reranker.rerank(query="fresh apples", documents=candidates, top_k=2)

        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["rank"], 1)
        self.assertEqual(reranked[1]["rank"], 2)
        # The document about fresh red apples should rank higher for the query 'fresh apples'
        self.assertEqual(reranked[0]["document_id"], "doc2")

    def test_reranker_fallback_when_disabled(self):
        hybrid = HybridRetriever(
            bm25_retriever=self.bm25,
            use_reranker=False,
            reranker_top_k=2,
        )
        results = hybrid.retrieve("መአሾ ሀጎስ")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_id"], "doc1")

    def test_reranker_fallback_on_error(self):
        class BrokenRanker:
            def rerank(self, request):
                raise RuntimeError("Ranker crashed")

        broken_reranker = Reranker(ranker=BrokenRanker())
        candidates: list[RetrievedDocument] = [
            {"document_id": "docA", "text": "Sample text A", "distance": 0.1, "rank": 1},
            {"document_id": "docB", "text": "Sample text B", "distance": 0.2, "rank": 2},
        ]
        results = broken_reranker.rerank("any query", candidates, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_id"], "docA")


if __name__ == "__main__":
    unittest.main()

