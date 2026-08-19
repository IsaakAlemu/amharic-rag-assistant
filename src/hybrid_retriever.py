"""Hybrid Retrieval module combining Dense Semantic Search (Chroma) and Lexical BM25 keyword matching with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from src.retriever import RetrievedDocument, strip_passage_prefix

_ETHIOPIC_WORD_REGEX = re.compile(r"[\w\u1200-\u137F]+")


def tokenize_amharic(text: str) -> list[str]:
    """Tokenize Amharic and Latin words for lexical matching."""
    return [t.lower() for t in _ETHIOPIC_WORD_REGEX.findall(text) if len(t) > 1]


class BM25Retriever:
    """Lightweight in-memory BM25 index tailored for Amharic document collections."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: list[int] = []
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[dict[str, Any]]) -> None:
        """
        Build BM25 index over a list of documents.
        Each doc dict must have 'id' (or 'document_id') and 'text'.
        """
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_lens = []
        self.doc_ids = []
        self.doc_texts = []
        self.doc_freqs = []
        df: dict[str, int] = defaultdict(int)

        for doc in documents:
            doc_id = str(doc.get("id", doc.get("document_id", "")))
            text = doc.get("text", "")
            tokens = tokenize_amharic(text)

            self.doc_ids.append(doc_id)
            self.doc_texts.append(text)
            self.doc_lens.append(len(tokens))

            tf: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.doc_freqs.append(tf)

            for token in tf.keys():
                df[token] += 1

        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

        # Compute IDF
        self.idf = {}
        for token, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search BM25 index and return ranked documents with scores."""
        query_tokens = tokenize_amharic(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: list[float] = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for i in range(self.corpus_size):
                tf = self.doc_freqs[i].get(token, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[i] += idf_val * (numerator / denominator)

        # Rank by descending score
        ranked_indices = sorted(
            [i for i in range(self.corpus_size) if scores[i] > 0],
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append(
                {
                    "document_id": self.doc_ids[idx],
                    "text": self.doc_texts[idx],
                    "bm25_score": scores[idx],
                    "rank": rank,
                }
            )
        return results


def reciprocal_rank_fusion(
    dense_results: list[RetrievedDocument],
    lexical_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    top_k: int = 3,
) -> list[RetrievedDocument]:
    """
    Combine Dense and BM25 results using Reciprocal Rank Fusion (RRF).
    Score = sum(1 / (k + rank))
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict[str, Any]] = {}

    # Accumulate Dense RRF scores
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["document_id"]
        scores[doc_id] += 1.0 / (rrf_k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = {"text": doc["text"], "distance": doc["distance"]}

    # Accumulate Lexical BM25 RRF scores
    for rank, doc in enumerate(lexical_results, start=1):
        doc_id = doc["document_id"]
        scores[doc_id] += 1.0 / (rrf_k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = {"text": doc["text"], "distance": 1.0 / (1.0 + doc.get("bm25_score", 1.0))}

    # Sort by descending fused RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]

    fused_results: list[RetrievedDocument] = []
    for rank, doc_id in enumerate(sorted_doc_ids, start=1):
        info = doc_map[doc_id]
        fused_results.append(
            {
                "document_id": doc_id,
                "text": info["text"],
                "distance": info["distance"],
                "rank": rank,
            }
        )

    return fused_results
