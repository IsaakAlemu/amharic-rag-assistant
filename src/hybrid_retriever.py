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


class Reranker:
    """Two-stage cross-encoder re-ranker using FlashRank with graceful fallback."""

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        ranker: Any = None,
    ):
        self.model_name = model_name
        self.ranker = ranker
        self._available = False

        if self.ranker is not None:
            self._available = True
        else:
            try:
                from flashrank import Ranker

                self.ranker = Ranker(model_name=model_name)
                self._available = True
            except Exception:
                self.ranker = None
                self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self.ranker is not None

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """
        Re-score and re-order candidate documents using FlashRank.
        If FlashRank is unavailable or fails, gracefully returns the top_k candidates as-is.
        """
        if not documents:
            return []

        if not self.is_available or self.ranker is None:
            return documents[:top_k]

        try:
            from flashrank import RerankRequest

            passages = [
                {
                    "id": str(doc.get("document_id", f"doc_{idx}")),
                    "text": doc.get("text", ""),
                    "metadata": {
                        "distance": doc.get("distance", 0.0),
                        "original_rank": doc.get("rank", idx + 1),
                    },
                }
                for idx, doc in enumerate(documents)
            ]

            request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(request)

            reranked: list[RetrievedDocument] = []
            for rank_idx, item in enumerate(results[:top_k], start=1):
                meta = item.get("metadata") or {}
                orig_dist = meta.get("distance")
                score = float(item.get("score", 0.0))
                # Preserve original distance if valid, or derive a normalized distance from score
                dist = orig_dist if orig_dist is not None else 1.0 / (1.0 + max(0.0, score))
                reranked.append(
                    {
                        "document_id": str(item.get("id", "")),
                        "text": item.get("text", ""),
                        "distance": float(dist),
                        "rank": rank_idx,
                    }
                )
            return reranked
        except Exception:
            return documents[:top_k]


class HybridRetriever:
    """
    Two-stage Hybrid Retriever combining Dense Semantic Search (Chroma),
    Lexical BM25 keyword matching (RRF fusion), and FlashRank re-ranking.
    """

    def __init__(
        self,
        collection: Any = None,
        embed_model: Any = None,
        bm25_retriever: BM25Retriever | None = None,
        reranker: Reranker | None = None,
        use_reranker: bool = True,
        reranker_top_k: int = 5,
        initial_top_k: int = 15,
        rrf_k: int = 60,
    ):
        self.collection = collection
        self.embed_model = embed_model
        self.bm25_retriever = bm25_retriever
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        self.initial_top_k = initial_top_k
        self.rrf_k = rrf_k
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker | None:
        if self._reranker is None and self.use_reranker:
            try:
                self._reranker = Reranker()
            except Exception:
                self._reranker = None
        return self._reranker

    def retrieve(
        self,
        query: str,
        *,
        collection: Any = None,
        embed_model: Any = None,
        bm25_retriever: BM25Retriever | None = None,
        top_k: int | None = None,
        initial_top_k: int | None = None,
        use_reranker: bool | None = None,
        reranker_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Execute two-stage hybrid retrieval:
        1. Retrieve top `initial_top_k` candidate documents via Dense (Chroma) + BM25 with RRF.
        2. Format documents for FlashRank and re-score against the query if re-ranking is enabled.
        3. Fall back to RRF ranking if re-ranking is disabled or fails.
        """
        col = collection if collection is not None else self.collection
        model = embed_model if embed_model is not None else self.embed_model
        bm25 = bm25_retriever if bm25_retriever is not None else self.bm25_retriever
        should_rerank = use_reranker if use_reranker is not None else self.use_reranker
        r_top_k = (
            reranker_top_k
            if reranker_top_k is not None
            else (top_k if top_k is not None else self.reranker_top_k)
        )
        init_k = (
            initial_top_k
            if initial_top_k is not None
            else max(self.initial_top_k, r_top_k * 2)
        )
        k_const = rrf_k if rrf_k is not None else self.rrf_k

        dense_results: list[RetrievedDocument] = []
        if col is not None and model is not None:
            from src.retriever import retrieve as dense_retrieve

            dense_results = dense_retrieve(query, col, model, top_k=init_k)

        lexical_results: list[dict[str, Any]] = []
        if bm25 is not None:
            lexical_results = bm25.search(query, top_k=init_k)

        # Step 1: Candidate retrieval and RRF fusion
        candidates = reciprocal_rank_fusion(
            dense_results,
            lexical_results,
            rrf_k=k_const,
            top_k=init_k,
        )

        if not candidates:
            return []

        # Step 2: Re-ranking
        if should_rerank:
            active_reranker = self.reranker or Reranker()
            if active_reranker and active_reranker.is_available:
                return active_reranker.rerank(query, candidates, top_k=r_top_k)

        # Step 3: Graceful fallback to RRF candidates
        return candidates[:r_top_k]

