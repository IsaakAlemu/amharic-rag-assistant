"""Semantic retrieval over the Chroma vector store."""

from __future__ import annotations

from typing import TypedDict

from src.errors import RetrievalError

PASSAGE_PREFIX = "passage: "


class RetrievedDocument(TypedDict):
    text: str
    document_id: str
    distance: float
    rank: int


def strip_passage_prefix(text: str) -> str:
    if text.startswith(PASSAGE_PREFIX):
        return text[len(PASSAGE_PREFIX) :]
    return text


def retrieve(query: str, collection, model, top_k: int = 3) -> list[RetrievedDocument]:
    try:
        query_embedding = model.encode(
            "query: " + query,
            normalize_embeddings=True,
        )
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
    except Exception as exc:
        raise RetrievalError(f"Retrieval failed: {exc}") from exc

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)
    distances = results.get("distances", [[]])[0] or [0.0] * len(documents)

    if not documents:
        return []

    retrieved: list[RetrievedDocument] = []
    for index, (doc_text, metadata, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        retrieved.append(
            {
                "text": strip_passage_prefix(doc_text),
                "document_id": str(metadata.get("filename", "unknown")),
                "distance": float(distance),
                "rank": index,
            }
        )
    return retrieved
