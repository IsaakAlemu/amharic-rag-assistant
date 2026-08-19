"""CLI script to ingest, chunk, and embed custom Amharic documents into ChromaDB.

Supports:
- JSON files (list of dicts with 'id' and 'text', or AmQA style)
- Plain text (.txt) and Markdown (.md) Amharic files
- Automatic sentence-boundary chunking with overlap
- Custom collection names and persistence directories
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from sentence_transformers import SentenceTransformer

from config import get_settings
from src.chunker import chunk_amharic_document, normalize_amharic_text
from src.logging_config import setup_logging


def load_file_documents(file_path: Path) -> list[dict]:
    """Extract raw documents from JSON, TXT, or MD files."""
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    docs: list[dict] = []
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, item in enumerate(data):
                doc_id = str(item.get("id", item.get("document_id", f"{file_path.stem}_{i}")))
                text = item.get("text", item.get("context", item.get("content", "")))
                if text.strip():
                    docs.append({"id": doc_id, "text": text, "metadata": item.get("metadata", {})})
        elif isinstance(data, dict) and "data" in data:
            # AmQA format
            for article in data["data"]:
                for para in article.get("paragraphs", []):
                    if "context" in para and "document_id" in para:
                        docs.append(
                            {
                                "id": str(para["document_id"]),
                                "text": para["context"],
                                "metadata": {"title": article.get("title", "")},
                            }
                        )
    elif suffix in (".txt", ".md"):
        with file_path.open("r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            docs.append({"id": file_path.stem, "text": text, "metadata": {"source_file": str(file_path)}})
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .json, .txt, or .md")

    return docs


def ingest_corpus(
    file_path: str,
    *,
    collection_name: str = "amqa_expanded",
    persist_path: str = "chroma_db",
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    batch_size: int = 64,
) -> int:
    """Process, chunk, and embed documents into ChromaDB."""
    settings = get_settings()
    path = Path(file_path)
    print(f"📖 Loading documents from: {path}")
    raw_docs = load_file_documents(path)
    print(f"✅ Found {len(raw_docs)} raw documents.")

    all_chunks = []
    for doc in raw_docs:
        chunks = chunk_amharic_document(
            doc["text"],
            document_id=doc["id"],
            chunk_size_chars=chunk_size,
            chunk_overlap_chars=chunk_overlap,
            extra_metadata=doc.get("metadata"),
        )
        all_chunks.extend(chunks)

    print(f"✂️ Created {len(all_chunks)} sentence-bounded chunks (size={chunk_size}, overlap={chunk_overlap}).")
    if not all_chunks:
        print("⚠️ No chunks produced.")
        return 0

    print(f"🧠 Loading embedding model: {settings.embed_model}...")
    embed_model = SentenceTransformer(settings.embed_model)

    print(f"💾 Initializing ChromaDB at: {persist_path} (collection='{collection_name}')")
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(collection_name)

    texts = [f"passage: {chunk.text}" for chunk in all_chunks]
    ids = [chunk.chunk_id for chunk in all_chunks]
    metadatas = [
        {
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
            "source_file": str(path.name),
        }
        for chunk in all_chunks
    ]

    print(f"⚡ Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    print(f"📥 Upserting to ChromaDB...")
    collection.upsert(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas,
    )

    print(f"🎉 Ingestion complete! Total items in collection: {collection.count()}")
    return len(all_chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk Amharic documents into ChromaDB.")
    parser.add_argument("--file", type=str, default="data/raw/train_data.json", help="Path to document file (.json, .txt, .md)")
    parser.add_argument("--collection", type=str, default="amqa", help="ChromaDB collection name")
    parser.add_argument("--persist-path", type=str, default="chroma_db", help="ChromaDB persistence path")
    parser.add_argument("--chunk-size", type=int, default=600, help="Target chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in characters")

    args = parser.parse_args()
    ingest_corpus(
        args.file,
        collection_name=args.collection,
        persist_path=args.persist_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
