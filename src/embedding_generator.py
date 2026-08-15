import chromadb

from src.errors import RetrievalError


def generate_embeddings(chunks, model, persist_path="chroma_db"):
    try:
        client = chromadb.PersistentClient(path=persist_path)
        collection = client.get_or_create_collection("amqa")
    except Exception as exc:
        raise RetrievalError(f"Failed to connect to Chroma at {persist_path}: {exc}") from exc

    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks — skipping re-embedding.")
        return collection

    if not chunks:
        raise RetrievalError("No documents available to embed.")

    texts = ["passage: " + chunk["text"] for chunk in chunks]

    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
    except Exception as exc:
        raise RetrievalError(f"Embedding generation failed: {exc}") from exc

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[f"{chunk['filename']}_{i}" for i, chunk in enumerate(chunks)],
        metadatas=[
            {"filename": chunk["filename"], "path": chunk["path"]} for chunk in chunks
        ],
    )

    print(f"Embedded and stored {len(chunks)} chunks.")
    return collection
