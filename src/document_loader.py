import json

from src.errors import RetrievalError


def load_documents(json_path):
    """
    Load AmQA-style data and return one document per paragraph,
    where the paragraph's context is treated as the full chunk text.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except FileNotFoundError as exc:
        raise RetrievalError(f"Data file not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise RetrievalError(f"Invalid JSON in data file: {json_path}") from exc

    if "data" not in raw:
        raise RetrievalError("Data file missing required 'data' field.")

    documents = []

    for article in raw["data"]:
        paragraphs = article.get("paragraphs", [])
        for para in paragraphs:
            if "context" not in para or "document_id" not in para:
                continue
            documents.append(
                {
                    "filename": str(para["document_id"]),
                    "path": json_path,
                    "text": para["context"],
                }
            )

    return documents
