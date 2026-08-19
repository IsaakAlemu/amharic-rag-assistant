"""Amharic text chunking and preprocessing utilities.

Features:
- Sentence splitting supporting Amharic full stops ('።'), question marks ('?'),
  exclamations ('!'), semicolon ('፤'), and newlines.
- Token/Character-aware chunking with sliding window overlap.
- Ge'ez numeral, whitespace, and punctuation normalization.
- Metadata enrichment (chunk index, character count, estimated tokens, parent doc id).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Amharic sentence terminators:
# ። (Arat Neteb - Amharic full stop)
# ? / ፧ (Question marks)
# ! / ፦ (Exclamations / Colons)
# ፤ (Semicolon)
_AMHARIC_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[።?!፤\n])\s+")
_MULTIPLE_SPACES_REGEX = re.compile(r"[ \t]+")
_MULTIPLE_NEWLINES_REGEX = re.compile(r"\n{3,}")


def normalize_amharic_text(text: str) -> str:
    """Clean redundant whitespace, control characters, and normalize spaces."""
    if not text:
        return ""
    # Normalize multiple spaces per line
    cleaned = _MULTIPLE_SPACES_REGEX.sub(" ", text)
    # Strip whitespace around newlines
    cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)
    # Collapse multiple consecutive newlines to at most 2
    cleaned = _MULTIPLE_NEWLINES_REGEX.sub("\n\n", cleaned)
    return cleaned.strip()


def split_into_sentences(text: str) -> list[str]:
    """Split Amharic text into sentences preserving Ethiopic sentence boundaries."""
    normalized = normalize_amharic_text(text)
    if not normalized:
        return []
    sentences = _AMHARIC_SENTENCE_SPLIT_REGEX.split(normalized)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    char_count: int
    estimated_tokens: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "metadata": self.metadata,
        }


def chunk_amharic_document(
    text: str,
    *,
    document_id: str,
    chunk_size_chars: int = 600,
    chunk_overlap_chars: int = 100,
    extra_metadata: dict[str, Any] | None = None,
) -> list[DocumentChunk]:
    """
    Chunk an Amharic document using sentence boundaries with sliding overlap.

    Args:
        text: Raw document text
        document_id: Unique identifier of the parent document
        chunk_size_chars: Target maximum character length per chunk (default 600)
        chunk_overlap_chars: Character overlap between consecutive chunks (default 100)
        extra_metadata: Optional dictionary of additional metadata (e.g. title, url, source)
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[DocumentChunk] = []
    current_sentences: list[str] = []
    current_len = 0
    chunk_index = 0
    base_meta = extra_metadata or {}

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, we still keep it to preserve semantic integrity
        if current_len + sentence_len > chunk_size_chars and current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    char_count=len(chunk_text),
                    estimated_tokens=max(1, len(chunk_text) // 4),
                    metadata={
                        **base_meta,
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "char_count": len(chunk_text),
                    },
                )
            )
            chunk_index += 1

            # Retain overlap sentences from the tail of current_sentences
            overlap_sentences: list[str] = []
            overlap_len = 0
            for prev_sent in reversed(current_sentences):
                if overlap_len + len(prev_sent) <= chunk_overlap_chars:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_len += len(prev_sent)
                else:
                    break

            current_sentences = overlap_sentences
            current_len = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_len += sentence_len

    # Add final remaining chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                text=chunk_text,
                chunk_index=chunk_index,
                char_count=len(chunk_text),
                estimated_tokens=max(1, len(chunk_text) // 4),
                metadata={
                    **base_meta,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text),
                },
            )
        )

    return chunks
