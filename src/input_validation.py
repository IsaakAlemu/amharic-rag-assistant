"""Load and validate user queries."""

from __future__ import annotations

from src.errors import ValidationError


def validate_query(query: str, *, max_chars: int) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise ValidationError("Please enter a question.")
    if len(cleaned) > max_chars:
        raise ValidationError(
            f"Question is too long ({len(cleaned)} chars). Maximum is {max_chars}."
        )
    return cleaned
