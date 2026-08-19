"""Load and validate user queries with security guardrails."""

from __future__ import annotations

from src.security import validate_and_sanitize_query


def validate_query(query: str, *, max_chars: int = 1000) -> str:
    """Validate query length, strip malicious control chars, and guard against prompt injection."""
    return validate_and_sanitize_query(query, max_chars=max_chars)

