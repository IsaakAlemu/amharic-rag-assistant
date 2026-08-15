"""Groq LLM generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from groq import Groq

from src.errors import GenerationError

REFUSAL_PHRASE = "I don't know based on the provided documents."


@dataclass
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def extract_rate_limit_details(exc: Exception) -> dict[str, Any] | None:
    """Capture rate-limit metadata from an API exception when safely available."""
    message = str(exc).lower()
    if "429" not in message and ("rate" not in message or "limit" not in message):
        return None

    details: dict[str, Any] = {"error_message": str(exc)}
    response = getattr(exc, "response", None)
    if response is None:
        return details

    headers = getattr(response, "headers", None)
    if headers is None:
        return details

    header_get = getattr(headers, "get", None)
    if not callable(header_get):
        return details

    for key in (
        "retry-after",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    ):
        value = header_get(key)
        if value is not None:
            details[key] = value

    return details


def _extract_usage(response) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )


def generate_answer(
    prompt: str,
    client: Groq,
    *,
    model: str,
    temperature: float = 0.1,
) -> GenerationResult:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "rate" in message and "limit" in message:
            raise GenerationError(
                "The language model is temporarily rate-limited. Please retry shortly."
            ) from exc
        if "token" in message and ("limit" in message or "tpm" in message):
            raise GenerationError(f"Token limit exceeded: {exc}") from exc
        raise GenerationError(f"Generation failed: {exc}") from exc

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise GenerationError("The language model returned an empty response.")

    prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
    return GenerationResult(
        text=content.strip(),
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
