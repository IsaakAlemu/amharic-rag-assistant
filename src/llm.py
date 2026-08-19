"""Multi-provider LLM generation (Gemini & Groq)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.errors import GenerationError

REFUSAL_PHRASE = "I don't know based on the provided documents."
AMHARIC_REFUSAL_PHRASE = "ከተሰጡት ሰነዶች በመነሳት ጥያቄውን መመለስ አልተቻለም።"


def is_refusal(text: str) -> bool:
    """Check if generated text matches English or Amharic refusal phrases."""
    clean = text.strip()
    return (
        REFUSAL_PHRASE in clean
        or AMHARIC_REFUSAL_PHRASE in clean
        or "አልተቻለም" in clean and ("ሰነድ" in clean or "ማስረጃ" in clean)
    )


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
    if "429" not in message and ("rate" not in message or "limit" not in message or "resource_exhausted" not in message):
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


def _generate_with_gemini(
    prompt: str,
    client: Any,
    *,
    model: str,
    temperature: float,
) -> GenerationResult:
    try:
        from google.genai import types

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
    except Exception as exc:
        message = str(exc).lower()
        if "429" in message or "resource_exhausted" in message or "rate" in message:
            raise GenerationError(
                "Gemini API rate limit reached. Please retry shortly."
            ) from exc
        raise GenerationError(f"Gemini generation failed: {exc}") from exc

    text = response.text
    if not text or not text.strip():
        raise GenerationError("Gemini returned an empty response.")

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    usage = getattr(response, "usage_metadata", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        completion_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)

    return GenerationResult(
        text=text.strip(),
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _generate_with_groq(
    prompt: str,
    client: Any,
    *,
    model: str,
    temperature: float,
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

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    return GenerationResult(
        text=content.strip(),
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def generate_answer(
    prompt: str,
    client: Any,
    *,
    model: str,
    temperature: float = 0.1,
) -> GenerationResult:
    """Generate answer using Gemini client (google.genai.Client) or Groq client."""
    client_type = type(client).__name__.lower()
    client_module = type(client).__module__.lower()
    if "genai" in client_module or "genai" in client_type:
        return _generate_with_gemini(prompt, client, model=model, temperature=temperature)
    elif "groq" in client_module or "groq" in client_type or hasattr(client, "chat"):
        return _generate_with_groq(prompt, client, model=model, temperature=temperature)
    else:
        raise GenerationError(f"Unsupported LLM client type: {type(client)}")


def generate_answer_stream(
    prompt: str,
    client: Any,
    *,
    model: str,
    temperature: float = 0.1,
):
    """Generate token stream using Gemini client or Groq client."""
    client_type = type(client).__name__.lower()
    client_module = type(client).__module__.lower()
    if "genai" in client_module or "genai" in client_type:
        from google.genai import types

        response_stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    elif "groq" in client_module or "groq" in client_type or hasattr(client, "chat"):
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    else:
        raise GenerationError(f"Unsupported LLM client type: {type(client)}")
