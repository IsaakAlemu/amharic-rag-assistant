"""Security guardrails, prompt injection detection, and input sanitization."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import ClassVar

from src.errors import ValidationError

# Common prompt injection / jailbreak patterns in English and Amharic
_INJECTION_PATTERNS = [
    # English injection patterns
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)\b",
    r"(?i)\byou\s+are\s+now\s+(unrestricted|in\s+developer\s+mode|dan)\b",
    r"(?i)\brepeat\s+(the\s+)?(system\s+prompt|initial\s+instructions)\b",
    r"(?i)\bwhat\s+(is|are)\s+your\s+(system\s+prompt|instructions|initial\s+prompt)\b",
    r"(?i)\bprint\s+(the\s+)?(system\s+prompt|system\s+message)\b",
    r"(?i)\bsystem\s+prompt\s+(leak|reveal|print|dump)\b",
    r"(?i)\b(system|developer)\s+mode\s+enabled\b",
    r"(?i)<\s*system\s*>",
    r"(?i)<\s*script\s*>",
    
    # Amharic injection patterns (direct translations of jailbreaks)
    r"የቀደመውን\s*መመሪያ\s*(እርሳው|ሰርዘው|ተወው)",
    r"የቀደሙትን\s*ህጎች\s*(ተወው|አትከተል|እርሳቸው)",
    r"የሲስተሙን\s*መመሪያ\s*(ንገረኝ|አሳየኝ|ግለጽ)",
    r"የመጀመሪያውን\s*ፕሮምፕት\s*(አውጣ|ንገረኝ)",
    r"ያለ\s*ምንም\s*ገደብ\s*መልስ",
]

_COMPILED_INJECTION_REGEX = [re.compile(p) for p in _INJECTION_PATTERNS]

# Control character removal regex (except standard whitespace and newlines)
_CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


@dataclass
class SecurityCheckResult:
    is_safe: bool
    sanitized_text: str
    flagged_reason: str | None = None


def sanitize_input_text(text: str) -> str:
    """Strip malicious control characters and normalize text."""
    if not text:
        return ""
    # Strip null bytes and control chars
    cleaned = _CONTROL_CHARS_REGEX.sub("", text)
    # Strip dangerous HTML script tags
    cleaned = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def check_prompt_injection(text: str) -> SecurityCheckResult:
    """
    Inspect input text for prompt injection and jailbreak signatures.
    Returns a SecurityCheckResult indicating safety and sanitized text.
    """
    sanitized = sanitize_input_text(text)
    if not sanitized:
        return SecurityCheckResult(is_safe=False, sanitized_text="", flagged_reason="Empty input")

    for pattern in _COMPILED_INJECTION_REGEX:
        if pattern.search(sanitized):
            return SecurityCheckResult(
                is_safe=False,
                sanitized_text=sanitized,
                flagged_reason="Potential prompt injection or adversarial command detected.",
            )

    return SecurityCheckResult(is_safe=True, sanitized_text=sanitized, flagged_reason=None)


def validate_and_sanitize_query(query: str, *, max_chars: int = 1000) -> str:
    """
    Comprehensive query validation, character-length enforcement,
    and prompt-injection security check.
    """
    cleaned = sanitize_input_text(query)
    if not cleaned:
        raise ValidationError("Please enter a question / እባክዎ ጥያቄ ያስገቡ።")

    if len(cleaned) > max_chars:
        raise ValidationError(
            f"Question is too long ({len(cleaned)} chars). Maximum is {max_chars}."
        )

    security_result = check_prompt_injection(cleaned)
    if not security_result.is_safe:
        raise ValidationError(
            "የገቡት ጥያቄ ተቀባይነት የሌለው የትዕዛዝ ይዘት አለው። እባክዎ ተገቢ የሆነ ጥያቄ ይጠይቁ። "
            "(Security check: Adversarial instruction or prompt injection attempt blocked.)"
        )

    return security_result.sanitized_text
