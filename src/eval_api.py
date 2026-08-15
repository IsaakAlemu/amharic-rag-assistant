"""Bounded Groq API retry helpers for evaluation scripts."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

RATE_LIMIT_MARKERS = ("rate-limited", "429", "rate limit", "too many requests")

# Per-call limits — never wait indefinitely.
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF_SECONDS = 4.0
DEFAULT_MAX_BACKOFF_SECONDS = 30.0

# Full eval viability — terminate early if quota is exhausted.
DEFAULT_MIN_ANSWER_SCORED_RATIO = 0.7
DEFAULT_MIN_TURNS_BEFORE_VIABILITY_CHECK = 4

T = TypeVar("T")


def classify_error(error: str | None) -> str | None:
    if not error:
        return None
    lower = error.lower()
    if any(marker in lower for marker in RATE_LIMIT_MARKERS):
        return "rate_limit"
    return "api_error"


def is_rate_limit_error(error: str | None) -> bool:
    return classify_error(error) == "rate_limit"


def backoff_delay(
    attempt: int,
    *,
    base_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> float:
    return min(max_seconds, base_seconds * (2**attempt))


def call_with_bounded_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    get_error: Callable[[T], str | None],
    label: str = "API call",
) -> tuple[T, int, str | None]:
    """
    Run fn up to max_retries times when rate-limited.

    Returns (result, retries_used, final_error_type).
    """
    last_result: T | None = None
    for attempt in range(max_retries):
        last_result = fn()
        error_type = classify_error(get_error(last_result))
        if error_type != "rate_limit":
            return last_result, attempt, error_type
        if attempt < max_retries - 1:
            delay = backoff_delay(
                attempt,
                base_seconds=base_backoff_seconds,
                max_seconds=max_backoff_seconds,
            )
            print(f"         {label} rate-limited — backoff {delay:.0f}s (retry {attempt + 1})")
            time.sleep(delay)
    assert last_result is not None
    return last_result, max_retries - 1, "rate_limit"


def evaluation_is_viable(
    *,
    turns_attempted: int,
    answer_scored: int,
    total_api_slots: int,
    min_answer_scored_ratio: float = DEFAULT_MIN_ANSWER_SCORED_RATIO,
    min_turns_before_check: int = DEFAULT_MIN_TURNS_BEFORE_VIABILITY_CHECK,
) -> bool:
    if turns_attempted < min_turns_before_check:
        return True
    if total_api_slots == 0:
        return False
    return (answer_scored / total_api_slots) >= min_answer_scored_ratio
