"""Token counting utilities for prompt budgeting."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

FALLBACK_CHARS_PER_TOKEN = 3.0


@lru_cache(maxsize=1)
def _load_tokenizer():
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            use_fast=True,
        )
        logger.info("Loaded Llama 3 tokenizer for token estimation.")
        return tokenizer
    except Exception as exc:
        logger.warning(
            "Could not load Llama tokenizer (%s). Using character-based estimate.",
            exc,
        )
        return None


class TokenCounter:
    """Estimate token counts for Llama-style models."""

    def __init__(self, *, chars_per_token: float = FALLBACK_CHARS_PER_TOKEN) -> None:
        self.chars_per_token = chars_per_token
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer()
        return self._tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return max(1, int(len(text) / self.chars_per_token))

    def estimate(self, text: str) -> int:
        return self.count(text)
