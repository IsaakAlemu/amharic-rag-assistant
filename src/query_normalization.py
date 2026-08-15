"""Amharic query normalization for retrieval experiments.

Addresses verified holdout failure patterns (see retrieval_failure_analysis.json):
- ሠ / ሰ spelling variants
- punctuation normalization
- year/date label formatting
- whitespace normalization

Disabled by default. Controlled experiment (results/normalization_experiment.json,
329 holdout Qs) showed net Hit@1 -47 (-14.29 pp) with 53 regressions vs 6
improvements — do not enable in production. Primary harm: stripping punctuation
(including ? and abbreviation dots) shifts E5 query embeddings away from
indexed passage text.
"""

from __future__ import annotations

import re
import unicodedata

# Set True only after a controlled experiment demonstrates net Hit@1 gain.
NORMALIZATION_ENABLED = False

# Verified morphology variant: query ሠናይት vs corpus ሰናይት (U+1220 → U+1230).
SZA_CHAR = "\u1220"  # ሠ
SA_CHAR = "\u1230"  # ሰ

ETHIOPIC_PUNCT_TO_SPACE = str.maketrans(
    {
        "\u1361": " ",
        "\u1362": " ",
        "\u1363": " ",
        "\u1364": " ",
        "\u1365": " ",
        "\u1366": " ",
        "\u1367": " ",
        "\u1368": " ",
        "?": " ",
        "!": " ",
        ";": " ",
        ":": " ",
        ",": " ",
        ".": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "\u201c": " ",
        "\u201d": " ",
        "'": " ",
        '"': " ",
        "/": " ",
        "\\": " ",
        "|": " ",
        "-": " ",
        "\u2013": " ",
        "\u2014": " ",
    }
)

_DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ዓ\s*\.\s*ም\s*\.?", re.UNICODE), "ዓ.ም"),
    (re.compile(r"እ\s*\.\s*ኤ\s*\.\s*አ\s*\.?", re.UNICODE), "እ.ኤ.አ"),
    (re.compile(r"(\d{4})\s*/\s*(\d{1,2})", re.UNICODE), r"\1 \2"),
    (re.compile(r"(\d{1,2})\s*/\s*(\d{4})", re.UNICODE), r"\1 \2"),
]


def normalize_amharic_query(query: str) -> str:
    """Return a normalized copy of *query* for embedding-based retrieval."""
    if not query:
        return query

    text = unicodedata.normalize("NFKC", query)
    text = text.replace(SZA_CHAR, SA_CHAR)
    text = text.translate(ETHIOPIC_PUNCT_TO_SPACE)

    for pattern, replacement in _DATE_PATTERNS:
        text = pattern.sub(replacement, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def maybe_normalize_query(query: str) -> str:
    """Apply normalization only when NORMALIZATION_ENABLED is True."""
    if not NORMALIZATION_ENABLED:
        return query
    return normalize_amharic_query(query)
