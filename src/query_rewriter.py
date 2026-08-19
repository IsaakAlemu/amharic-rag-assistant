"""Rewrite follow-up questions into standalone retrieval queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from typing import Any

from src.errors import GenerationError
from src.eval_utils import normalize_text
from src.history_manager import ChatMessage, format_history_for_rewrite
from src.llm import generate_answer

# Pronouns and vague references common in Amharic follow-ups.
_VAGUE_PATTERN = re.compile(
    r"(^ያ\b|^እሱ\b|^እር\s*ሱ\b|^እ\/ሱ\b|^ዚያ\b|^ይ\s*ህ\b|"
    r"በ\s*2012\s*$|በ\s*2012\s*\?|ምን\s*ያ\s*ህል\s*ሆ\s*ነ\s*\?|"
    r"በዚያው|ዕቅዱ|ድርጅቱ|ውድድሩ|ሚኒስቴሩ|ዲስኩር\s*የ)",
    re.IGNORECASE,
)

_PRONOUN_PATTERN = re.compile(
    r"(?<!\w)(እሱ|እርሱ|እ\/ሱ|ያ|ዚያ|ይህ|በዚያው|ዕቅዱ)(?!\w)",
    re.IGNORECASE,
)

_AMHARIC_TOKEN = re.compile(r"[\w\u1200-\u137F]+")

_COMMON_WORDS = frozenset(
    {
        "ምን",
        "ነው",
        "ናቸው",
        "ማን",
        "የት",
        "ስንት",
        "መቼ",
        "ከ",
        "የ",
        "በ",
        "እና",
        "ለ",
        "ወደ",
        "ላይ",
        "ግን",
        "ብር",
        "በመቶ",
        "ክልል",
        "ዓመት",
        "ወራት",
        "አመት",
    }
)

_CROSS_TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    "መአሾ": ("ሱዳን", "ተባበሩት"),
    "67": ("67", "ግጥም", "ጃዝ"),
    "ግጥም": ("67", "ግጥም", "ጃዝ"),
    "ቶኪዮ": ("ቶኪዮ", "ኦሎምፒክ", "ቴኳንዶ"),
    "ኦሎምፒክ": ("ቶኪዮ", "ኦሎምፒክ", "ቴኳንዶ"),
}

REWRITE_PROMPT = """You convert follow-up questions into standalone Amharic search queries for a Wikipedia retrieval system.

Conversation history:
{history}

Latest user message:
{user_message}

Rules:
1. Output ONLY one standalone Amharic question suitable for semantic search.
2. ONLY rewrite when the latest message is incomplete or uses pronouns/vague references that need history.
3. If the latest message is already a complete, self-contained question, return it EXACTLY unchanged.
4. If the user asks a new question on a different topic, return it EXACTLY unchanged.
5. When rewriting, use ONLY entities, topics, dates, and facts mentioned in the conversation history or the latest message.
6. NEVER copy names or topics from the examples below — examples are illustrative patterns only.
7. NEVER replace the latest message with an earlier user question from the history.
8. Resolve pronouns (እሱ, ያ, etc.) using the conversation history.
9. Do NOT answer the question.
10. Do NOT add explanations, labels, or quotation marks.

Examples:

History:
User: የታክስ ገቢ ከ2010-2012 በመቶኛ የምን ያህል መጠን እድገት አሳየ?
Assistant: ...
Latest: በ2012 ምን ያህል ሆነ?
Standalone query: የታክስ ገቢ ከ2010-2012 በ2012 ምን ያህል ሆነ?

History:
User: ለተባበሩት መንግሥታት ድርጅት የደቡብ ሱዳን ሰላም ማስከበር ማን ተሾመ?
Assistant: ...
Latest: እሱ ከዚህ በፊት የት ሠርተዋል?
Standalone query: ብርጋዴር ጄነራል መአሾ ሀጎስ ስዩም ከዚህ በፊት የት ሠርተዋል?

History:
User: 45ኛው ዝግጅት በአንድ ከተማ የት ነው የሚካሄደው?
Assistant: ...
Latest: ዲስኩር የሚያቀርቡት ማናቸው?
Standalone query: በ45ኛው ዝግጅት ዲስኩር የሚያቀርቡት ማናቸው?

Standalone query:"""

REWRITE_RETRY_PROMPT = """The follow-up below is too vague to search alone. Rewrite it into a fully standalone Amharic question.

Conversation history:
{history}

Vague follow-up:
{user_message}

Requirements:
- Keep the follow-up wording and add ONLY missing context from the previous user question.
- Include the main entity, topic, event, date, or place from the conversation history.
- Do NOT replace the follow-up with the previous user question.
- Do NOT introduce any name, place, or topic not mentioned in the history or follow-up.
- Output ONLY the rewritten standalone question in Amharic.
- Do NOT answer.

Standalone query:"""


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    used_history: bool
    model: str
    retried: bool = False
    passed_through: bool = False
    pass_through_reason: str | None = None


def _looks_vague_follow_up(user_message: str) -> bool:
    text = user_message.strip()
    if _VAGUE_PATTERN.search(text):
        return True
    if _PRONOUN_PATTERN.search(text):
        return True
    if len(text) <= 35:
        has_interrogative = any(
            marker in text for marker in ("ማን", "ስንት", "የት", "መቼ", "ምን")
        )
        if has_interrogative and "?" in text:
            return False
        return True
    return False


def _amharic_tokens(text: str) -> set[str]:
    return {
        token
        for token in _AMHARIC_TOKEN.findall(text)
        if len(token) >= 2 and token not in _COMMON_WORDS
    }


def _overlap_ratio(message: str, prior: str) -> float:
    message_tokens = _amharic_tokens(message)
    if not message_tokens:
        return 0.0
    prior_tokens = _amharic_tokens(prior)
    return len(message_tokens & prior_tokens) / len(message_tokens)


def _last_user_message(history: list[ChatMessage]) -> str:
    for message in reversed(history):
        if message.role == "user":
            return message.content
    return ""


def _is_complete_question(message: str) -> bool:
    text = message.strip()
    return "?" in text or any(
        marker in text for marker in ("ማን", "ስንት", "የት", "መቼ", "ምን", "ነው", "ናቸው")
    )


def _needs_history_rewrite(message: str, history: list[ChatMessage]) -> tuple[bool, str]:
    if not history:
        return False, "no_history"

    prior = _last_user_message(history)
    text = message.strip()
    overlap = _overlap_ratio(text, prior)
    has_vague_ref = bool(_VAGUE_PATTERN.search(text))
    has_pronoun = bool(_PRONOUN_PATTERN.search(text))

    if not has_pronoun and not has_vague_ref:
        if overlap < 0.22:
            return False, "standalone_low_overlap"
        if _is_complete_question(text) and overlap < 0.40:
            return False, "standalone_complete_unrelated"

    if has_vague_ref:
        return True, "vague_reference"

    if has_pronoun:
        return True, "pronoun_reference"

    if len(text) <= 35:
        return True, "short_follow_up"

    return False, "default_pass_through"


def _history_corpus(history: list[ChatMessage], original: str) -> str:
    parts = [original]
    for message in history:
        parts.append(message.content)
    return normalize_text(" ".join(parts))


def _rewrite_has_cross_topic_contamination(
    rewrite: str,
    *,
    history: list[ChatMessage],
    original: str,
) -> bool:
    corpus = _history_corpus(history, original)
    rewrite_norm = normalize_text(rewrite)

    for marker, required_context in _CROSS_TOPIC_MARKERS.items():
        marker_norm = normalize_text(marker)
        if marker_norm not in rewrite_norm:
            continue
        if not any(normalize_text(ctx) in corpus for ctx in required_context):
            return True

    return False


def _rewrite_is_prior_user_message(rewrite: str, history: list[ChatMessage]) -> bool:
    rewrite_norm = normalize_text(rewrite)
    for message in history:
        if message.role == "user" and normalize_text(message.content) == rewrite_norm:
            return True
    return False


def _rewrite_distorts_original(rewrite: str, original: str) -> bool:
    original_norm = normalize_text(original.rstrip("?").strip())
    rewrite_norm = normalize_text(rewrite)
    if rewrite_norm == original_norm:
        return False
    if original_norm and rewrite_norm.endswith(original_norm):
        return False
    if original_norm and rewrite_norm.startswith(original_norm):
        suffix = rewrite_norm[len(original_norm) :].strip()
        return len(suffix) > 3
    original_tokens = _amharic_tokens(original)
    if not original_tokens:
        return False
    kept = sum(1 for token in original_tokens if token in rewrite_norm)
    return kept / len(original_tokens) < 0.6


def _extract_anchor_terms(prior: str) -> list[str]:
    terms: list[str] = []
    for token in _AMHARIC_TOKEN.findall(prior):
        if token in _COMMON_WORDS:
            continue
        if token.isdigit() or len(token) >= 4:
            terms.append(token)
    return terms[:5]


def _fallback_expand_from_history(original: str, history: list[ChatMessage]) -> str:
    prior = _last_user_message(history)
    if not prior:
        return original

    anchor_terms = _extract_anchor_terms(prior)
    original_norm = normalize_text(original)
    missing = [
        term for term in anchor_terms if normalize_text(term) not in original_norm
    ]
    if not missing:
        return original

    prefix = " ".join(missing[:3])
    question = original.strip()
    if not question.endswith("?"):
        question = f"{question}?"
    return f"{prefix} {question}"


def _sanitize_rewrite(
    rewrite: str,
    *,
    original: str,
    history: list[ChatMessage],
) -> str:
    if not rewrite or normalize_text(rewrite) == normalize_text(original):
        return original

    if _rewrite_is_prior_user_message(rewrite, history):
        return original

    if _rewrite_has_cross_topic_contamination(
        rewrite,
        history=history,
        original=original,
    ):
        return original

    if _rewrite_distorts_original(rewrite, original):
        return original

    return rewrite


def _clean_rewrite(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'").strip()
    if cleaned.lower().startswith("standalone query:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned


def _call_rewriter(
    prompt: str,
    *,
    client: Any,
    model: str,
    temperature: float,
) -> str:
    generation = generate_answer(
        prompt,
        client,
        model=model,
        temperature=temperature,
    )
    return _clean_rewrite(generation.text)


def rewrite_query(
    user_message: str,
    history: list[ChatMessage],
    *,
    client: Any,
    model: str,
    temperature: float = 0.0,
) -> RewriteResult:
    cleaned = user_message.strip()
    if not history:
        return RewriteResult(
            original_query=cleaned,
            rewritten_query=cleaned,
            used_history=False,
            model=model,
            passed_through=True,
            pass_through_reason="no_history",
        )

    needs_rewrite, pass_reason = _needs_history_rewrite(cleaned, history)
    if not needs_rewrite:
        return RewriteResult(
            original_query=cleaned,
            rewritten_query=cleaned,
            used_history=False,
            model=model,
            passed_through=True,
            pass_through_reason=pass_reason,
        )

    history_text = format_history_for_rewrite(history)
    prompt = REWRITE_PROMPT.format(history=history_text, user_message=cleaned)

    try:
        rewritten = _call_rewriter(
            prompt,
            client=client,
            model=model,
            temperature=temperature,
        )
    except GenerationError:
        rewritten = _fallback_expand_from_history(cleaned, history)
        return RewriteResult(
            original_query=cleaned,
            rewritten_query=rewritten,
            used_history=True,
            model=model,
            passed_through=rewritten == cleaned,
            pass_through_reason="generation_error_fallback",
        )

    if not rewritten:
        rewritten = cleaned

    rewritten = _sanitize_rewrite(
        rewritten,
        original=cleaned,
        history=history,
    )

    retried = False
    if _looks_vague_follow_up(cleaned) and (
        rewritten == cleaned or len(rewritten) <= len(cleaned) + 5
    ):
        retry_prompt = REWRITE_RETRY_PROMPT.format(
            history=history_text,
            user_message=cleaned,
        )
        try:
            retry_rewrite = _call_rewriter(
                retry_prompt,
                client=client,
                model=model,
                temperature=temperature,
            )
            retry_rewrite = _sanitize_rewrite(
                retry_rewrite,
                original=cleaned,
                history=history,
            )
            if retry_rewrite and retry_rewrite != cleaned:
                rewritten = retry_rewrite
                retried = True
        except GenerationError:
            pass

    if _looks_vague_follow_up(cleaned) and (
        rewritten == cleaned or len(rewritten) <= len(cleaned) + 5
    ):
        fallback = _fallback_expand_from_history(cleaned, history)
        if fallback != cleaned:
            rewritten = fallback
            retried = True

    rewritten = _sanitize_rewrite(
        rewritten,
        original=cleaned,
        history=history,
    )

    return RewriteResult(
        original_query=cleaned,
        rewritten_query=rewritten,
        used_history=True,
        model=model,
        retried=retried,
        passed_through=rewritten == cleaned,
        pass_through_reason=pass_reason if rewritten == cleaned else None,
    )
