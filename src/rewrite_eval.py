"""Evaluation logic for query rewriting (no retrieval or generation)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

from src.eval_utils import normalize_text
from src.query_rewriter import _looks_vague_follow_up

PRONOUN_PATTERN = re.compile(
    r"(?<!\w)(እሱ|እርሱ|እ\/ሱ|ያ|ዚያ|ይህ|በዚያው|ዕቅዱ)(?!\w)",
    re.IGNORECASE,
)

TOPIC_KEYWORDS = {
    "olympics": ["ቶኪዮ", "ኦሎምፒክ", "ቴኳንዶ", "58"],
    "tax": ["ታክስ", "ገቢ"],
    "poetry": ["67", "ግጥም", "ጃዝ", "ዲስኩር"],
    "health": ["ጤና", "ደህንነት", "ዕቅድ"],
    "budget": ["በጀት", "ደቡብ", "ቢሊዮን"],
    "minister": ["ገቢ", "ሚኒስቴር", "ሚኒስትር"],
    "haile": ["ኃይሌ", "ልብ"],
    "un": ["መአሾ", "ሱዳን", "ተባበሩት"],
}


@dataclass
class RewriteEvalChecks:
    entity_topic_preservation: bool | None = None
    pronoun_resolution: bool | None = None
    standalone: bool | None = None
    history_preserved: bool | None = None
    no_irrelevant_history: bool | None = None
    meaning_preserved: bool | None = None

    @property
    def success(self) -> bool:
        values = [
            self.entity_topic_preservation,
            self.pronoun_resolution,
            self.standalone,
            self.history_preserved,
            self.no_irrelevant_history,
            self.meaning_preserved,
        ]
        applicable = [value for value in values if value is not None]
        return bool(applicable) and all(applicable)


@dataclass
class RewriteEvalResult:
    scenario_id: str
    category: str
    turn_index: int
    conversation_history: str
    original_follow_up: str
    generated_rewrite: str
    expected_intent: str
    rewrite_should_contain: list[str] = field(default_factory=list)
    checks: RewriteEvalChecks = field(default_factory=RewriteEvalChecks)
    failure_categories: list[str] = field(default_factory=list)
    success: bool = False
    retried: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checks"] = asdict(self.checks)
        return payload


def _contains_term(text: str, term: str) -> bool:
    return normalize_text(term) in normalize_text(text)


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    return all(_contains_term(text, term) for term in terms)


def _extract_pronouns(text: str) -> list[str]:
    return PRONOUN_PATTERN.findall(text)


def _history_user_text(history: list[dict]) -> str:
    return " ".join(turn["user"] for turn in history)


def _infer_expected_intent(turn: dict, history: list[dict], category: str) -> str:
    if turn.get("rewrite_should_contain"):
        terms = ", ".join(turn["rewrite_should_contain"])
        return f"Standalone question preserving context terms: {terms}"

    if category == "topic_change":
        return "New standalone topic — rewrite should remain unchanged."

    if category.startswith("out_of_scope"):
        return "Standalone out-of-scope question — rewrite should remain unchanged."

    if category == "pronoun_reference":
        prior = history[-1]["user"] if history else ""
        return f"Resolve pronouns using prior question context: {prior[:80]}..."

    if history:
        return f"Expand vague follow-up using prior context: {history[-1]['user'][:80]}..."

    return "Produce a standalone retrieval query."


def _should_remain_unchanged(turn: dict, category: str) -> bool:
    if turn.get("rewrite_should_be_unchanged"):
        return True
    if category == "topic_change" and len(turn["user"]) > 40:
        return True
    if category.startswith("out_of_scope"):
        return True
    return False


def _detect_wrong_topic_terms(
    rewrite: str,
    *,
    category: str,
    history: list[dict],
    turn: dict,
) -> list[str]:
    if category != "topic_change" or not history:
        return []

    prior_text = _history_user_text(history)
    current_text = turn["user"]
    wrong_terms: list[str] = []

    for keywords in TOPIC_KEYWORDS.values():
        prior_hits = [term for term in keywords if _contains_term(prior_text, term)]
        current_hits = [term for term in keywords if _contains_term(current_text, term)]
        rewrite_hits = [term for term in keywords if _contains_term(rewrite, term)]

        if prior_hits and not current_hits and rewrite_hits:
            wrong_terms.extend(rewrite_hits)

    return sorted(set(wrong_terms))


def evaluate_rewrite(
    *,
    scenario_id: str,
    category: str,
    turn_index: int,
    history: list[dict],
    turn: dict,
    original: str,
    rewrite: str,
    retried: bool = False,
    error: str | None = None,
) -> RewriteEvalResult:
    history_text = "\n".join(
        f"User: {item['user']}\nAssistant: ..." for item in history
    )
    expected_intent = _infer_expected_intent(turn, history, category)
    expected_terms = turn.get("rewrite_should_contain", [])
    checks = RewriteEvalChecks()
    failures: list[str] = []

    if error:
        return RewriteEvalResult(
            scenario_id=scenario_id,
            category=category,
            turn_index=turn_index,
            conversation_history=history_text,
            original_follow_up=original,
            generated_rewrite=rewrite,
            expected_intent=expected_intent,
            rewrite_should_contain=expected_terms,
            checks=checks,
            failure_categories=["api_error"],
            success=False,
            retried=retried,
            error=error,
        )

    unchanged_expected = _should_remain_unchanged(turn, category)

    if expected_terms:
        checks.entity_topic_preservation = _contains_all_terms(rewrite, expected_terms)
        if not checks.entity_topic_preservation:
            failures.append("entity_missing")

    original_pronouns = _extract_pronouns(original)
    if original_pronouns or category == "pronoun_reference":
        resolved = True
        if original_pronouns:
            unresolved = [
                pronoun
                for pronoun in original_pronouns
                if _contains_term(rewrite, pronoun)
                and not (expected_terms and _contains_all_terms(rewrite, expected_terms))
            ]
            if unresolved and not (expected_terms and _contains_all_terms(rewrite, expected_terms)):
                resolved = False
        if expected_terms and category == "pronoun_reference":
            resolved = resolved and _contains_all_terms(rewrite, expected_terms)
        checks.pronoun_resolution = resolved
        if not resolved:
            failures.append("pronoun_unresolved")

    if unchanged_expected:
        checks.standalone = normalize_text(original) == normalize_text(rewrite)
        if not checks.standalone:
            failures.append("meaning_changed")
    else:
        expanded = normalize_text(rewrite) != normalize_text(original)
        long_enough = len(rewrite) >= len(original) + 5
        vague = _looks_vague_follow_up(original)
        if expected_terms:
            checks.standalone = _contains_all_terms(rewrite, expected_terms) and (
                expanded or not vague
            )
        else:
            checks.standalone = expanded or not vague
        if not checks.standalone:
            failures.append("not_standalone")

    if history:
        if expected_terms:
            checks.history_preserved = _contains_all_terms(rewrite, expected_terms)
        elif unchanged_expected:
            checks.history_preserved = True
        else:
            prior_user = history[-1]["user"]
            prior_tokens = [
                token
                for token in re.findall(r"[\w\u1200-\u137F]+", prior_user)
                if len(token) >= 3
            ]
            checks.history_preserved = any(
                _contains_term(rewrite, token) for token in prior_tokens[:6]
            )
        if checks.history_preserved is False:
            failures.append("history_not_preserved")

    wrong_topic_terms = _detect_wrong_topic_terms(
        rewrite,
        category=category,
        history=history,
        turn=turn,
    )
    checks.no_irrelevant_history = len(wrong_topic_terms) == 0
    if not checks.no_irrelevant_history:
        failures.append("irrelevant_added")

    if unchanged_expected:
        checks.meaning_preserved = normalize_text(original) == normalize_text(rewrite)
    elif expected_terms:
        checks.meaning_preserved = _contains_all_terms(rewrite, expected_terms)
    else:
        checks.meaning_preserved = checks.standalone

    if checks.meaning_preserved is False and "meaning_changed" not in failures:
        failures.append("meaning_changed")

    return RewriteEvalResult(
        scenario_id=scenario_id,
        category=category,
        turn_index=turn_index,
        conversation_history=history_text,
        original_follow_up=original,
        generated_rewrite=rewrite,
        expected_intent=expected_intent,
        rewrite_should_contain=expected_terms,
        checks=checks,
        failure_categories=sorted(set(failures)),
        success=checks.success,
        retried=retried,
        error=error,
    )


def _should_remain_unchanged_for_row(row: RewriteEvalResult) -> bool:
    if row.category == "topic_change" and len(row.original_follow_up) > 40:
        return True
    if row.category.startswith("out_of_scope"):
        return True
    return False


def summarize_rewrite_eval(results: list[RewriteEvalResult]) -> dict:
    def rate(name: str) -> float | None:
        values = [
            getattr(row.checks, name)
            for row in results
            if getattr(row.checks, name) is not None
        ]
        if not values:
            return None
        return sum(1 for value in values if value) / len(values)

    follow_up_rows = [row for row in results if row.turn_index > 0]
    scored_rows = [row for row in follow_up_rows if row.rewrite_should_contain]

    pass_through_rows = [
        row for row in follow_up_rows if _should_remain_unchanged_for_row(row)
    ]
    out_of_scope_rows = [
        row for row in follow_up_rows if row.category.startswith("out_of_scope")
    ]

    def pass_through_accuracy(rows: list[RewriteEvalResult]) -> float | None:
        if not rows:
            return None
        correct = sum(
            1
            for row in rows
            if row.checks.standalone is True and row.checks.meaning_preserved is True
        )
        return correct / len(rows)

    standalone_pass_through = pass_through_accuracy(pass_through_rows)
    out_of_scope_pass_through = pass_through_accuracy(out_of_scope_rows)
    combined_pass_through_rows = pass_through_rows
    correct_pass_through_rate = pass_through_accuracy(combined_pass_through_rows)

    return {
        "turns_evaluated": len(results),
        "follow_up_turns": len(follow_up_rows),
        "rewrite_success_rate": sum(1 for row in follow_up_rows if row.success)
        / len(follow_up_rows)
        if follow_up_rows
        else None,
        "entity_topic_preservation_rate": rate("entity_topic_preservation"),
        "pronoun_resolution_rate": rate("pronoun_resolution"),
        "standalone_rate": rate("standalone"),
        "history_preserved_rate": rate("history_preserved"),
        "no_irrelevant_history_rate": rate("no_irrelevant_history"),
        "meaning_preserved_rate": rate("meaning_preserved"),
        "standalone_pass_through_accuracy": standalone_pass_through,
        "out_of_scope_pass_through_accuracy": out_of_scope_pass_through,
        "correct_pass_through_rate": correct_pass_through_rate,
        "pass_through_cases": len(pass_through_rows),
        "legacy_rewrite_match_rate": (
            sum(1 for row in scored_rows if row.checks.entity_topic_preservation)
            / len(scored_rows)
            if scored_rows
            else None
        ),
        "errors": sum(1 for row in results if row.error),
    }
