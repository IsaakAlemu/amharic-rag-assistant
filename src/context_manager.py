"""Rank-preserving context budgeting for RAG prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.prompt_builder import SYSTEM_RULES, SYSTEM_RULES_COMPRESSED, build_prompt
from src.retriever import RetrievedDocument
from src.token_counter import TokenCounter

ContextStrategyName = Literal[
    "baseline",
    "compressed_system",
    "budget_6000",
    "budget_4500",
    "budget_3500",
]

STRATEGY_BUDGETS: dict[str, int | None] = {
    "baseline": None,
    "compressed_system": None,
    "budget_6000": 6000,
    "budget_4500": 4500,
    "budget_3500": 3500,
}

MIN_DOC_CHARS = 120
TRUNCATION_STEP_CHARS = 80


@dataclass
class ContextDecision:
    strategy: str
    original_doc_count: int
    final_doc_count: int
    truncated_docs: list[str] = field(default_factory=list)
    dropped_docs: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    system_rules_variant: str = "full"


@dataclass
class PreparedContext:
    docs: list[RetrievedDocument]
    prompt: str
    decision: ContextDecision


def _clone_docs(docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
    return [
        {
            "text": doc["text"],
            "document_id": doc["document_id"],
            "distance": doc["distance"],
            "rank": doc["rank"],
        }
        for doc in docs
    ]


def _system_rules_for_strategy(strategy: str) -> tuple[str, str]:
    if strategy == "compressed_system" or strategy.startswith("budget_"):
        return SYSTEM_RULES_COMPRESSED, "compressed"
    return SYSTEM_RULES, "full"


def _prompt_tokens(
    query: str,
    docs: list[RetrievedDocument],
    *,
    system_rules: str,
    token_counter: TokenCounter,
) -> int:
    return token_counter.count(build_prompt(query, docs, system_rules=system_rules))


def _truncate_doc_from_end(doc: RetrievedDocument, chars: int) -> RetrievedDocument:
    trimmed = doc.copy()
    trimmed["text"] = doc["text"][:-chars] if chars < len(doc["text"]) else doc["text"][:MIN_DOC_CHARS]
    return trimmed


def apply_context_strategy(
    query: str,
    docs: list[RetrievedDocument],
    *,
    strategy: str,
    token_counter: TokenCounter,
    max_prompt_tokens: int | None = None,
) -> PreparedContext:
    """
    Apply a controlled context strategy.

    Rank-preserving policy:
    1. Never reorder retrieved documents.
    2. Truncate lower-ranked documents from the tail first.
    3. Drop lower-ranked documents only after truncation is exhausted.
    4. Rank-1 is truncated only as a last resort; it is never dropped if it is the
       only remaining document.
    """
    if strategy not in STRATEGY_BUDGETS and max_prompt_tokens is None:
        raise ValueError(f"Unknown context strategy: {strategy}")

    working = _clone_docs(docs)
    system_rules, rules_variant = _system_rules_for_strategy(strategy)
    budget = max_prompt_tokens if max_prompt_tokens is not None else STRATEGY_BUDGETS[strategy]

    decision = ContextDecision(
        strategy=strategy,
        original_doc_count=len(docs),
        final_doc_count=len(working),
        system_rules_variant=rules_variant,
    )

    if budget is None:
        prompt = build_prompt(query, working, system_rules=system_rules)
        decision.prompt_tokens = token_counter.count(prompt)
        decision.final_doc_count = len(working)
        return PreparedContext(docs=working, prompt=prompt, decision=decision)

    while working:
        tokens = _prompt_tokens(
            query,
            working,
            system_rules=system_rules,
            token_counter=token_counter,
        )
        if tokens <= budget:
            break

        # Lowest rank = highest rank number = least relevant retrieved doc.
        target = max(working, key=lambda doc: doc["rank"])
        if len(target["text"]) > MIN_DOC_CHARS + TRUNCATION_STEP_CHARS:
            working = [
                _truncate_doc_from_end(doc, TRUNCATION_STEP_CHARS)
                if doc["rank"] == target["rank"]
                else doc
                for doc in working
            ]
            decision.truncated_docs.append(target["document_id"])
            continue

        if target["rank"] == 1 and len(working) == 1:
            # Last resort: keep rank-1 but trim to minimum useful size.
            working = [_truncate_doc_from_end(target, len(target["text"]) - MIN_DOC_CHARS)]
            decision.truncated_docs.append(target["document_id"])
            break

        working = [doc for doc in working if doc["rank"] != target["rank"]]
        decision.dropped_docs.append(target["document_id"])

    prompt = build_prompt(query, working, system_rules=system_rules)
    decision.prompt_tokens = token_counter.count(prompt)
    decision.final_doc_count = len(working)
    return PreparedContext(docs=working, prompt=prompt, decision=decision)
