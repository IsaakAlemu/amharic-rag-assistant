"""Conversation history storage and token-aware truncation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from src.token_counter import TokenCounter

Role = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str
    sources: list[dict] | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatMessage:
        return cls(
            role=data["role"],
            content=data["content"],
            sources=data.get("sources"),
        )


@dataclass
class ConversationState:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[ChatMessage] = field(default_factory=list)

    def clear(self) -> None:
        self.session_id = str(uuid4())
        self.messages = []

    def add_user(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str, *, sources: list[dict] | None = None) -> None:
        self.messages.append(
            ChatMessage(role="assistant", content=content, sources=sources)
        )

    def turn_count(self) -> int:
        return len(self.messages)

    def to_dict_list(self) -> list[dict]:
        return [message.to_dict() for message in self.messages]

    @classmethod
    def from_dict_list(
        cls,
        messages: list[dict],
        *,
        session_id: str | None = None,
    ) -> ConversationState:
        state = cls(session_id=session_id or str(uuid4()))
        state.messages = [ChatMessage.from_dict(item) for item in messages]
        return state


class HistoryManager:
    """Truncate conversation history without affecting retrieval queries."""

    def __init__(
        self,
        *,
        max_turns: int = 6,
        max_history_tokens: int = 800,
    ) -> None:
        self.max_turns = max_turns
        self.max_history_tokens = max_history_tokens

    def select_for_rewrite(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """History used only for query rewriting (excludes the latest user turn)."""
        if not messages:
            return []
        prior = messages[:-1] if messages[-1].role == "user" else messages
        return self._truncate(prior)

    def select_for_prompt(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """History shown in generation prompt (excludes the latest user turn)."""
        if not messages:
            return []
        prior = messages[:-1] if messages[-1].role == "user" else messages
        return self._truncate(prior)

    def _truncate(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not messages:
            return []

        trimmed = messages[-self.max_turns :]
        token_counter = TokenCounter()

        while trimmed:
            text = format_history_for_prompt(trimmed)
            if token_counter.count(text) <= self.max_history_tokens:
                return trimmed
            trimmed = trimmed[1:]

        return trimmed[-1:] if trimmed else []


def format_history_for_prompt(messages: list[ChatMessage]) -> str:
    if not messages:
        return "(No prior conversation.)"

    lines: list[str] = []
    for message in messages:
        label = "User" if message.role == "user" else "Assistant"
        lines.append(f"{label}: {message.content}")
    return "\n".join(lines)


def format_history_for_rewrite(messages: list[ChatMessage]) -> str:
    if not messages:
        return "(No prior conversation.)"

    lines: list[str] = []
    for message in messages:
        role = "User" if message.role == "user" else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)
