"""Typed exceptions for the RAG pipeline."""


class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""


class ConfigError(RAGError):
    """Missing or invalid configuration."""


class ValidationError(RAGError):
    """Invalid user input."""


class RetrievalError(RAGError):
    """Vector store or retrieval failure."""


class GenerationError(RAGError):
    """LLM API failure."""


class TokenBudgetError(RAGError):
    """Prompt exceeds token budget even after truncation."""
