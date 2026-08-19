"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.errors import ConfigError

load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got {raw!r}") from exc


def _get_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    groq_api_key: str | None
    gemini_api_key: str | None
    llm_model: str
    embed_model: str
    top_k: int
    temperature: float
    data_path: str
    chroma_path: str
    max_query_chars: int
    log_level: str
    splits_dir: str
    holdout_ratio: float
    split_seed: int
    eval_chroma_path: str
    judge_model: str
    model_8b: str
    model_70b: str
    max_prompt_tokens: int | None
    context_strategy: str
    rewrite_model: str
    max_history_turns: int
    max_history_tokens: int


def get_settings(*, require_groq: bool = False, require_gemini: bool = False) -> Settings:
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip() or None

    if llm_provider == "gemini" or require_gemini:
        if not gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Add your Gemini key to .env or set LLM_PROVIDER=groq."
            )
    if llm_provider == "groq" or require_groq:
        if not groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Add your Groq key to .env or set LLM_PROVIDER=gemini."
            )

    default_model = "gemini-3.6-flash" if llm_provider == "gemini" else "llama-3.3-70b-versatile"
    default_rewrite_model = "gemini-3.6-flash" if llm_provider == "gemini" else "llama-3.1-8b-instant"

    return Settings(
        llm_provider=llm_provider,
        groq_api_key=groq_api_key,
        gemini_api_key=gemini_api_key,
        llm_model=os.getenv("LLM_MODEL", default_model),
        embed_model=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"),
        top_k=_get_int("TOP_K", 3),
        temperature=_get_float("TEMPERATURE", 0.1),
        data_path=os.getenv("DATA_PATH", "data/raw/train_data.json"),
        chroma_path=os.getenv("CHROMA_PATH", "chroma_db"),
        max_query_chars=_get_int("MAX_QUERY_CHARS", 1000),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        splits_dir=os.getenv("SPLITS_DIR", "data/splits"),
        holdout_ratio=_get_float("HOLDOUT_RATIO", 0.2),
        split_seed=_get_int("SPLIT_SEED", 42),
        eval_chroma_path=os.getenv("EVAL_CHROMA_PATH", "chroma_db_train"),
        judge_model=os.getenv("JUDGE_MODEL", "gemini-3.6-flash"),
        model_8b=os.getenv("LLM_MODEL_8B", "llama-3.1-8b-instant"),
        model_70b=os.getenv("LLM_MODEL_70B", "llama-3.3-70b-versatile"),
        max_prompt_tokens=_get_optional_int("MAX_PROMPT_TOKENS"),
        context_strategy=os.getenv("CONTEXT_STRATEGY", "baseline"),
        rewrite_model=os.getenv("REWRITE_MODEL", default_rewrite_model),
        max_history_turns=_get_int("MAX_HISTORY_TURNS", 6),
        max_history_tokens=_get_int("MAX_HISTORY_TOKENS", 800),
    )


def train_split_path(settings: Settings) -> str:
    return os.path.join(settings.splits_dir, "train.json")


def holdout_split_path(settings: Settings) -> str:
    return os.path.join(settings.splits_dir, "holdout.json")


def split_manifest_path(settings: Settings) -> str:
    return os.path.join(settings.splits_dir, "split_manifest.json")
