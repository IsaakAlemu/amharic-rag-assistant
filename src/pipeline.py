"""RAG pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from groq import Groq

from config import Settings, train_split_path
from src.history_manager import ConversationState, HistoryManager, format_history_for_prompt
from src.input_validation import validate_query
from src.citations import CitationValidation, build_citation_metadata
from src.llm import (
    REFUSAL_PHRASE,
    GenerationResult,
    extract_rate_limit_details,
    generate_answer,
)
from src.logging_config import get_logger, log_event
from src.prompt_builder import build_conversational_prompt, build_prompt
from src.query_rewriter import rewrite_query
from src.retriever import RetrievedDocument, retrieve
from src.token_counter import TokenCounter

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    answer: str
    sources: list[RetrievedDocument]
    timings_ms: dict[str, float] = field(default_factory=dict)
    refusal: bool = False
    skipped_generation: bool = False
    model: str = ""
    error: str | None = None


@dataclass
class ConversationPipelineResult(PipelineResult):
    rewritten_query: str = ""
    retrieval_query: str = ""
    prompt_tokens_estimated: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    rewrite_model: str = ""
    history_turns_used: int = 0
    citations: list[CitationValidation] = field(default_factory=list)
    rate_limit_info: dict[str, Any] | None = None


def load_vector_collection(settings: Settings, embed_model):
    from src.document_loader import load_documents
    from src.embedding_generator import generate_embeddings

    documents = load_documents(settings.data_path)
    if not documents:
        raise RuntimeError(
            f"No documents loaded from {settings.data_path}. "
            "Check that the data file exists and is valid."
        )

    collection = generate_embeddings(
        documents,
        embed_model,
        persist_path=settings.chroma_path,
    )

    if collection.count() == 0:
        raise RuntimeError(
            f"Chroma collection is empty at {settings.chroma_path}. "
            "Run the embedding step to build the index."
        )

    return collection


def load_eval_collection(settings: Settings, embed_model):
    """Load a train-only vector index for hold-out evaluation."""
    from src.document_loader import load_documents
    from src.embedding_generator import generate_embeddings

    data_path = train_split_path(settings)
    documents = load_documents(data_path)
    if not documents:
        raise RuntimeError(
            f"No documents loaded from {data_path}. "
            "Run scripts/split_dataset.py first."
        )

    collection = generate_embeddings(
        documents,
        embed_model,
        persist_path=settings.eval_chroma_path,
    )

    if collection.count() == 0:
        raise RuntimeError(
            f"Eval Chroma collection is empty at {settings.eval_chroma_path}."
        )

    return collection


def answer_question(
    query: str,
    *,
    client: Groq,
    embed_model,
    collection,
    settings: Settings,
) -> PipelineResult:
    t_start = time.perf_counter()

    try:
        cleaned_query = validate_query(query, max_chars=settings.max_query_chars)
    except Exception as exc:
        return PipelineResult(
            answer="",
            sources=[],
            error=str(exc),
            timings_ms={"total": (time.perf_counter() - t_start) * 1000},
        )

    t_retrieve_start = time.perf_counter()
    sources = retrieve(cleaned_query, collection, embed_model, top_k=settings.top_k)
    retrieve_ms = (time.perf_counter() - t_retrieve_start) * 1000

    log_event(
        logger,
        "retrieval_complete",
        query=cleaned_query,
        doc_ids=[source["document_id"] for source in sources],
        distances=[source["distance"] for source in sources],
        latency_ms={"retrieve": round(retrieve_ms, 2)},
    )

    if not sources:
        total_ms = (time.perf_counter() - t_start) * 1000
        return PipelineResult(
            answer=REFUSAL_PHRASE,
            sources=[],
            refusal=True,
            skipped_generation=True,
            timings_ms={"retrieve": retrieve_ms, "total": total_ms},
        )

    prompt = build_prompt(cleaned_query, sources)

    t_generate_start = time.perf_counter()
    try:
        generation: GenerationResult = generate_answer(
            prompt,
            client,
            model=settings.llm_model,
            temperature=settings.temperature,
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000
        return PipelineResult(
            answer="",
            sources=sources,
            error=str(exc),
            timings_ms={
                "retrieve": retrieve_ms,
                "generate": (time.perf_counter() - t_generate_start) * 1000,
                "total": total_ms,
            },
        )
    generate_ms = (time.perf_counter() - t_generate_start) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000

    refusal = REFUSAL_PHRASE in generation.text

    log_event(
        logger,
        "generation_complete",
        query=cleaned_query,
        model=generation.model,
        refusal=refusal,
        latency_ms={
            "retrieve": round(retrieve_ms, 2),
            "generate": round(generate_ms, 2),
            "total": round(total_ms, 2),
        },
    )

    return PipelineResult(
        answer=generation.text,
        sources=sources,
        refusal=refusal,
        model=generation.model,
        timings_ms={
            "retrieve": retrieve_ms,
            "generate": generate_ms,
            "total": total_ms,
        },
    )


def answer_conversation(
    user_message: str,
    conversation: ConversationState,
    *,
    client: Groq,
    embed_model,
    collection,
    settings: Settings,
) -> ConversationPipelineResult:
    """
    Multi-turn RAG:
    - rewrite follow-ups into standalone retrieval queries
    - retrieve ONLY on rewritten query (history never enters retriever)
    - keep history separate from retrieved documents in the prompt
    """
    t_start = time.perf_counter()
    token_counter = TokenCounter()
    history_manager = HistoryManager(
        max_turns=settings.max_history_turns,
        max_history_tokens=settings.max_history_tokens,
    )

    try:
        cleaned_query = validate_query(user_message, max_chars=settings.max_query_chars)
    except Exception as exc:
        return ConversationPipelineResult(
            answer="",
            sources=[],
            error=str(exc),
            timings_ms={"total": (time.perf_counter() - t_start) * 1000},
        )

    conversation.add_user(cleaned_query)
    rewrite_history = history_manager.select_for_rewrite(conversation.messages)
    prompt_history = history_manager.select_for_prompt(conversation.messages)

    t_rewrite_start = time.perf_counter()
    rewrite = rewrite_query(
        cleaned_query,
        rewrite_history,
        client=client,
        model=settings.rewrite_model,
        temperature=0.0,
    )
    rewrite_ms = (time.perf_counter() - t_rewrite_start) * 1000

    t_retrieve_start = time.perf_counter()
    sources = retrieve(
        rewrite.rewritten_query,
        collection,
        embed_model,
        top_k=settings.top_k,
    )
    retrieve_ms = (time.perf_counter() - t_retrieve_start) * 1000

    log_event(
        logger,
        "conversation_retrieval_complete",
        query_original=cleaned_query,
        query_rewritten=rewrite.rewritten_query,
        doc_ids=[source["document_id"] for source in sources],
        distances=[source["distance"] for source in sources],
        latency_ms={"rewrite": round(rewrite_ms, 2), "retrieve": round(retrieve_ms, 2)},
    )

    if not sources:
        total_ms = (time.perf_counter() - t_start) * 1000
        conversation.add_assistant(REFUSAL_PHRASE)
        return ConversationPipelineResult(
            answer=REFUSAL_PHRASE,
            sources=[],
            refusal=True,
            skipped_generation=True,
            rewritten_query=rewrite.rewritten_query,
            retrieval_query=rewrite.rewritten_query,
            rewrite_model=rewrite.model,
            history_turns_used=len(prompt_history),
            timings_ms={
                "rewrite": rewrite_ms,
                "retrieve": retrieve_ms,
                "total": total_ms,
            },
        )

    history_text = format_history_for_prompt(prompt_history)
    prompt = build_conversational_prompt(
        cleaned_query,
        sources,
        history_text,
    )
    prompt_tokens_estimated = token_counter.count(prompt)

    t_generate_start = time.perf_counter()
    try:
        generation: GenerationResult = generate_answer(
            prompt,
            client,
            model=settings.llm_model,
            temperature=settings.temperature,
        )
    except Exception as exc:
        conversation.messages.pop()
        total_ms = (time.perf_counter() - t_start) * 1000
        return ConversationPipelineResult(
            answer="",
            sources=sources,
            error=str(exc),
            rewritten_query=rewrite.rewritten_query,
            retrieval_query=rewrite.rewritten_query,
            rewrite_model=rewrite.model,
            history_turns_used=len(prompt_history),
            prompt_tokens_estimated=prompt_tokens_estimated,
            rate_limit_info=extract_rate_limit_details(exc),
            timings_ms={
                "rewrite": rewrite_ms,
                "retrieve": retrieve_ms,
                "generate": (time.perf_counter() - t_generate_start) * 1000,
                "total": total_ms,
            },
        )
    generate_ms = (time.perf_counter() - t_generate_start) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000

    refusal = REFUSAL_PHRASE in generation.text
    citations = build_citation_metadata(generation.text, sources)
    conversation.add_assistant(
        generation.text,
        sources=[dict(source) for source in sources],
    )

    log_event(
        logger,
        "conversation_generation_complete",
        query_original=cleaned_query,
        query_rewritten=rewrite.rewritten_query,
        model=generation.model,
        refusal=refusal,
        prompt_tokens_estimated=prompt_tokens_estimated,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        citation_ranks=[item["rank"] for item in citations],
        latency_ms={
            "rewrite": round(rewrite_ms, 2),
            "retrieve": round(retrieve_ms, 2),
            "generate": round(generate_ms, 2),
            "total": round(total_ms, 2),
        },
    )

    return ConversationPipelineResult(
        answer=generation.text,
        sources=sources,
        refusal=refusal,
        model=generation.model,
        rewritten_query=rewrite.rewritten_query,
        retrieval_query=rewrite.rewritten_query,
        rewrite_model=rewrite.model,
        history_turns_used=len(prompt_history),
        prompt_tokens_estimated=prompt_tokens_estimated,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        total_tokens=generation.total_tokens,
        citations=citations,
        timings_ms={
            "rewrite": rewrite_ms,
            "retrieve": retrieve_ms,
            "generate": generate_ms,
            "total": total_ms,
        },
    )
