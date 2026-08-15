"""Build strict, context-only prompts for the LLM."""

from __future__ import annotations

from src.retriever import RetrievedDocument

REFUSAL_PHRASE = "I don't know based on the provided documents."

SYSTEM_RULES = """You are a strict question-answering assistant. Follow these rules exactly:

1. Answer ONLY using facts explicitly stated in the Context below.
2. Before answering, check: does the Context actually contain information that directly answers the Question? If the Context is about a different topic, person, or time period than what's being asked, it does NOT count as an answer.
3. If the Context does not directly and clearly answer the Question, you MUST respond with exactly: "I don't know based on the provided documents."
4. Do NOT guess, infer, or use any name/fact/number from the Context just because it appears there — it must specifically answer the Question asked.
5. Do NOT use any outside knowledge, even if you know the real-world answer.
6. When you use a fact from the Context, cite the source number inline like [1] or [2]."""

SYSTEM_RULES_COMPRESSED = """You are a strict Amharic QA assistant. Rules:
1. Answer ONLY from the Context below.
2. If the Context does not directly answer the Question, respond exactly: "I don't know based on the provided documents."
3. Do NOT use outside knowledge or guess.
4. Cite sources inline as [1], [2], etc."""

CONVERSATIONAL_RULES = """You are a strict Amharic question-answering assistant in a multi-turn conversation.

Rules:
1. Answer ONLY using facts explicitly stated in the Retrieved Documents below.
2. The Conversation History helps you understand the latest question, but it is NOT a source of facts.
3. Do NOT treat previous assistant messages as factual evidence.
4. Before answering, check whether the Retrieved Documents directly answer the Latest Question.
5. If the Retrieved Documents do not directly and clearly answer the Latest Question, respond exactly: "I don't know based on the provided documents."
6. Do NOT guess, infer, or use outside knowledge.
7. Cite retrieved documents inline as [1], [2], etc."""


def _format_retrieved_documents(retrieved_docs: list[RetrievedDocument]) -> str:
    if not retrieved_docs:
        return "(No relevant documents were retrieved.)\n"

    blocks: list[str] = []
    for doc in retrieved_docs:
        blocks.append(f"[{doc['rank']}] (Document {doc['document_id']})\n{doc['text']}\n")
    return "\n".join(blocks)


def build_prompt(
    query: str,
    retrieved_docs: list[RetrievedDocument],
    *,
    system_rules: str = SYSTEM_RULES,
) -> str:
    prompt = system_rules + "\n\nRetrieved Documents:\n\n"
    prompt += _format_retrieved_documents(retrieved_docs)
    prompt += f"\nQuestion:\n{query}\n\nAnswer:"
    return prompt


def build_conversational_prompt(
    latest_question: str,
    retrieved_docs: list[RetrievedDocument],
    conversation_history: str,
    *,
    system_rules: str = CONVERSATIONAL_RULES,
) -> str:
    prompt = system_rules + "\n\n"
    prompt += "Conversation History (for understanding only — NOT a source of facts):\n"
    prompt += conversation_history + "\n\n"
    prompt += "Retrieved Documents (your ONLY source of facts):\n\n"
    prompt += _format_retrieved_documents(retrieved_docs)
    prompt += f"\nLatest Question:\n{latest_question}\n\nAnswer:"
    return prompt


def build_prompt_parts(
    query: str,
    retrieved_docs: list[RetrievedDocument],
    *,
    system_rules: str = SYSTEM_RULES,
) -> dict[str, str]:
    context = _format_retrieved_documents(retrieved_docs)
    suffix = f"Question:\n{query}\n\nAnswer:"
    return {
        "system_rules": system_rules,
        "context": context,
        "question_suffix": suffix,
        "full": system_rules + "\n\nRetrieved Documents:\n\n" + context + suffix,
    }
