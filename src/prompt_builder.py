"""Build strict, context-only prompts for the LLM."""

from __future__ import annotations

from src.retriever import RetrievedDocument

REFUSAL_PHRASE = "I don't know based on the provided documents."

SYSTEM_RULES = """You are an articulate, helpful, and natural Amharic conversational assistant. Follow these rules carefully:

1. Grounding: Answer strictly using facts supported by the Retrieved Documents below.
2. Tone & Fluency: Express answers in natural, fluent, and clear Amharic. Do not sound stiff or mechanical.
3. Citations: Whenever you state a fact from the documents, place the source rank inline, e.g. [1] or [2].
4. Insufficient Context: If the retrieved documents do not contain enough facts to answer the question, politely decline by stating exactly: "I don't know based on the provided documents." or "ከተሰጡት ሰነዶች በመነሳት ጥያቄውን መመለስ አልተቻለም።"
5. Do NOT speculate, hallucinate, or bring in ungrounded outside facts."""

SYSTEM_RULES_COMPRESSED = """You are a strict, fluent Amharic QA assistant. Rules:
1. Answer ONLY using facts from the Context below in natural Amharic.
2. If Context lacks the answer, respond: "ከተሰጡት ሰነዶች በመነሳት ጥያቄውን መመለስ አልተቻለም።"
3. Do NOT use outside knowledge or hallucinate.
4. Cite sources inline as [1], [2], etc."""

CONVERSATIONAL_RULES = """You are an articulate, fluent Amharic conversational assistant in a multi-turn dialogue.

Rules:
1. Grounding: Answer strictly using facts supported by the Retrieved Documents below.
2. Conversation Flow: Use the Conversation History to understand pronouns, context, and follow-ups naturally.
3. Tone: Speak in clear, warm, and natural Amharic. Avoid sounding overly robotic.
4. Citations: Cite the retrieved documents inline as [1], [2], etc. for every factual statement.
5. Grounded Refusal: If the retrieved documents do not contain the answer to the latest question, state: "ከተሰጡት ሰነዶች በመነሳት ጥያቄውን መመለስ አልተቻለም።" (or "I don't know based on the provided documents.").
6. Do NOT fabricate facts or use ungrounded external assumptions."""


def _format_retrieved_documents(retrieved_docs: list[RetrievedDocument]) -> str:
    if not retrieved_docs:
        return "<retrieved_evidence>\n(No relevant documents were retrieved.)\n</retrieved_evidence>\n"

    blocks: list[str] = ["<retrieved_evidence>"]
    for doc in retrieved_docs:
        blocks.append(f"<document rank=\"{doc['rank']}\" id=\"{doc['document_id']}\">\n{doc['text']}\n</document>")
    blocks.append("</retrieved_evidence>\n")
    return "\n".join(blocks)


def build_prompt(
    query: str,
    retrieved_docs: list[RetrievedDocument],
    *,
    system_rules: str = SYSTEM_RULES,
) -> str:
    prompt = system_rules + "\n\n"
    prompt += _format_retrieved_documents(retrieved_docs)
    prompt += f"\n<user_question>\n{query}\n</user_question>\n\nAnswer:"
    return prompt


def build_conversational_prompt(
    latest_question: str,
    retrieved_docs: list[RetrievedDocument],
    conversation_history: str,
    *,
    system_rules: str = CONVERSATIONAL_RULES,
) -> str:
    prompt = system_rules + "\n\n"
    prompt += f"<conversation_history>\n{conversation_history}\n</conversation_history>\n\n"
    prompt += _format_retrieved_documents(retrieved_docs)
    prompt += f"\n<user_question>\n{latest_question}\n</user_question>\n\nAnswer:"
    return prompt


def build_prompt_parts(
    query: str,
    retrieved_docs: list[RetrievedDocument],
    *,
    system_rules: str = SYSTEM_RULES,
) -> dict[str, str]:
    context = _format_retrieved_documents(retrieved_docs)
    suffix = f"\n<user_question>\n{query}\n</user_question>\n\nAnswer:"
    return {
        "system_rules": system_rules,
        "context": context,
        "question_suffix": suffix,
        "full": system_rules + "\n\n" + context + suffix,
    }
