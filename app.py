import html
import re

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

import time

from config import get_settings
from src.errors import ConfigError
from src.history_manager import ConversationState
from src.llm import AMHARIC_REFUSAL_PHRASE, REFUSAL_PHRASE, generate_answer_stream, is_refusal
from src.logging_config import setup_logging
from src.pipeline import (
    answer_conversation,
    complete_conversation_turn,
    load_vector_collection,
    prepare_conversation_context,
)

# ── Configuration ────────────────────────────────────────────────────────────

try:
    settings = get_settings()
    setup_logging(settings.log_level)
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

st.set_page_config(page_title="Amharic RAG Chat", page_icon="💬", layout="wide")

# ── Helper: Format message text with styled citation badges ─────────────────

def render_message_html(content: str, is_assistant: bool = False) -> str:
    escaped = html.escape(content)
    if is_assistant:
        escaped = re.sub(
            r"\[(\d+)\]",
            r'<span class="am-inline-cite">\1</span>',
            escaped,
        )
    return escaped


# ── CSS — scoped, icon-safe ──────────────────────────────────────────────────
# RULES:
#  1. Never target body, div, p, or [class*="st-"] for font-family.
#  2. Always restore Material Symbols font explicitly.
#  3. All custom classes use the .am- prefix so they're identifiable.
st.markdown(
    """
    <style>
    /* ── Layout: center the chat column at 820px on all viewports ─────────── */
    .main .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    .stMainBlockContainer {
        max-width: 820px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 4rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* ── Constrain chat messages and bottom input bar ──────────────────────── */
    [data-testid="stChatMessage"] {
        max-width: 820px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    [data-testid="stBottom"] > div,
    [data-testid="stChatInput"] {
        max-width: 820px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* ── Compact header row ────────────────────────────────────────────────── */
    .am-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128,128,128,0.15);
        margin-bottom: 0.9rem;
    }
    .am-header-left {
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .am-header-right {
        font-size: 0.78rem;
        color: #6b7280;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── Amharic text in chat messages — SCOPED, never global ─────────────── */
    .am-msg-text {
        font-family: 'Noto Sans Ethiopic', 'Nyala', 'Abyssinica SIL', 'Segoe UI', sans-serif;
        font-size: 1.02rem;
        line-height: 1.68;
    }

    /* ── Inline citation badge in assistant answers ────────────────────────── */
    .am-inline-cite {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 600;
        color: #4f46e5;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 4px;
        padding: 0.05rem 0.32rem;
        margin: 0 0.15rem;
        vertical-align: 0.12em;
        line-height: 1;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        user-select: none;
    }

    /* ── Grounded refusal card ─────────────────────────────────────────────── */
    .am-refusal {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        background: rgba(245, 158, 11, 0.07);
        border-left: 3px solid #f59e0b;
        padding: 0.7rem 0.9rem;
        border-radius: 6px;
    }
    .am-refusal-icon {
        font-size: 1rem;
        margin-top: 0.05rem;
        flex-shrink: 0;
    }
    .am-refusal-body {}
    .am-refusal-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #b45309;
        margin-bottom: 0.2rem;
    }
    .am-refusal-text {
        font-family: 'Noto Sans Ethiopic', 'Nyala', 'Abyssinica SIL', 'Segoe UI', sans-serif;
        font-size: 0.97rem;
        line-height: 1.5;
        color: inherit;
        margin-bottom: 0.2rem;
    }
    .am-refusal-desc {
        font-size: 0.75rem;
        color: #9ca3af;
        line-height: 1.35;
    }

    /* ── Compact source chips row ──────────────────────────────────── */
    .am-source-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.55rem;
    }
    .am-source-chip {
        display: inline-block;
        font-size: 0.72rem;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #4f46e5;
        cursor: default;
        letter-spacing: 0.01em;
    }

    /* ── Quick-start chips (empty state) ────────────────────────────────────── */
    .am-qs-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.45rem;
        margin-top: 1.6rem;
    }
    .am-qs-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.6rem;
    }

    /* ── Sidebar ────────────────────────────────────────────────────────────── */
    .am-sidebar-app-name {
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.05rem;
    }
    .am-sidebar-tagline {
        font-size: 0.72rem;
        color: #9ca3af;
        margin-bottom: 1rem;
    }

    /* ── CRITICAL: Protect Material Symbols ligature font from any override ── */
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    [class*="material-symbol"],
    [class*="material-icon"],
    [data-testid*="Icon"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline loader (cached) ─────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    if settings.llm_provider == "gemini":
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
    else:
        client = Groq(api_key=settings.groq_api_key)
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    return client, embed_model, collection


try:
    client, embed_model, collection = load_pipeline()
except Exception as exc:
    st.error(f"Failed to load the RAG pipeline: {exc}")
    st.stop()

# ── Session state ────────────────────────────────────────────────────────────

MAX_SESSION_TURNS = 12

if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationState()
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True
if "last_error" not in st.session_state:
    st.session_state.last_error = None
# Per-message run metrics: dict[int, dict] keyed by message index (assistant messages only)
if "message_metrics" not in st.session_state:
    st.session_state.message_metrics = {}

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="am-sidebar-app-name">💬 Amharic RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="am-sidebar-tagline">Grounded Conversational QA · AmQA Knowledge Base</div>', unsafe_allow_html=True)

    # Session turn counter and limit guard
    user_turns = sum(1 for m in st.session_state.conversation.messages if m.role == "user")
    turns_left = max(0, MAX_SESSION_TURNS - user_turns)
    
    st.caption(f"🎯 **Model Provider:** `{settings.llm_provider.upper()}` (`{settings.llm_model}`)")
    st.caption(f"💬 **Turns Used:** `{user_turns} / {MAX_SESSION_TURNS}`")

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.conversation = ConversationState()
        st.session_state.last_error = None
        st.session_state.message_metrics = {}
        st.rerun()

    st.session_state.show_sources = st.checkbox(
        "Show retrieved sources",
        value=st.session_state.show_sources,
    )

    st.divider()

    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown(
            f"""
**Purpose**  
Grounded, conversational Amharic Question Answering over a verified AmQA Wikipedia knowledge base.

**Corpus Details**  
- **Knowledge Base:** ~286 passage-level documents from AmQA.
- **Embeddings:** `intfloat/multilingual-e5-small` dense vector representations.
- **Storage:** ChromaDB local vector collection (`top_k={settings.top_k}`).

**Architecture Flow**
```
User Question (Amharic)
↓
Gemini 3.6 Flash / LLaMA-8B (Query Rewriter)
↓
Multilingual-E5 Dense Retrieval (ChromaDB)
↓
Top-3 Evidence Filtering
↓
Gemini 3.6 Flash / LLaMA-70B (Grounded Generator)
↓
Inline Citation Parsing & Fact Validation
```

**Retrieval Benchmark (329-Q Holdout Set)**  
- **Hit@1:** 72.64%  
- **Hit@3:** 83.89%  
- **MRR (Mean Reciprocal Rank):** 0.778  
*(Note: Retrieval metrics reflect the dense vector search performance on the 329-question holdout benchmark across all LLM providers.)*

**Grounding & Anti-Hallucination**  
Answers are strictly bounded by retrieved evidence. When knowledge is missing, the system outputs an explicit refusal rather than hallucinating facts.
            """
        )

    st.markdown(
        f"<div style='font-size:0.7rem;color:#9ca3af;padding-top:0.4rem;'>"
        f"Session ID: <code>{st.session_state.conversation.session_id[:8]}</code></div>",
        unsafe_allow_html=True,
    )

# ── Compact header (main area) ───────────────────────────────────────────────

st.markdown(
    """
    <div class="am-header">
        <span class="am-header-left">💬 Amharic RAG Assistant</span>
        <span class="am-header-right">Amharic · AmQA Corpus · Grounded</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Error banner ─────────────────────────────────────────────────────────────

if st.session_state.last_error:
    st.error(f"⚠️ {st.session_state.last_error}")

# ── Conversation messages ─────────────────────────────────────────────────────
# Each assistant message may have:
#   message.sources  →  list[dict] with keys: rank, document_id, distance, text
#   message_metrics[idx] → run details stored when the message was produced

messages = st.session_state.conversation.messages

for idx, message in enumerate(messages):
    with st.chat_message(message.role):

        # ── Message content ──
        if message.role == "assistant" and is_refusal(message.content):
            # Grounded refusal — visually intentional, not an error
            st.markdown(
                f"""
                <div class="am-refusal">
                    <div class="am-refusal-icon">🛡️</div>
                    <div class="am-refusal-body">
                        <div class="am-refusal-label">Grounded Refusal · የተረጋገጠ እምቢታ</div>
                        <div class="am-refusal-text">{html.escape(message.content)}</div>
                        <div class="am-refusal-desc">የተገኘው መረጃ ለጥያቄው በቂ ማስረጃ አልያዘም። ሞዴሉ ከራሱ ከመገመት ይልቅ በማስረጃ ላይ ተመስርቶ ምላሽ ሰጥቷል።<br>(The retrieved corpus did not contain sufficient evidence. The system strictly refused to speculate.)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Normal message — Amharic typography with styled citation badges
            formatted_text = render_message_html(
                message.content,
                is_assistant=(message.role == "assistant"),
            )
            st.markdown(
                f'<div class="am-msg-text">{formatted_text}</div>',
                unsafe_allow_html=True,
            )

        # ── Source chips + passage expander (assistant only) ──
        if message.role == "assistant" and message.sources and st.session_state.show_sources:
            sources = message.sources

            # Compact chip row
            chips_html = '<div class="am-source-chips">' + "".join(
                f'<span class="am-source-chip">#{s["rank"]} · Doc {s["document_id"]}</span>'
                for s in sources
            ) + "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)

            # Full passages in a collapsed expander
            with st.expander(f"Retrieved passages ({len(sources)})", expanded=False):
                for source in sources:
                    st.markdown(
                        f"**\\[{source['rank']}\\] Document `{source['document_id']}`** "
                        f"— distance `{source['distance']:.4f}`"
                    )
                    st.markdown(
                        f'<div class="am-msg-text">{html.escape(source["text"])}</div>',
                        unsafe_allow_html=True,
                    )
                    st.divider()

        # ── Per-message run details (collapsed by default) ──
        if message.role == "assistant" and idx in st.session_state.message_metrics:
            m = st.session_state.message_metrics[idx]
            with st.expander("Run details", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", f"{m.get('total_s', 0):.2f}s")
                c2.metric("Rewrite", f"{m.get('rewrite_s', 0):.2f}s")
                c3.metric("Retrieve", f"{m.get('retrieve_s', 0):.2f}s")
                c4.metric("Generate", f"{m.get('generate_s', 0):.2f}s")

                t1, t2 = st.columns(2)
                t1.metric("Prompt tokens", m.get("prompt_tokens") or "—")
                t2.metric("Completion tokens", m.get("completion_tokens") or "—")

                if m.get("rewritten_query"):
                    st.caption(f"Retrieval query: {m['rewritten_query']}")

                citations = m.get("citations") or []
                if citations:
                    cit_str = "  ".join(
                        f"[{c['rank']}]→`{c['document_id']}`" if c.get("valid")
                        else f"[{c['rank']}] invalid"
                        for c in citations
                    )
                    st.caption(f"Citations: {cit_str}")

# ── Quick-start chips (empty state only) ─────────────────────────────────────

prompt_to_run = None

SAMPLES = [
    {
        "chip": "Factual QA",
        "query": "ለተባበሩት መንግሥታት ድርጅት የደቡብ ሱዳን ሰላም ማስከበር ማን ተሾመ?",
        "hint": "Who was appointed to the UN South Sudan peacekeeping mission?",
    },
    {
        "chip": "Conversational follow-up",
        "query": "እሱ ከዚህ በፊት የት ሠርተዋል?",
        "hint": "Where did he work before? (tests 8B pronoun rewriting)",
    },
    {
        "chip": "Symposium fact",
        "query": "ኢትዮጵያ የአስትሮኖሚካል ሲምፖዚየም ያዘጋጀች ስንተኛዋ አፍሪካዊ ሀገር ናት?",
        "hint": "Which rank African country hosted the astronomy symposium?",
    },
    {
        "chip": "Test refusal",
        "query": "የፈረንሳይ ዋና ከተማ ማን ናት?",
        "hint": "Capital of France? (out-of-corpus — tests grounded refusal)",
    },
]

if not messages:
    st.markdown('<div class="am-qs-label">Try a sample question</div>', unsafe_allow_html=True)

    # Render as small inline buttons in a flex row via columns trick
    cols = st.columns(len(SAMPLES))
    for i, sample in enumerate(SAMPLES):
        with cols[i]:
            if st.button(
                sample["chip"],
                key=f"qs_{i}",
                help=f"{sample['hint']}\n\n{sample['query']}",
                use_container_width=True,
            ):
                prompt_to_run = sample["query"]

# ── Chat input ────────────────────────────────────────────────────────────────

user_turns = sum(1 for m in st.session_state.conversation.messages if m.role == "user")
if user_turns >= MAX_SESSION_TURNS:
    st.info("💡 You have reached the conversation turn limit for this session. Please click **'＋ New conversation'** in the sidebar to start a new chat.")
    user_input = None
else:
    user_input = st.chat_input("Ask a question in Amharic…")

if user_input:
    prompt_to_run = user_input

# ── Run pipeline (Real-time Streamed Generation) ───────────────────────────────

if prompt_to_run:
    # 1. Render immediate user message bubble in UI
    with st.chat_message("user"):
        st.markdown(
            f'<div class="am-msg-text">{html.escape(prompt_to_run)}</div>',
            unsafe_allow_html=True,
        )

    # 2. Prepare Context (Rewriting + Dense Retrieval + Prompt Assembly)
    with st.spinner("Retrieving context from AmQA knowledge base…"):
        ctx, early_result = prepare_conversation_context(
            prompt_to_run,
            st.session_state.conversation,
            client=client,
            embed_model=embed_model,
            collection=collection,
            settings=settings,
        )

    if early_result is not None:
        if early_result.error:
            st.session_state.last_error = early_result.error
        else:
            st.session_state.last_error = None
            asst_idx = len(st.session_state.conversation.messages) - 1
            st.session_state.message_metrics[asst_idx] = {
                "rewritten_query": early_result.rewritten_query,
                "rewrite_s": early_result.timings_ms.get("rewrite", 0) / 1000,
                "retrieve_s": early_result.timings_ms.get("retrieve", 0) / 1000,
                "generate_s": 0.0,
                "total_s": early_result.timings_ms.get("total", 0) / 1000,
                "prompt_tokens": None,
                "completion_tokens": None,
                "citations": [],
            }
        st.rerun()

    # 3. Streamed Generation into Assistant Message Box
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        t_gen_start = time.perf_counter()

        try:
            stream = generate_answer_stream(
                ctx.prompt,
                client,
                model=settings.llm_model,
                temperature=settings.temperature,
            )
            for chunk in stream:
                full_response += chunk
                formatted_live = render_message_html(full_response, is_assistant=True)
                message_placeholder.markdown(
                    f'<div class="am-msg-text">{formatted_live} ▌</div>',
                    unsafe_allow_html=True,
                )

            # Final render without cursor
            formatted_final = render_message_html(full_response, is_assistant=True)
            message_placeholder.markdown(
                f'<div class="am-msg-text">{formatted_final}</div>',
                unsafe_allow_html=True,
            )
            gen_ms = (time.perf_counter() - t_gen_start) * 1000

            # 4. Finalize turn and compute citation mappings & metrics
            result = complete_conversation_turn(
                ctx,
                full_response,
                st.session_state.conversation,
                model_name=settings.llm_model,
                generate_ms=gen_ms,
            )
            st.session_state.last_error = None
            asst_idx = len(st.session_state.conversation.messages) - 1
            st.session_state.message_metrics[asst_idx] = {
                "rewritten_query": result.rewritten_query,
                "rewrite_s": result.timings_ms.get("rewrite", 0) / 1000,
                "retrieve_s": result.timings_ms.get("retrieve", 0) / 1000,
                "generate_s": result.timings_ms.get("generate", 0) / 1000,
                "total_s": result.timings_ms.get("total", 0) / 1000,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "citations": result.citations,
            }

        except Exception as exc:
            # Clean up user message on failure
            if st.session_state.conversation.messages and st.session_state.conversation.messages[-1].role == "user":
                st.session_state.conversation.messages.pop()
            st.session_state.last_error = str(exc)

    st.rerun()
