# Conversational Amharic RAG — Evidence-Grounded Question Answering

An end-to-end **conversational Retrieval-Augmented Generation (RAG)** system for answering Amharic questions from a fixed AmQA-style Wikipedia corpus. This is an **AI Engineering portfolio / academic project** — a working question-answering application with multi-turn chat, grounded generation, and citations — not a retrieval benchmark repository.

---

## Demo

**Empty state — chat-first interface with sample questions**

![Amharic RAG empty state](docs/images/empty-state.png)

**A grounded answer with inline citations and sources**

![Amharic RAG grounded answer with citations](docs/images/qa-example.png)

**Grounded refusal on an out-of-scope question**

![Amharic RAG grounded refusal](docs/images/refusal-example.png)

---

## 1. Project Overview

The system lets users ask questions in Amharic through a Streamlit chat interface. Each turn:

1. Validates the user query
2. Rewrites follow-up questions into standalone retrieval queries using conversation history
3. Retrieves relevant paragraphs with dense embeddings
4. Generates an answer strictly from retrieved evidence
5. Returns inline citations, source passages, and latency/token observability

Knowledge is limited to the project corpus (~286 paragraph-level documents). The model is instructed to refuse when retrieved context does not support an answer.

---

## 2. What the System Does

| Capability | Description |
|------------|-------------|
| **Amharic QA** | Users ask questions in Amharic about topics covered in the corpus |
| **Multi-turn chat** | Follow-up questions use prior conversation context via query rewriting |
| **Evidence grounding** | Answers must come from retrieved documents, not outside knowledge |
| **Citations** | Answers cite sources inline as `[1]`, `[2]`, etc., mapped to document IDs |
| **Refusal behavior** | Returns a fixed refusal phrase when context is insufficient or retrieval is empty |

The system does **not** perform live web search, cover topics outside the corpus, or guarantee factual correctness beyond what retrieval and prompting enforce.

---

## 3. Architecture

### Production path

```
Streamlit UI
  ↓
app.py
  ↓
answer_conversation()          [src/pipeline.py]
  ↓
query validation               [src/input_validation.py]
  ↓
conversation history           [src/history_manager.py]
  ↓
8B query rewriting             [src/query_rewriter.py]
  ↓
multilingual-e5-small          [SentenceTransformer + src/embedding_generator.py]
  ↓
Chroma top-k retrieval         [src/retriever.py]
  ↓
conversational prompt          [src/prompt_builder.py]
  ↓
70B generation (Groq)          [src/llm.py]
  ↓
citation parsing/validation    [src/citations.py]
  ↓
answer + sources + observability
```

**Design rules:**

- **Retrieval uses the rewritten query only** — conversation history is never embedded or passed to the retriever.
- **Generation sees history for understanding**, but the prompt explicitly forbids treating chat history as a source of facts.
- **Empty retrieval skips generation** and returns the refusal phrase immediately.

### Production vs evaluation

| Production (runtime) | Evaluation / experimental (not in live app) |
|----------------------|-----------------------------------------------|
| `app.py`, `config.py` | `scripts/*` |
| `src/pipeline.py` (`answer_conversation`) | `src/context_manager.py`, `src/query_normalization.py` |
| `src/history_manager.py`, `src/query_rewriter.py` | `src/eval_utils.py`, `src/eval_api.py` |
| `src/retriever.py`, `src/embedding_generator.py` | `data/splits/`, `data/eval/` |
| `src/prompt_builder.py`, `src/llm.py`, `src/citations.py` | `results/*` (eval artifacts) |
| `data/raw/train_data.json` | `chroma_db_train/` (train-only eval index) |

`main.py` provides a **legacy single-turn CLI** via `answer_question()` — it does not use the conversational rewrite path.

---

## 4. Models and Components

| Component | Value |
|-----------|-------|
| **Rewrite model** | `llama-3.1-8b-instant` (`REWRITE_MODEL`), temperature `0.0` in pipeline |
| **Generation model** | `llama-3.3-70b-versatile` (`LLM_MODEL`), temperature `0.1` |
| **LLM provider** | Groq (`GROQ_API_KEY` required) |
| **Embeddings** | `intfloat/multilingual-e5-small` |
| **E5 query prefix** | `query: ` |
| **E5 passage prefix** | `passage: ` |
| **Vector store** | Chroma (persistent), collection `amqa`, path `chroma_db/` |
| **Similarity** | Cosine distance on L2-normalized embeddings |
| **Retrieval** | `top_k = 3` |
| **History turns** | `MAX_HISTORY_TURNS = 6` |
| **History token budget** | `MAX_HISTORY_TOKENS = 800` |
| **Query length limit** | `MAX_QUERY_CHARS = 1000` |
| **Corpus size** | ~286 paragraph-level documents |
| **Refusal phrase** | `"I don't know based on the provided documents."` |

---

## 5. User-Facing Features

- Amharic question answering
- Multi-turn conversation with session state
- Conversational query rewriting for follow-ups and pronoun resolution
- Retrieved source display (rank, document ID, distance, passage text)
- Inline citations (`[1]`, `[2]`, …) with citation-to-document mapping
- Grounded refusal when retrieval is empty or context is insufficient
- **New conversation** button to reset session

**Not implemented** (do not expect these in the app):

- Authentication or multi-user database-backed sessions
- Live web search
- Production reranking
- Query normalization
- Automatic out-of-scope classifier
- LLM-as-judge scoring inside the UI
- Conversation export or share

---

## 6. UI and Observability

The Streamlit sidebar and chat interface expose per-turn diagnostics:

| Signal | Description |
|--------|-------------|
| Rewritten query | Standalone query sent to the retriever |
| Rewrite latency | Time for 8B query rewrite |
| Retrieve latency | Time for embedding + Chroma search |
| Generate latency | Time for 70B answer generation |
| Total latency | End-to-end turn time |
| Prompt tokens | Groq-reported prompt token count |
| Completion tokens | Groq-reported completion token count |
| Citations | Parsed ranks mapped to document IDs (valid/invalid) |

Structured JSON logging is emitted from the pipeline for debugging (`src/logging_config.py`).

---

## 7. Corpus and Retrieval

- **Source:** AmQA-style JSON at `data/raw/train_data.json`
- **Granularity:** One vector per paragraph (`document_id` + `context` text)
- **Indexing:** On first run, if `chroma_db/` is empty, all paragraphs are embedded and stored in Chroma (skip-if-already-embedded check in `src/embedding_generator.py`)
- **Retrieval quality:** Useful but imperfect — see [Evaluation](#8-evaluation) and [Limitations](#15-limitations)

Query normalization (Amharic character/spacing rules) was tested in evaluation and **rejected** because it reduced retrieval performance.

---

## 8. Evaluation

Evaluation artifacts live in `results/` (mostly gitignored; `results/eval_summary.json` is whitelisted). Metrics below are from **completed, valid runs** — reported for transparency, not as product guarantees.

**Canonical retrieval source:** `results/topk_sweep.json` (controlled 329-question holdout top-k sweep). The committed summary `results/eval_summary.json` now agrees with that sweep.

**Production retrieval (`TOP_K=3`):** Hit@1 = **72.95%**, Hit@3 = **84.19%**, MRR ≈ **0.781**.

**Sweep-only metrics (not production `TOP_K=3`):** Hit@5 = **87.23%** and Hit@10 = **89.36%** come from retrieving **5** and **10** documents respectively in the top-k sweep — not from the production setting of 3.

### 8.1 Single-turn retrieval (329-question holdout)

Full corpus index, holdout questions from `data/splits/holdout.json`, E5 + Chroma cosine. See `results/topk_sweep.json` for full detail.

| Metric | Value | Notes |
|--------|-------|-------|
| Hit@1 | 72.95% | Production `TOP_K=3` |
| Hit@3 | 84.19% | Production `TOP_K=3` |
| Hit@5 | 87.23% | Sweep with `top_k=5` (not production) |
| Hit@10 | 89.36% | Sweep with `top_k=10` (not production) |
| MRR | 0.781 | Production `TOP_K=3` |

Retrieval is useful but imperfect: roughly one in four questions misses the gold document at rank 1. Query normalization was tested and rejected because it hurt these metrics.

### 8.2 Conversational retrieval (corrected, scenario-based)

Rewrite (8B) + retrieval only — **no 70B generation**. Small scenario set (`results/conversation_retrieval_eval_p1.json`).

| Metric | Value |
|--------|-------|
| Scored retrieval turns | 18 |
| Follow-up turns | 8 |
| Hit@1 | 100% |
| Hit@3 | 100% |
| MRR | 1.000 |
| Follow-up Hit@1 | 100% |
| Rewrite success | 100% |

**Important:** This is a **small scenario-based evaluation** on curated follow-up patterns. It demonstrates that rewriting helps retrieval on those scenarios; it must **not** be presented as proof of general 100% conversational retrieval accuracy.

### 8.3 Rewrite-only evaluation

8B rewrite model only — no retrieval, no 70B (`results/rewrite_eval.json`). Ten follow-up turns.

| Metric | Value |
|--------|-------|
| Rewrite success | 100% |
| Meaning preservation | 100% |
| Entity/topic preservation | 100% |
| Pronoun resolution | 100% |
| Irrelevant-history avoidance | 100% |
| Standalone pass-through accuracy | 100% |
| Out-of-scope pass-through accuracy | 100% |
| API errors | 0 |

### 8.4 Generation pilot (10 questions)

Bounded end-to-end pilot (`results/phase5_generation_pilot_10.json`). **Pilot only — not a general accuracy benchmark.**

| Item | Value |
|------|-------|
| Completed | 10 / 10 |
| Avg generation latency | ~31 s |
| Avg prompt tokens | ~7,004 |
| Avg completion tokens | ~79 |
| Rate-limit errors | 0 |
| Manual/heuristic review | 7 clearly supported, 1 partially supported, 2 refusals, 0 clearly unsupported |

There is **no sufficiently broad independent LLM-judge evaluation** behind these labels. Do **not** convert this into a single "generation accuracy percentage."

### 8.5 Generation experiment (30 questions — incomplete)

A 30-question end-to-end run (`results/phase5_end_to_end_30.json`) **stopped at 3 / 30** due to repeated Groq rate-limit errors. This partial run must **not** be reported as a 30-question benchmark — mention it only as an evaluation limitation.

### 8.6 Reranking

Reranking is **not** in production. A cross-encoder pilot was abandoned because CPU/runtime constraints prevented a valid result. **No claim** that reranking improves the system should be made.

### 8.7 Phase 6 production UI smoke test

Manual single-question smoke test through Streamlit — all checks passed:

- Streamlit startup
- Pipeline execution
- Answer generation
- Source display
- Latency reporting
- Token information
- Citation parsing
- Citation-to-document mapping
- Answer text unchanged by citation parsing
- No Groq 429 during the test

---

## 9. Engineering Quality

| Area | Implementation |
|------|----------------|
| Structured logging | `src/logging_config.py` — JSON event logs from pipeline stages |
| Centralized configuration | `config.py` — environment variables with validation |
| Typed errors | `src/errors.py` — `ConfigError`, `RetrievalError`, `GenerationError`, etc. |
| Query validation | `src/input_validation.py` — empty/length checks |
| History management | `src/history_manager.py` — turn and token-budget truncation |
| Token estimation | `src/token_counter.py` — prompt size estimation |
| API token observability | `src/llm.py` — Groq usage fields surfaced to UI |
| Citation parsing/validation | `src/citations.py` — parse `[n]` ranks, validate against retrieved sources |
| Rate-limit metadata | `src/llm.py` — extracts retry/rate-limit headers on failure |
| Unit tests | `tests/test_citations.py` — **9 / 9 passed** (supported automated test suite) |

Early root-level `test_*.py` scripts are archived under `archive/early_experiments/` — historical/API experiments, not active tests.

---

## 10. Project Structure

```
first_rag_project/
├── app.py                  # Primary entry — Streamlit conversational UI
├── main.py                 # Legacy single-turn CLI (non-conversational)
├── config.py               # Environment-based settings
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── docs/
│   └── images/                   # Screenshots referenced in this README
├── archive/
│   └── early_experiments/        # Archived historical/API experiment scripts
├── data/
│   ├── raw/train_data.json       # Production corpus
│   ├── splits/                   # Train/holdout splits (eval)
│   └── eval/                     # Conversation scenarios, benchmarks
├── src/                          # Core library (pipeline, retrieval, LLM, citations)
├── scripts/                      # Evaluation and experiment scripts
├── results/                      # Evaluation artifacts (mostly gitignored)
├── tests/
│   └── test_citations.py         # Supported unit tests (9/9)
├── chroma_db/                    # Gitignored — local runtime vector store
└── chroma_db_train/              # Gitignored — train-only eval index
```

---

## 11. Installation

```bash
git clone <repo-url>
cd first_rag_project
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
```

Edit `.env` and set your `GROQ_API_KEY`.

```bash
streamlit run app.py
```

**Dependencies** (`requirements.txt`): `streamlit`, `chromadb`, `sentence-transformers`, `groq`, `python-dotenv`. The `google-genai` package is used by evaluation scripts only (Gemini judge).

**Optional — single-turn CLI:**

```bash
python main.py
```

---

## 12. First Run

On the first launch of `streamlit run app.py`:

1. **Embedding model download** — SentenceTransformer fetches `intfloat/multilingual-e5-small` if not cached locally.
2. **Chroma index build** — If `chroma_db/` is empty, all ~286 paragraphs are embedded and persisted. This can take several minutes.
3. **Groq API required** — Both rewrite (8B) and generation (70B) call Groq on every conversational turn.

`chroma_db/` is gitignored and should not be committed. Each clone builds its own local index.

---

## 13. Configuration

All settings load from environment variables. See `.env.example` for the full list.

### Required for production

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required** — Groq API access |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Generation model |
| `REWRITE_MODEL` | `llama-3.1-8b-instant` | Query rewrite model |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding model |
| `TOP_K` | `3` | Passages retrieved per query |
| `TEMPERATURE` | `0.1` | Generation temperature |
| `DATA_PATH` | `data/raw/train_data.json` | Corpus JSON path |
| `CHROMA_PATH` | `chroma_db` | Vector store directory |
| `MAX_HISTORY_TURNS` | `6` | Max prior turns kept in history |
| `MAX_HISTORY_TOKENS` | `800` | Token budget for history in prompt |
| `MAX_QUERY_CHARS` | `1000` | Maximum user message length |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Optional / evaluation-only

| Variable | Default | Used for |
|----------|---------|----------|
| `GEMINI_API_KEY` | — | Evaluation scripts (LLM-as-judge) |
| `SPLITS_DIR` | `data/splits` | Train/holdout split files |
| `HOLDOUT_RATIO` | `0.2` | Split creation ratio |
| `SPLIT_SEED` | `42` | Reproducible split seed |
| `EVAL_CHROMA_PATH` | `chroma_db_train` | Train-only index for strict eval |
| `JUDGE_MODEL` | `gemini-2.0-flash` | Generation judging in eval scripts |
| `LLM_MODEL_8B` / `LLM_MODEL_70B` | — | Phase 3 model comparison |
| `CONTEXT_STRATEGY` | `baseline` | Context compression experiments |

---

## 14. Running Evaluations

Evaluation scripts are separate from the production app. They do not run automatically.

```bash
# Create train/holdout splits
python scripts/split_dataset.py

# Single-turn retrieval on holdout
python scripts/eval_retrieval.py

# Rewrite-only (8B, no retrieval or generation)
python scripts/eval_rewrite_only.py

# Conversational retrieval (rewrite + retrieval, no 70B)
python scripts/eval_conversation_retrieval.py

# Full suite (retrieval + optional generation judge)
python scripts/run_all_evals.py
python scripts/run_all_evals.py --skip-generation   # skip Gemini judge calls
```

Holdout evaluation indexes the **full paragraph corpus** while testing on held-out **questions** — gold paragraphs remain retrievable. For strict document generalization (train-only index):

```bash
python scripts/eval_retrieval.py --index-mode train_only
```

Individual eval scripts also exist for token measurement, model benchmarking, normalization experiments, and Phase 5 generation pilots — see `scripts/`.

---

## 15. Limitations

1. **Retrieval Hit@1 is 72.95%** — wrong or missing documents still occur on holdout questions.
2. **Fixed corpus** — knowledge is limited to ~286 AmQA paragraphs; no live Wikipedia or web access.
3. **Small conversational retrieval eval** — 18 scored turns on curated scenarios; 100% on that set does not generalize.
4. **Generation eval is a 10-question pilot only** — no statistically strong end-to-end accuracy estimate.
5. **30-question generation run incomplete** — stopped at 3/30 due to Groq rate limits.
6. **Generation latency** — 70B calls averaged ~31 s in the pilot; first run also waits on model download and index build.
7. **Groq API dependency** — subject to rate limits, quotas, and availability.
8. **Reranking not integrated** — cross-encoder experiment abandoned; not part of production.
9. **Citation validation checks rank mapping, not factual support** — a valid `[1]` citation does not prove the answer is correct.
10. **Refusal phrase is English** — returned even for Amharic questions when context is insufficient.

---

## 16. Security

Never commit secrets or local runtime artifacts:

| Path | Reason |
|------|--------|
| `.env` | Contains `GROQ_API_KEY` and other secrets |
| `.streamlit/secrets.toml` | Streamlit secrets |
| `chroma_db/` | Local vector store (reproducible from corpus) |
| `chroma_db_train/` | Evaluation index |

Copy `.env.example` to `.env` locally and keep API keys out of version control.

---

## 17. Project Status

**Complete as an AI Engineering project** with a working conversational Streamlit application, grounded prompting, citations, observability, and documented evaluation. Retrieval and rewrite components are measured; end-to-end generation accuracy is **not** fully benchmarked at scale. Suitable for portfolio demonstration with the limitations above clearly understood.

---

## 18. License

Portfolio / educational project.
