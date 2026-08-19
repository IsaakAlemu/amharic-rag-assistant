# Conversational Amharic RAG Assistant

An end-to-end **conversational Retrieval-Augmented Generation (RAG)** system tailored for the Amharic language over an AmQA-derived Wikipedia knowledge base.

The system features **hybrid retrieval (dense vector search + lexical BM25 fused via Reciprocal Rank Fusion)**, conversational multi-turn query rewriting, **multi-layer prompt-injection defenses**, strict factual grounding with inline citations, real-time token streaming, and automated CI testing.

---

## 1. Key Features

- **Amharic-Native QA & Real-Time Streaming:** Responds to questions in natural, fluent Amharic with live token-by-token streaming in a Streamlit web interface.
- **Hybrid Retrieval (Dense + BM25 via RRF):** Combines `intfloat/multilingual-e5-small` dense embeddings with an in-memory Amharic BM25 keyword retriever using Reciprocal Rank Fusion ($k=60$) to capture both semantic meaning and exact entity/acronym matches (e.g. `የተ.መ.ድ / ዩኤን ኤድስ`).
- **Conversational Multi-Turn Context:** Uses an LLM rewriter to resolve pronouns, ellipsis, and context across turns into standalone retrieval queries without polluting the retriever with raw chat history.
- **Evidence-Grounded Generation:** Enforces strict boundary rules where answers are derived exclusively from retrieved documents, citing sources inline as `[1]`, `[2]`.
- **Bilingual Grounded Refusal:** Accurately outputs an explicit refusal (`"ከተሰጡት ሰነዶች በመነሳት ጥያቄውን መመለስ አልተቻለም።"`) when context is missing, preventing hallucinated assertions.
- **Multi-Layer Security & Guardrails:**
  - Input sanitization (strips null bytes, control characters, and `<script>` injections).
  - Adversarial classifier detecting prompt-injection and jailbreak attempts in both English and Amharic.
  - XML delimiter isolation (`<retrieved_evidence>`, `<user_question>`) preventing context confusion.
- **Ge'ez Sentence-Boundary Chunking:** Custom text chunker aware of Ethiopic punctuation (`።`, `፤`, `?`, `!`) with sliding character overlap.
- **Multi-Provider LLM Support:** Configurable support for **Google Gemini** (`gemini-3.6-flash`) and **Groq** (`llama-3.3-70b-versatile` / `openai/gpt-oss-120b`).
- **Observability & Quota Protection:** Displays per-turn execution latencies (rewrite, retrieve, generate), token metrics, valid citation mappings, and an interactive 12-turn session counter.
- **GitHub Actions CI & Docker Configuration:** 23 unit tests executed automatically via GitHub Actions, accompanied by a production `Dockerfile`.

---

## 2. Architecture

```
User Question (Amharic)
        │
        ▼
[1. Input Validation & Security Guardrails]
    • Length & control character sanitization
    • Adversarial prompt injection classifier (EN / AM)
        │
        ▼
[2. Conversational Context & Query Rewriter]
    • Resolves multi-turn pronouns & references
    • Produces a standalone search query
        │
        ▼
[3. Hybrid Retrieval Engine]
    ├── Dense Semantic Search (ChromaDB + multilingual-e5-small)
    └── Lexical Keyword Search (Amharic BM25)
        │
        ▼
[4. Reciprocal Rank Fusion (RRF)]
    • Fuses dense and lexical ranks: Score = Σ 1 / (60 + rank)
    • Yields Top-K grounded evidence passages
        │
        ▼
[5. XML-Delimited Grounded Generation]
    • Prompt wrapped in <retrieved_evidence> & <user_question>
    • Google Gemini / Groq LLM
    • Real-time token streaming
        │
        ▼
[6. Citation Validation & UI Presentation]
    • Maps inline markers [1], [2] to retrieved source documents
    • Emits refusal card if evidence is insufficient
```

### Core Design Principles

1. **Retrieval uses the rewritten query only:** Conversation history is never directly embedded into vector space, preventing topic drift across turns.
2. **Context isolation via XML tags:** Prevents untrusted user inputs from overriding system instructions or confusing previous chat history with factual evidence.
3. **Empty retrieval short-circuits generation:** If retrieval yields no context, generation is skipped and the grounded refusal phrase is returned immediately.

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Web UI** | Streamlit | Chat interface, streaming tokens, telemetry, typography |
| **LLM Providers** | Google Gemini (`google-genai`) / Groq | Conversational rewriting and grounded generation |
| **Embeddings** | `intfloat/multilingual-e5-small` | 384-dimensional dense multilingual vector embeddings |
| **Vector Store** | ChromaDB | Persistent local cosine vector database |
| **Lexical Search** | Custom BM25 Index | Amharic word tokenization and keyword matching |
| **Rank Fusion** | Reciprocal Rank Fusion (RRF) | Merging dense vector and BM25 candidate ranks |
| **CI** | GitHub Actions | Automated Python 3.11 unit test suite on push/PR |
| **Container** | Docker (`python:3.11-slim`) | Multi-stage production containerization |

---

## 4. Evaluation and Empirical Results

All reported metrics are measured on completed, reproducible runs from the project benchmark artifacts (`results/`):

### 4.1 Single-Turn Retrieval Baseline (329-Question Holdout Set)

Evaluated on 329 held-out test questions across the ~286 passage AmQA knowledge base:

| Metric | Score | Description |
|---|---|---|
| **Hit@1** | **72.95%** | Gold document retrieved at rank 1 (production `top_k=3` setting) |
| **Hit@3** | **84.19%** | Gold document retrieved in top 3 (production `top_k=3` setting) |
| **Hit@5** | **87.23%** | Gold document retrieved in top 5 (top-k sweep) |
| **Hit@10** | **89.36%** | Gold document retrieved in top 10 (top-k sweep) |
| **MRR** | **0.781** | Mean Reciprocal Rank |

### 4.2 Dense vs. Hybrid Retrieval (BM25 + RRF)

Hybrid retrieval resolves edge cases where dense embeddings miss exact Ethiopian acronyms or rare named entities.

**Example Benchmark Query:** `"የተ.መ.ድ አካል ዩኤን ኤድስ በምን ላይ ትኩረት አድርጎ ይሠራል?"` *(Gold Document: `451675`)*

| Retrieval Method | Rank 1 Result | Rank of Gold Document (`451675`) |
|---|---|---|
| **Pure Dense (E5 + Chroma)** | `Doc 451575` | **Rank 2** (Missed at Top-1) |
| **Pure BM25 Lexical** | `Doc 451675` | **Rank 1** (Exact acronym match) |
| **Hybrid Fusion (Dense + BM25 via RRF)** | `Doc 451675` | **Rank 1** (**Promoted to Top-1**) |

### 4.3 Security & Guardrail Verification

- **Direct Adversarial Tests:** 100% of tested canonical injection attempts in English and Amharic (e.g. *"ignore previous instructions"*, *"የቀደመውን መመሪያ እርሳው"*) are intercepted by `src/security.py`.
- **Paraphrased / Delimiter Robustness:** Queries bypassing keyword regex are contained by `<user_question>` XML boundaries, triggering grounded refusal without leaking system instructions.
- **Automated Test Suite:** **23 / 23 unit tests pass** across citations, chunking, security, and hybrid retrieval.

---

## 5. Repository Structure

```
amharic-rag-assistant/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (Python 3.11)
├── src/
│   ├── chunker.py             # Ge'ez sentence-boundary text chunker
│   ├── citations.py           # Citation parsing & document mapping
│   ├── document_loader.py     # AmQA JSON dataset loader
│   ├── embedding_generator.py # E5 vector embedding & Chroma indexer
│   ├── errors.py              # Typed domain exceptions
│   ├── history_manager.py     # Sliding window conversation memory
│   ├── hybrid_retriever.py    # BM25 index & Reciprocal Rank Fusion
│   ├── input_validation.py    # Query length & security entrypoint
│   ├── llm.py                 # Multi-provider LLM client (Gemini / Groq)
│   ├── logging_config.py      # Structured JSON event logging
│   ├── pipeline.py            # End-to-end RAG pipeline orchestration
│   ├── prompt_builder.py      # XML-delimited prompt assembly
│   ├── query_rewriter.py      # Conversational history query rewriter
│   ├── retriever.py           # Dense semantic Chroma retrieval
│   ├── security.py            # Input sanitization & injection classifier
│   └── token_counter.py       # Prompt token estimation
├── tests/
│   ├── test_chunker.py        # Tokenizer & chunking unit tests (5 tests)
│   ├── test_citations.py      # Citation validation unit tests (9 tests)
│   ├── test_hybrid_retriever.py # BM25 & RRF fusion unit tests (3 tests)
│   └── test_security.py       # Sanitization & injection tests (6 tests)
├── scripts/
│   ├── ingest_corpus.py       # CLI tool to chunk & ingest custom documents
│   ├── split_dataset.py       # Train/holdout dataset splitter
│   ├── eval_retrieval.py      # Single-turn retrieval evaluation
│   ├── eval_rewrite_only.py   # Query rewrite benchmark
│   ├── eval_conversation_retrieval.py # Conversational retrieval evaluation
│   └── run_all_evals.py       # Evaluation orchestrator
├── data/
│   ├── raw/train_data.json    # AmQA Wikipedia knowledge corpus
│   └── splits/                # Train and holdout benchmark splits
├── results/                   # Benchmark evaluation artifacts
├── app.py                     # Primary Streamlit web application
├── main.py                    # Single-turn CLI interface
├── config.py                  # Centralized configuration & settings
├── Dockerfile                 # Production Docker deployment container
├── requirements.txt           # Python dependencies
└── .env.example               # Environment variables template
```

---

## 6. Installation & Setup

### Prerequisites

- Python 3.10 or 3.11
- A Google Gemini API Key (or Groq API Key)

### 1. Clone the repository

```bash
git clone https://github.com/IsaakAlemu/amharic-rag-assistant.git
cd amharic-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the template file to `.env`:

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and configure your chosen provider:

```env
# For Google Gemini (default)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Or for Groq
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_actual_groq_api_key_here
```

---

## 7. Running the Application

### Interactive Web UI (Streamlit)

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. On initial launch:
1. `intfloat/multilingual-e5-small` weights are downloaded (~133 MB).
2. The AmQA corpus (~286 documents) is automatically embedded and indexed in `chroma_db/`.

### Single-Turn CLI

```bash
python main.py
```

---

## 8. Docker Deployment

Build and run the container locally:

```bash
# Build the Docker image
docker build -t amharic-rag-assistant .

# Run the container
docker run -p 8501:8501 --env-file .env amharic-rag-assistant
```

The web service will be available at `http://localhost:8501`.

---

## 9. Running Tests & Evaluations

### Automated Test Suite (23 Tests)

Run all unit tests:

```bash
python -m unittest discover tests -v
```

### Running Benchmark Evaluations

```bash
# Evaluate single-turn retrieval on the 329-question holdout set
python scripts/eval_retrieval.py

# Evaluate multi-turn query rewriting
python scripts/eval_rewrite_only.py

# Ingest and chunk custom documents (.json, .txt, .md)
python scripts/ingest_corpus.py --file data/raw/train_data.json --collection amqa
```

---

## 10. Limitations

1. **Corpus Scope:** Grounded knowledge is currently bounded to ~286 AmQA Wikipedia articles; out-of-corpus queries will be intentionally refused.
2. **Retrieval Precision:** Single-turn dense retrieval Hit@1 is 72.95% on the holdout benchmark — dense search occasionally ranks a non-gold passage at rank 1.
3. **Citation Scope:** Inline citations validate mapping to retrieved passage ranks (`[1]`, `[2]`), but do not verify semantic factuality beyond what prompting and retrieval restrict.

---

## 11. Author

**Isaak Alemu**  
Built independently as an AI Engineering portfolio project.
- **GitHub:** [@IsaakAlemu](https://github.com/IsaakAlemu)
