"""Phase 1: Generate end-to-end RAG answers for evaluation questions.

Appends each result to results/phase5_e2e_generations.jsonl immediately upon completion.
Resumes automatically if interrupted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groq import Groq
from sentence_transformers import SentenceTransformer

from config import get_settings
from src.errors import GenerationError
from src.history_manager import ConversationState
from src.hybrid_retriever import reciprocal_rank_fusion
from src.llm import generate_answer
from src.pipeline import get_bm25_index, load_vector_collection
from src.prompt_builder import build_prompt
from src.retriever import retrieve

GEN_OUTPUT_FILE = Path("results/phase5_e2e_generations.jsonl")
ERROR_LOG_FILE = Path("results/phase5_e2e_errors.log")
DATASET_FILE = Path("data/phase5_e2e_questions.json")


def log_error(question_id: int | str, error: Exception | str) -> None:
    ERROR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [Phase 1 Generation] Question {question_id}: {error}\n")


def load_completed_and_failed_question_ids() -> tuple[set[int | str], set[int | str]]:
    completed = set()
    if GEN_OUTPUT_FILE.exists():
        with open(GEN_OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "question_id" in data:
                        completed.add(data["question_id"])
                except Exception:
                    continue

    failed = set()
    if ERROR_LOG_FILE.exists():
        with open(ERROR_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "Question " in line:
                    try:
                        # Extract question id from '[timestamp] [Phase 1 Generation] Question 12345: ...'
                        part = line.split("Question ")[1].split(":")[0].strip()
                        if part.isdigit():
                            failed.add(int(part))
                        else:
                            failed.add(part)
                    except Exception:
                        continue

    return completed, failed


def call_llm_with_retry(
    prompt: str,
    client: Groq,
    model: str,
    temperature: float = 0.1,
    max_retries: int = 5,
    timeout_seconds: float = 35.0,
):
    delay = 3.0
    for attempt in range(1, max_retries + 1):
        t_call_start = time.perf_counter()
        ts_start = datetime.utcnow().strftime("%H:%M:%S")
        print(f"  [{ts_start}] [Attempt {attempt}/{max_retries}] Starting Groq API call (timeout={timeout_seconds}s)...", flush=True)
        try:
            # Direct chat completions call with explicit request timeout
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=timeout_seconds,
            )
            elapsed = time.perf_counter() - t_call_start
            ts_end = datetime.utcnow().strftime("%H:%M:%S")
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            p_tok = getattr(usage, "prompt_tokens", None) if usage else None
            c_tok = getattr(usage, "completion_tokens", None) if usage else None
            print(f"  [{ts_end}] Groq call succeeded in {elapsed:.2f}s (prompt_tokens={p_tok}, completion_tokens={c_tok})", flush=True)

            from src.llm import GenerationResult
            return GenerationResult(
                text=content.strip(),
                model=model,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t_call_start
            ts_err = datetime.utcnow().strftime("%H:%M:%S")
            msg = str(exc)
            print(f"  [{ts_err}] Groq call failed after {elapsed:.2f}s ({type(exc).__name__}: {msg[:70]}...)", flush=True)

            retry_after = None
            orig_exc = getattr(exc, "__cause__", None) or exc
            response = getattr(orig_exc, "response", None)
            if response is not None and hasattr(response, "headers"):
                ra_header = response.headers.get("retry-after")
                if ra_header:
                    try:
                        retry_after = float(ra_header)
                    except ValueError:
                        pass

            if attempt == max_retries:
                raise exc

            wait_time = retry_after if retry_after is not None else min(60.0, delay + random.uniform(0.5, 2.0))
            print(f"  Backing off for {wait_time:.1f}s before retry...", flush=True)
            time.sleep(wait_time)
            delay = min(60.0, delay * 2.5)


def main():
    settings = get_settings(require_groq=True)
    if not DATASET_FILE.exists():
        print(f"Dataset not found at {DATASET_FILE}. Please run prepare_phase5_e2e_dataset.py first.")
        sys.exit(1)

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["questions"]

    GEN_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    completed_ids, failed_ids = load_completed_and_failed_question_ids()
    print(f"Phase 1: Total questions={len(questions)}, Succeeded={len(completed_ids)}, Previously Failed={len(failed_ids)}")

    # Initialize RAG models
    print("Loading pipeline components (SentenceTransformer, Chroma, BM25, Groq client with 35s timeout)...")
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)
    bm25_index = get_bm25_index(settings)
    groq_client = Groq(api_key=settings.groq_api_key, timeout=35.0)
    gen_model = "openai/gpt-oss-120b"

    processed = 0
    for idx, item in enumerate(questions, 1):
        qid = item["question_id"]
        q_text = item["question"]
        ref_ans = item.get("reference_answer", "")
        outcome = item.get("retrieval_outcome", "unknown")

        if qid in completed_ids:
            print(f"[{idx}/{len(questions)}] Skipping Question {qid} (already succeeded).")
            continue

        if qid in failed_ids:
            print(f"[{idx}/{len(questions)}] Skipping Question {qid} (previously failed after max retries).")
            continue

        print(f"[{idx}/{len(questions)}] Processing Question {qid}: {q_text[:50]}...")

        # 1. Retrieval (trim to top 2 chunks to control prompt token budget)
        t_ret_start = time.perf_counter()
        try:
            dense_sources = retrieve(q_text, collection, embed_model, top_k=4)
            lexical_sources = bm25_index.search(q_text, top_k=4)
            sources = reciprocal_rank_fusion(dense_sources, lexical_sources, top_k=2)
            
            # Truncate each chunk text to max 800 chars to prevent massive TPM spikes
            trimmed_sources = []
            for s in sources:
                s_copy = dict(s)
                if len(s_copy.get("text", "")) > 800:
                    s_copy["text"] = s_copy["text"][:800] + "…"
                trimmed_sources.append(s_copy)
            sources = trimmed_sources

            ret_ms = (time.perf_counter() - t_ret_start) * 1000
        except Exception as exc:
            print(f"  Retrieval error for question {qid}: {exc}")
            log_error(qid, f"Retrieval error: {exc}")
            continue

        # Build context text & prompt
        prompt = build_prompt(q_text, sources)

        # 2. Generation with retry
        t_gen_start = time.perf_counter()
        try:
            gen_res = call_llm_with_retry(prompt, groq_client, model=gen_model, temperature=0.1, max_retries=5)
            gen_ms = (time.perf_counter() - t_gen_start) * 1000
            generated_answer = gen_res.text
        except Exception as exc:
            print(f"  Generation failed after max retries for question {qid}: {exc}")
            log_error(qid, f"Generation failure after retries: {exc}")
            # Move to next question, never crash entire run
            continue

        # 3. Write row to generations.jsonl
        record = {
            "question_id": qid,
            "question": q_text,
            "reference_answer": ref_ans,
            "retrieval_outcome": outcome,
            "gold_document_id": item.get("gold_document_id", ""),
            "retrieved_context": [
                {
                    "rank": s["rank"],
                    "document_id": str(s["document_id"]),
                    "distance": s.get("distance"),
                    "text": s["text"]
                }
                for s in sources
            ],
            "generated_answer": generated_answer,
            "generation_model": gen_model,
            "retrieval_latency_ms": round(ret_ms, 2),
            "generation_latency_ms": round(gen_ms, 2),
            "prompt_tokens": gen_res.prompt_tokens,
            "completion_tokens": gen_res.completion_tokens,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        with open(GEN_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        completed_ids.add(qid)
        processed += 1
        print(f"  Completed Question {qid} (Answer: {generated_answer[:45]}...) -> Saved.")

        # Pace calls by 45s so the ~8000 TPM window clears between questions
        print("  Pacing: waiting 45s for TPM quota window to refresh...")
        time.sleep(45.0)

    print(f"\nPhase 1 Generation complete! Processed {processed} questions. Total records in {GEN_OUTPUT_FILE}: {len(completed_ids)}")


if __name__ == "__main__":
    main()
