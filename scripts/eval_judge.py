"""Phase 2: Evaluate generated answers using Gemini (gemini-3.6-flash pinned) as LLM Judge.

Appends each scored evaluation to results/phase5_e2e_judged.jsonl immediately upon completion.
Resumes automatically if interrupted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google import genai
from google.genai import types
from config import get_settings

JUDGE_MODEL_PINNED = "gemini-3.6-flash"

GEN_INPUT_FILE = Path("results/phase5_e2e_generations.jsonl")
JUDGE_OUTPUT_FILE = Path("results/phase5_e2e_judged.jsonl")
ERROR_LOG_FILE = Path("results/phase5_e2e_errors.log")

JUDGE_PROMPT_TEMPLATE = """You are evaluating the output of a Retrieval-Augmented Generation system for Amharic question answering. You will be given: the user's question, the context chunks the system retrieved, the system's generated answer, and (if available) a reference answer.

Score the generated answer on four axes:

FAITHFULNESS (1-5): Is every claim in the answer supported by the retrieved context? 5 = fully grounded, no unsupported claims. 1 = largely fabricated or contradicts the context.

RELEVANCE (1-5): Does the answer actually address the question asked? 5 = directly and completely answers it. 1 = off-topic or non-responsive.

CORRECTNESS (1-5): Where a reference answer is provided, does the generated answer match it in substance? 5 = fully correct. 1 = substantively wrong. If no reference answer is provided, score based on plausibility given the context instead and note "no_reference: true".

CITATION_ACCURACY (1-5 or null): Does the answer accurately cite the specific retrieved passage rank(s) (e.g. [1], [2]) that actually support each stated claim? 5 = all inline citations are accurate and map to the supporting passages. 1 = citations are completely misleading/fabricated. null = if the answer contains no citations or is a refusal where citations are not applicable.

Also return ABSTAIN_APPROPRIATE (true/false/null): true if the question is out-of-scope / context-missing and the system correctly declined to answer; false if it should have declined but didn't, or vice versa; null if not applicable.

Return ONLY valid JSON, no markdown fences, no preamble:
{{
  "faithfulness": <int 1-5>,
  "relevance": <int 1-5>,
  "correctness": <int 1-5>,
  "citation_accuracy": <int 1-5|null>,
  "no_reference": <bool>,
  "abstain_appropriate": <bool|null>,
  "reasoning": "<1-2 sentence justification>"
}}

Question: {question}
Retrieved context: {context}
Generated answer: {answer}
Reference answer: {reference_answer}
"""


def log_error(question_id: int | str, error: Exception | str) -> None:
    ERROR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [Phase 2 Judge] Question {question_id}: {error}\n")


def load_completed_judged_ids() -> set[int | str]:
    if not JUDGE_OUTPUT_FILE.exists():
        return set()
    completed = set()
    with open(JUDGE_OUTPUT_FILE, "r", encoding="utf-8") as f:
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
    return completed


def clean_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    # Remove markdown formatting if any was returned
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def call_gemini_judge_with_retry(
    client: genai.Client,
    prompt: str,
    max_retries: int = 5
) -> dict:
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL_PINNED,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            raw = response.text or ""
            return clean_json_response(raw)
        except Exception as exc:
            msg = str(exc)
            retry_after = None
            if "retrydelay" in msg.lower():
                match = re.search(r"retrydelay['\":\s]+(\d+)", msg.lower())
                if match:
                    retry_after = float(match.group(1))

            if attempt == max_retries:
                raise exc

            wait_time = retry_after if retry_after is not None else min(60.0, delay + random.uniform(0.2, 1.0))
            print(f"  [Attempt {attempt}/{max_retries}] Judge call failed ({type(exc).__name__}: {msg[:60]}...). Backing off for {wait_time:.1f}s...")
            time.sleep(wait_time)
            delay = min(60.0, delay * 2)


def main():
    settings = get_settings(require_gemini=True)
    if not GEN_INPUT_FILE.exists():
        print(f"Generation output not found at {GEN_INPUT_FILE}. Run eval_generate.py first.")
        sys.exit(1)

    generations = []
    with open(GEN_INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line.strip()))

    JUDGE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = load_completed_judged_ids()
    print(f"Phase 2: Total generations={len(generations)}, Already judged={len(completed_ids)}")
    print(f"Pinned Judge Model: {JUDGE_MODEL_PINNED}")

    client = genai.Client(api_key=settings.gemini_api_key)

    processed = 0
    for idx, gen in enumerate(generations, 1):
        qid = gen["question_id"]
        q_text = gen["question"]
        ref_ans = gen.get("reference_answer") or "none provided"
        gen_ans = gen.get("generated_answer", "")
        ret_context = gen.get("retrieved_context", [])

        if qid in completed_ids:
            print(f"[{idx}/{len(generations)}] Skipping Question {qid} (already judged).")
            continue

        print(f"[{idx}/{len(generations)}] Judging Question {qid}: {q_text[:50]}...")

        # Format context chunks for rubric
        context_str = "\n\n".join(
            f"[{c.get('rank', i+1)}] Doc ID: {c.get('document_id')}\n{c.get('text', '')}"
            for i, c in enumerate(ret_context)
        )

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=q_text,
            context=context_str,
            answer=gen_ans,
            reference_answer=ref_ans
        )

        try:
            scores = call_gemini_judge_with_retry(client, prompt, max_retries=5)
        except Exception as exc:
            print(f"  Judging failed after max retries for question {qid}: {exc}")
            log_error(qid, f"Judge failure after retries: {exc}")
            continue

        judged_record = {
            "question_id": qid,
            "retrieval_outcome": gen.get("retrieval_outcome", "unknown"),
            "faithfulness": scores.get("faithfulness"),
            "relevance": scores.get("relevance"),
            "correctness": scores.get("correctness"),
            "citation_accuracy": scores.get("citation_accuracy"),
            "no_reference": scores.get("no_reference", False),
            "abstain_appropriate": scores.get("abstain_appropriate"),
            "reasoning": scores.get("reasoning", ""),
            "judge_model": JUDGE_MODEL_PINNED,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        with open(JUDGE_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(judged_record, ensure_ascii=False) + "\n")
            f.flush()

        completed_ids.add(qid)
        processed += 1
        print(f"  Scored Question {qid} (F:{scores.get('faithfulness')}, R:{scores.get('relevance')}, C:{scores.get('correctness')}, Cite:{scores.get('citation_accuracy')}) -> Saved.")

        # Fixed 1.5s delay to prevent burst RPM throttling
        time.sleep(1.5)

    print(f"\nPhase 2 Judging complete! Processed {processed} items. Total judged records in {JUDGE_OUTPUT_FILE}: {len(completed_ids)}")


if __name__ == "__main__":
    main()
