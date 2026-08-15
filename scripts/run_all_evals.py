"""Run all evaluation scripts and write a combined summary."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from scripts.eval_generation import run_generation_eval
from scripts.eval_out_of_scope import run_out_of_scope_eval
from scripts.eval_retrieval import run_retrieval_eval
from scripts.split_dataset import main as split_main
from src.eval_utils import save_json
from src.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full evaluation suite.")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation eval (requires GEMINI_API_KEY and API calls).",
    )
    parser.add_argument(
        "--skip-out-of-scope",
        action="store_true",
        help="Skip out-of-scope eval (requires GROQ_API_KEY and API calls).",
    )
    parser.add_argument(
        "--generation-limit",
        type=int,
        default=30,
        help="Max generation eval questions.",
    )
    parser.add_argument(
        "--output",
        default="results/eval_summary.json",
        help="Combined summary output path.",
    )
    args = parser.parse_args()

    settings = get_settings(require_groq=not args.skip_out_of_scope)
    setup_logging(settings.log_level)

    split_main()

    retrieval = run_retrieval_eval(settings=settings)
    out_of_scope = None
    generation = None

    if not args.skip_out_of_scope:
        out_of_scope = run_out_of_scope_eval(settings=settings)

    if not args.skip_generation:
        generation = run_generation_eval(
            settings=get_settings(require_gemini=True),
            limit=args.generation_limit,
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "embed_model": settings.embed_model,
            "llm_model": settings.llm_model,
            "top_k": settings.top_k,
            "holdout_ratio": settings.holdout_ratio,
            "split_seed": settings.split_seed,
        },
        "retrieval": retrieval,
        "generation": generation,
        "out_of_scope": out_of_scope,
    }
    save_json(args.output, summary)

    print("=" * 50)
    print("Combined evaluation summary")
    print("=" * 50)
    metrics = retrieval["metrics"]
    print(f"Retrieval Hit@1: {metrics['hit_at_1']:.2%}")
    print(f"Retrieval MRR:   {metrics['mrr']:.4f}")
    if out_of_scope:
        print(f"Refusal accuracy: {out_of_scope['refusal_accuracy']:.2%}")
    if generation:
        print("Generation averages:")
        for key, value in generation.get("averages", {}).items():
            print(f"  {key}: {value:.2f}/5")
    print(f"Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
