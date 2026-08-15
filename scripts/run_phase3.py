"""Run all Phase 3 measurement scripts (no retrieval changes)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from scripts.benchmark_models import run_model_benchmark
from scripts.eval_context_strategies import run_context_strategy_eval
from scripts.eval_tokens import run_token_eval
from src.eval_utils import save_json
from src.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 measurement suite.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--skip-model-benchmark",
        action="store_true",
        help="Skip live 8B vs 70B API benchmark.",
    )
    args = parser.parse_args()

    settings = get_settings(require_groq=not args.skip_model_benchmark)
    setup_logging(settings.log_level)

    token_baseline = run_token_eval(settings=settings, sample_size=args.sample_size)
    context_strategies = run_context_strategy_eval(
        settings=settings,
        sample_size=args.sample_size,
    )

    model_benchmark = None
    if not args.skip_model_benchmark:
        model_benchmark = run_model_benchmark(settings=settings)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "token_baseline_summary": token_baseline["summary"],
        "context_strategies": {
            name: {
                "answer_preservation_rate": data["answer_preservation_rate"],
                "gold_document_preservation_rate": data["gold_document_preservation_rate"],
                "prompt_tokens_p95": data["prompt_tokens"]["p95"],
            }
            for name, data in context_strategies["strategies"].items()
        },
        "model_benchmark_summary": model_benchmark["summary"] if model_benchmark else None,
    }
    save_json("results/phase3_summary.json", summary)
    print("Phase 3 measurements saved to results/phase3_summary.json")


if __name__ == "__main__":
    main()
