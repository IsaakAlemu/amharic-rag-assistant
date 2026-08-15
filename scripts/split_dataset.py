"""Create train/hold-out splits by document_id."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, holdout_split_path, split_manifest_path, train_split_path
from src.eval_utils import (
    iter_paragraphs,
    load_amqa_raw,
    paragraphs_to_amqa,
    save_json,
    split_paragraphs_by_document_id,
)
from src.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Split AmQA data by document_id.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSON path (defaults to DATA_PATH from config).",
    )
    args = parser.parse_args()

    settings = get_settings(require_groq=False)
    setup_logging(settings.log_level)

    input_path = args.input or settings.data_path
    raw = load_amqa_raw(input_path)
    paragraphs = iter_paragraphs(raw)

    train_paragraphs, holdout_paragraphs, manifest = split_paragraphs_by_document_id(
        paragraphs,
        holdout_ratio=settings.holdout_ratio,
        seed=settings.split_seed,
    )

    train_path = train_split_path(settings)
    holdout_path = holdout_split_path(settings)
    manifest_path = split_manifest_path(settings)

    save_json(train_path, paragraphs_to_amqa(train_paragraphs))
    save_json(holdout_path, paragraphs_to_amqa(holdout_paragraphs))
    save_json(manifest_path, manifest)

    print(f"Wrote train split: {train_path} ({len(train_paragraphs)} paragraphs)")
    print(f"Wrote holdout split: {holdout_path} ({len(holdout_paragraphs)} paragraphs)")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
