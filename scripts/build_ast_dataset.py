#!/usr/bin/env python
"""Build clean/AST dataset splits for the spam-text project."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ast_dataset import build_ast_dataset
from src.config import Config


def parse_source_path(value: str):
    """Parse SOURCE=PATH or use the directory name as the source."""
    if "=" in value:
        source, path = value.split("=", 1)
        return source.strip(), Path(path).expanduser()
    path = Path(value).expanduser()
    return path.name or "project_msglog", path


def main() -> None:
    cfg = Config.AST_DATASET_CONFIG
    parser = argparse.ArgumentParser(
        description="Build AST train/val/test splits without training any model."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Project msglog directory. Use SOURCE=PATH to preserve dataset provenance. Can be repeated.",
    )
    parser.add_argument(
        "--canonical-jsonl",
        action="append",
        default=[],
        help="Canonical JSONL with text/label fields. Use SOURCE=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(Config.DATA_DIR) / cfg["output_dir_name"]),
        help="Output directory for canonical JSONL, msglog text exports, and manifest.",
    )
    parser.add_argument("--train-ratio", type=float, default=cfg["train_ratio"])
    parser.add_argument("--val-ratio", type=float, default=cfg["val_ratio"])
    parser.add_argument("--test-ratio", type=float, default=cfg["test_ratio"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--max-variants-spam", type=int, default=cfg["max_variants_spam"])
    parser.add_argument("--max-variants-normal", type=int, default=cfg["max_variants_normal"])
    parser.add_argument(
        "--ast-strength",
        choices=["mild", "balanced", "strong"],
        default=cfg.get("ast_strength", "mild"),
        help="Text-level AST perturbation profile. Existing full results use mild.",
    )
    args = parser.parse_args()

    input_dirs = [parse_source_path(item) for item in args.input_dir]
    canonical_jsonl = [parse_source_path(item) for item in args.canonical_jsonl]

    if not input_dirs and not canonical_jsonl:
        default_msglog = Path(Config.MSG_LOG_DIR)
        input_dirs = [("project_msglog", default_msglog)]

    stats = build_ast_dataset(
        input_dirs=input_dirs,
        canonical_jsonl=canonical_jsonl,
        output_dir=Path(args.output_dir).expanduser(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_variants_spam=args.max_variants_spam,
        max_variants_normal=args.max_variants_normal,
        ast_strength=args.ast_strength,
    )

    print("AST dataset build completed.")
    print(f"Loaded: {stats.loaded}")
    print(f"Kept after dedup: {stats.kept_after_dedup}")
    print(f"Dropped duplicates: {stats.dropped_duplicates}")
    print(f"Dropped conflicting labels: {stats.dropped_conflicting_labels}")
    print(f"AST strength: {args.ast_strength}")
    print(f"Output: {Path(args.output_dir).expanduser()}")


if __name__ == "__main__":
    main()
