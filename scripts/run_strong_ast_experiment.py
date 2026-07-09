#!/usr/bin/env python
"""Build and run a separate strong-AST experiment.

The existing submission output is based on mild AST. This script creates a
separate strong AST dataset and trains/evaluates a separate output directory so
the two experiment profiles can be compared without overwriting each other.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.submission_pipeline import (
    DEFAULT_ENSEMBLE_MODELS,
    DEFAULT_MODELS,
    DEFAULT_MODES,
    FULL_MODELS,
    FULL_MODES,
    RunConfig,
    ast_quality_review,
    attempt_huggingface_gated,
    confidence_search_attack,
    train_and_evaluate,
    write_submission_report,
)
from src.ast_dataset import build_ast_dataset
from src.config import Config


def _resolve_pairs(raw_pairs: Sequence[Sequence[str]]) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for source, path in raw_pairs:
        pairs.append((str(source), Path(path)))
    return pairs


def sources_from_manifest(path: Path) -> Tuple[List[Tuple[str, Path]], List[Tuple[str, Path]]]:
    if not path.exists():
        return [("project_msglog", Path(Config.MSG_LOG_DIR))], []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    settings = manifest.get("settings") or {}
    return (
        _resolve_pairs(settings.get("input_dirs") or []),
        _resolve_pairs(settings.get("canonical_jsonl") or []),
    )


def parse_args() -> argparse.Namespace:
    strong_cfg = Config.STRONG_AST_DATASET_CONFIG
    parser = argparse.ArgumentParser(description="Run the full strong-AST training/evaluation pipeline.")
    parser.add_argument("--base-manifest", default="data/ast_experiment/manifest.json")
    parser.add_argument("--dataset-dir", default=str(Path(Config.DATA_DIR) / strong_cfg["output_dir_name"]))
    parser.add_argument("--output-dir", default="output/submission_strong_ast_20260706_full")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Train the expanded matrix with Focal Loss modes, BiLSTM-Attention, and ensemble_vote.",
    )
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--w2v-epochs", type=int, default=20)
    parser.add_argument("--clf-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=strong_cfg["seed"])
    parser.add_argument("--fgm-epsilon", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble_vote evaluation.")
    parser.add_argument("--ensemble-models", nargs="+", default=DEFAULT_ENSEMBLE_MODELS)
    parser.add_argument("--max-variants-spam", type=int, default=strong_cfg["max_variants_spam"])
    parser.add_argument("--max-variants-normal", type=int, default=strong_cfg["max_variants_normal"])
    parser.add_argument("--ast-strength", choices=["mild", "balanced", "strong"], default=strong_cfg["ast_strength"])
    parser.add_argument("--dynamic-vocab-top-k", type=int, default=strong_cfg.get("dynamic_vocab_top_k", 80))
    parser.add_argument("--no-dynamic-vocab", action="store_true")
    parser.add_argument("--confidence-attack-limit", type=int, default=0)
    parser.add_argument("--confidence-attack-strength", choices=["mild", "balanced", "strong"], default="strong")
    parser.add_argument("--review-sample-size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    modes = FULL_MODES if args.full_matrix and args.modes == DEFAULT_MODES else args.modes
    models = FULL_MODELS if args.full_matrix and args.models == DEFAULT_MODELS else args.models

    if not args.skip_build:
        input_dirs, canonical_jsonl = sources_from_manifest(Path(args.base_manifest))
        if not input_dirs and not canonical_jsonl:
            input_dirs = [("project_msglog", Path(Config.MSG_LOG_DIR))]
        stats = build_ast_dataset(
            input_dirs=input_dirs,
            canonical_jsonl=canonical_jsonl,
            output_dir=dataset_dir,
            train_ratio=Config.STRONG_AST_DATASET_CONFIG["train_ratio"],
            val_ratio=Config.STRONG_AST_DATASET_CONFIG["val_ratio"],
            test_ratio=Config.STRONG_AST_DATASET_CONFIG["test_ratio"],
            seed=args.seed,
            max_variants_spam=args.max_variants_spam,
            max_variants_normal=args.max_variants_normal,
            ast_strength=args.ast_strength,
            use_dynamic_vocab=not args.no_dynamic_vocab,
            dynamic_vocab_top_k=args.dynamic_vocab_top_k,
        )
        print("Strong AST dataset build completed.")
        print(f"Loaded: {stats.loaded}")
        print(f"Kept after dedup: {stats.kept_after_dedup}")
        print(f"Output: {dataset_dir}")

    cfg = RunConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        modes=modes,
        models=models,
        vector_size=args.vector_size,
        max_vocab=args.max_vocab,
        max_len=args.max_len,
        w2v_epochs=args.w2v_epochs,
        clf_epochs=args.clf_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        fgm_epsilon=args.fgm_epsilon,
        focal_gamma=args.focal_gamma,
        learning_rate=args.learning_rate,
        confidence_attack_limit=args.confidence_attack_limit,
        confidence_attack_strength=args.confidence_attack_strength,
        review_sample_size=args.review_sample_size,
        run_ensemble=not args.no_ensemble,
        ensemble_models=args.ensemble_models,
    )
    results = train_and_evaluate(cfg)
    attack_summary = confidence_search_attack(cfg)
    review_summary = ast_quality_review(cfg)
    hf_results = attempt_huggingface_gated(cfg)
    report_path = write_submission_report(cfg, results, attack_summary, review_summary, hf_results)

    print(f"\nStrong AST pipeline completed in {(time.time() - start) / 60:.2f} min")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
