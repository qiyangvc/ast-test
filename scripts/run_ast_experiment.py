#!/usr/bin/env python
"""Plan or run AST experiments.

Default behavior is dry-run planning. Add ``--execute`` only when you are ready
to train and evaluate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ast_experiment import ASTExperimentConfig, run_ast_experiment, write_dry_run_plan
from src.config import Config


def main() -> None:
    exp_cfg = Config.AST_EXPERIMENT_CONFIG
    parser = argparse.ArgumentParser(description="Dry-run or execute AST experiments.")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(Config.DATA_DIR) / Config.AST_DATASET_CONFIG["output_dir_name"]),
        help="Directory produced by scripts/build_ast_dataset.py.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(Path(Config.OUTPUT_DIR) / "ast_runs"),
        help="Experiment runtime output directory.",
    )
    parser.add_argument(
        "--mode",
        choices=exp_cfg["modes"],
        default=exp_cfg["default_mode"],
        help="Experiment mode.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=exp_cfg["models"],
        default=exp_cfg["default_models"],
        help="Models to include.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train and evaluate. Omit this flag for dry-run planning.",
    )
    parser.add_argument("--seed", type=int, default=Config.AST_DATASET_CONFIG["seed"])
    parser.add_argument("--display-step", type=int, default=50)
    parser.add_argument(
        "--plan-output",
        default=None,
        help="Optional JSON file for dry-run plan output.",
    )
    args = parser.parse_args()

    config = ASTExperimentConfig(
        dataset_dir=Path(args.dataset_dir).expanduser(),
        work_dir=Path(args.work_dir).expanduser(),
        mode=args.mode,
        models=args.models,
        execute=args.execute,
        seed=args.seed,
        display_step=args.display_step,
    )
    result = run_ast_experiment(config)

    if result.get("status") == "dry_run":
        print("AST experiment dry-run plan:")
        for item in result["plan"]:
            print(f"- {item}")
        if result["missing"]:
            print("\nMissing required dataset files/directories:")
            for item in result["missing"]:
                print(f"- {item}")
        if args.plan_output:
            write_dry_run_plan(Path(args.plan_output).expanduser(), result)
            print(f"\nPlan written to: {Path(args.plan_output).expanduser()}")
    else:
        print("AST experiment completed.")
        print(result)


if __name__ == "__main__":
    main()
