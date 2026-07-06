#!/usr/bin/env python
"""Evaluate a trained output directory on another AST dataset split."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ast_dataset import TextRecord, load_ast_jsonl
from src.ast_metrics import binary_metrics
from src.submission_serving import SubmissionModelService


def evaluate_by_attack(records: Sequence[TextRecord], pred: Sequence[int]) -> Dict[str, Dict[str, object]]:
    grouped_y: Dict[str, List[int]] = defaultdict(list)
    grouped_pred: Dict[str, List[int]] = defaultdict(list)
    for record, pred_id in zip(records, pred):
        key = record.attack_type or "clean"
        grouped_y[key].append(record.label_id)
        grouped_pred[key].append(int(pred_id))
    return {
        key: binary_metrics(grouped_y[key], grouped_pred[key]).__dict__
        for key in sorted(grouped_y)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-evaluate saved models on a separate AST dataset.")
    parser.add_argument("--output-dir", required=True, help="Trained submission output directory.")
    parser.add_argument("--dataset-dir", required=True, help="AST dataset directory containing canonical/*.jsonl.")
    parser.add_argument("--split", default="test_ast")
    parser.add_argument("--name", default="cross_ast", help="Short label used in output filenames.")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)
    records = load_ast_jsonl(dataset_dir / "canonical" / f"{args.split}.jsonl")
    if not records:
        raise SystemExit(f"No records found for {dataset_dir}/canonical/{args.split}.jsonl")

    service = SubmissionModelService(output_dir)
    available = service.available_models()
    selected_modes = args.modes or sorted(available)
    y_true = [record.label_id for record in records]
    texts = [record.text for record in records]

    results: Dict[str, object] = {
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "records": len(records),
        "runs": {},
    }

    for mode in selected_modes:
        model_names = args.models or available.get(mode, [])
        mode_result = {}
        for model_name in model_names:
            if model_name not in available.get(mode, []):
                continue
            predictions = service.predict_many(texts, mode=mode, model_name=model_name)
            y_pred = [int(item["label_id"]) for item in predictions]
            mode_result[model_name] = {
                "metrics": binary_metrics(y_true, y_pred).__dict__,
                "classification_report": classification_report(
                    y_true,
                    y_pred,
                    digits=4,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
                "by_attack": evaluate_by_attack(records, y_pred),
            }
            metrics = mode_result[model_name]["metrics"]
            print(
                f"{mode}/{model_name}: "
                f"acc={metrics['accuracy']:.4f} "
                f"spam_recall={metrics['spam_recall']:.4f}"
            )
        if mode_result:
            results["runs"][mode] = mode_result

    out_path = Path(args.out) if args.out else output_dir / "metrics" / f"{args.name}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cross evaluation written to {out_path}")


if __name__ == "__main__":
    main()
