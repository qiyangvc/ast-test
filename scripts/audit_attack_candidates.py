#!/usr/bin/env python
"""Audit AST candidate breadth before training/evaluation."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adversarial_text import ChineseSpamTextAttacker
from src.ast_dataset import TextRecord, load_ast_jsonl


def resolve_path(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def load_split(dataset_dir: Path, split_name: str) -> List[TextRecord]:
    path = dataset_dir / "canonical" / f"{split_name}.jsonl"
    return load_ast_jsonl(path) if path.exists() else []


def candidate_audit(
    records: Sequence[TextRecord],
    strength: str,
    max_variants: int,
    seed: int,
    seed_count: int,
) -> Dict[str, object]:
    attacker = ChineseSpamTextAttacker(seed=seed)
    variant_counts = Counter()
    attack_counts = Counter()
    operation_counts = Counter()
    length_ratios = []
    no_candidate_examples = []
    low_candidate_examples = []

    for record in records:
        candidates = []
        for offset in range(seed_count):
            attacker.rng.seed(seed + offset)
            candidates.extend(
                attacker.generate(
                    record.segmented,
                    record.label,
                    max_variants=max_variants,
                    strength=strength,
                )
            )
        unique = {item.adversarial: item for item in candidates}
        variant_counts[len(unique)] += 1
        if not unique:
            no_candidate_examples.append({"id": record.id, "text": record.text})
            continue
        if len(unique) <= 2:
            low_candidate_examples.append({"id": record.id, "text": record.text, "unique_candidates": len(unique)})
        for item in unique.values():
            attack_counts[item.attack_type] += 1
            for operation in item.operations:
                operation_counts[operation.split("->", 1)[0]] += 1
            length_ratios.append(len(item.adversarial) / max(len(item.original), 1))

    counts = list(variant_counts.elements())
    return {
        "records": len(records),
        "strength": strength,
        "max_variants_per_seed": max_variants,
        "seed_count": seed_count,
        "records_with_candidates": len(records) - len(no_candidate_examples),
        "candidate_generation_rate": (len(records) - len(no_candidate_examples)) / max(len(records), 1),
        "avg_unique_candidates": round(mean(counts), 2) if counts else 0.0,
        "variant_count_histogram": dict(sorted(variant_counts.items())),
        "attack_type_counts": dict(attack_counts.most_common()),
        "top_operation_keys": dict(operation_counts.most_common(40)),
        "avg_length_ratio": round(mean(length_ratios), 4) if length_ratios else 0.0,
        "no_candidate_examples": no_candidate_examples[:40],
        "low_candidate_examples": low_candidate_examples[:40],
    }


def summarize_confidence_attack(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    by_attack = defaultdict(lambda: {"rows": 0, "successes": 0, "normal_prob_delta": 0.0})
    for row in rows:
        attack_type = row.get("attack_type") or "original"
        item = by_attack[attack_type]
        item["rows"] += 1
        item["successes"] += int(bool(row.get("success")))
        item["normal_prob_delta"] += float(row.get("best_normal_prob", 0.0)) - float(row.get("original_normal_prob", 0.0))
    by_attack_out = {}
    for attack_type, item in by_attack.items():
        by_attack_out[attack_type] = {
            "rows": item["rows"],
            "successes": item["successes"],
            "success_rate": item["successes"] / max(item["rows"], 1),
            "avg_normal_prob_delta": item["normal_prob_delta"] / max(item["rows"], 1),
        }
    return {
        "exists": True,
        "path": str(path),
        "rows": len(rows),
        "successes": sum(int(bool(row.get("success"))) for row in rows),
        "success_rate": sum(int(bool(row.get("success"))) for row in rows) / max(len(rows), 1),
        "by_attack_type": by_attack_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AST candidate generation breadth without training.")
    parser.add_argument("--dataset-dir", default="data/ast_experiment_strong")
    parser.add_argument("--split", default="test_clean")
    parser.add_argument("--label", choices=["spam", "normal", "all"], default="spam")
    parser.add_argument("--strength", choices=["mild", "balanced", "strong"], default="strong")
    parser.add_argument("--max-variants", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=1000, help="0 means all matching records.")
    parser.add_argument("--confidence-attack-jsonl", default="", help="Optional confidence-search JSONL to summarize.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    dataset_dir = resolve_path(args.dataset_dir)
    records = load_split(dataset_dir, args.split)
    if args.label != "all":
        records = [record for record in records if record.label == args.label]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.sample_size > 0:
        records = records[: args.sample_size]

    payload = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "label": args.label,
        "sample_size": len(records),
        "candidate_audit": candidate_audit(records, args.strength, args.max_variants, args.seed, args.seed_count),
    }
    if args.confidence_attack_jsonl:
        payload["confidence_attack_summary"] = summarize_confidence_attack(resolve_path(args.confidence_attack_jsonl))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        out_path = resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
