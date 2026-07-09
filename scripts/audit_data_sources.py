#!/usr/bin/env python
"""Audit external sources and built AST datasets without training models."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ast_dataset import TextRecord, deduplicate_records, load_ast_jsonl, load_canonical_jsonl


def resolve_path(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def label_counts(records: Iterable[TextRecord]) -> Dict[str, int]:
    return dict(Counter(record.label for record in records))


def source_counts(records: Iterable[TextRecord]) -> Dict[str, int]:
    return dict(Counter(record.source for record in records))


def length_stats(records: Sequence[TextRecord]) -> Dict[str, float]:
    if not records:
        return {"records": 0, "avg_chars": 0.0, "p95_chars": 0.0, "avg_tokens": 0.0, "p95_tokens": 0.0}
    char_lengths = sorted(len(record.text) for record in records)
    token_lengths = sorted(len(record.segmented.split()) for record in records)
    p95_idx = min(int(len(records) * 0.95), len(records) - 1)
    return {
        "records": len(records),
        "avg_chars": round(mean(char_lengths), 2),
        "p95_chars": float(char_lengths[p95_idx]),
        "avg_tokens": round(mean(token_lengths), 2),
        "p95_tokens": float(token_lengths[p95_idx]),
    }


def audit_external_sources(external_dir: Path) -> Dict[str, object]:
    canonical_dir = external_dir / "canonical"
    files = sorted(canonical_dir.glob("*.jsonl")) if canonical_dir.exists() else []
    all_records: List[TextRecord] = []
    file_rows = []
    for path in files:
        records = load_canonical_jsonl(path)
        all_records.extend(records)
        file_rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
                "line_count": read_jsonl_count(path),
                "loaded_records": len(records),
                "label_counts": label_counts(records),
                "source_counts": source_counts(records),
                "length_stats": length_stats(records),
            }
        )

    kept, duplicates, conflicts = deduplicate_records(all_records)
    return {
        "external_dir": str(external_dir),
        "canonical_files": file_rows,
        "total_loaded": len(all_records),
        "total_label_counts": label_counts(all_records),
        "total_source_counts": source_counts(all_records),
        "dedup": {
            "kept_after_dedup": len(kept),
            "dropped_duplicates": duplicates,
            "dropped_conflicting_labels": conflicts,
        },
        "length_stats": length_stats(all_records),
    }


def load_split(dataset_dir: Path, split_file: str) -> List[TextRecord]:
    path = dataset_dir / "canonical" / split_file
    return load_ast_jsonl(path) if path.exists() else []


def overlap_counts(clean_splits: Dict[str, List[TextRecord]]) -> Dict[str, int]:
    by_split = {name: {record.text for record in records} for name, records in clean_splits.items()}
    overlaps = {}
    for left, right in combinations(sorted(by_split), 2):
        overlaps[f"{left}__{right}"] = len(by_split[left].intersection(by_split[right]))
    return overlaps


def parent_mismatches(dataset_dir: Path) -> Dict[str, int]:
    result = {}
    for split in ("train", "val", "test"):
        clean = load_split(dataset_dir, f"{split}_clean.jsonl")
        ast = load_split(dataset_dir, f"{split}_ast.jsonl")
        clean_ids = {record.id for record in clean}
        result[split] = sum(1 for record in ast if record.parent_id not in clean_ids)
    return result


def audit_dataset_dir(dataset_dir: Path) -> Dict[str, object]:
    canonical_dir = dataset_dir / "canonical"
    manifest_path = dataset_dir / "manifest.json"
    clean_splits = {split: load_split(dataset_dir, f"{split}_clean.jsonl") for split in ("train", "val", "test")}
    rows = {}
    for split in ("train", "val", "test"):
        for suffix in ("clean", "ast", "clean_ast"):
            name = f"{split}_{suffix}"
            records = load_split(dataset_dir, f"{name}.jsonl")
            rows[name] = {
                "records": len(records),
                "label_counts": label_counts(records),
                "source_counts": source_counts(records),
                "attack_counts": dict(Counter(record.attack_type or "clean" for record in records)),
                "length_stats": length_stats(records),
            }

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "dataset_dir": str(dataset_dir),
        "canonical_exists": canonical_dir.exists(),
        "manifest": {
            "exists": manifest_path.exists(),
            "settings": manifest.get("settings", {}),
            "stats": manifest.get("stats", {}),
        },
        "splits": rows,
        "clean_text_overlap": overlap_counts(clean_splits),
        "ast_parent_id_mismatches": parent_mismatches(dataset_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data sources, splits, deduplication, and AST leakage checks.")
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=None,
        help="AST dataset directory. Can be repeated.",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()
    dataset_dirs = args.dataset_dir or ["data/ast_experiment", "data/ast_experiment_strong"]

    payload = {
        "external": audit_external_sources(resolve_path(args.external_dir)),
        "datasets": [audit_dataset_dir(resolve_path(path)) for path in dataset_dirs],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        out_path = resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
