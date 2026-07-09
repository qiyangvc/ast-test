#!/usr/bin/env python
"""Rule-based quality audit for generated AST JSONL files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ast_dataset import TextRecord, load_ast_jsonl


AGGRESSIVE_NORMAL_ATTACKS = {
    "contact_obfuscation",
    "mixed",
    "strong_mixed",
    "semantic_rewrite",
    "pinyin_abbreviation",
    "multi_keyword_obfuscation",
    "contact_split",
    "url_obfuscation",
    "amount_obfuscation",
    "strong_phrase_variant",
}


def resolve_path(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def load_split(dataset_dir: Path, split_name: str) -> List[TextRecord]:
    path = dataset_dir / "canonical" / f"{split_name}.jsonl"
    return load_ast_jsonl(path) if path.exists() else []


def parent_lookup(dataset_dir: Path) -> Dict[str, TextRecord]:
    parents = {}
    for split in ("train", "val", "test"):
        for record in load_split(dataset_dir, f"{split}_clean"):
            parents[record.id] = record
    return parents


def quality_flags(record: TextRecord, parents: Dict[str, TextRecord]) -> Tuple[List[str], Dict[str, object]]:
    flags: List[str] = []
    parent = parents.get(record.parent_id or "")
    parent_text = str(record.metadata.get("parent_text") or (parent.text if parent else ""))
    if not record.text.strip():
        flags.append("empty_text")
    if not record.is_adversarial:
        flags.append("is_adversarial_false")
    if not record.attack_type:
        flags.append("missing_attack_type")
    if not record.operations:
        flags.append("empty_operations")
    if not record.parent_id or parent is None:
        flags.append("missing_parent")
    if parent_text and record.text == parent_text:
        flags.append("unchanged_from_parent")
    if record.label == "normal" and record.attack_type in AGGRESSIVE_NORMAL_ATTACKS:
        flags.append("aggressive_attack_on_normal")

    parent_len = max(len(parent_text), 1)
    ratio = len(record.text) / parent_len
    if parent_text and ratio < 0.35:
        flags.append("length_ratio_too_short")
    if parent_text and ratio > 2.8:
        flags.append("length_ratio_too_long")
    if len(record.text) < 2:
        flags.append("too_short")

    detail = {
        "id": record.id,
        "split": record.split,
        "label": record.label,
        "attack_type": record.attack_type,
        "parent_id": record.parent_id,
        "parent_text": parent_text,
        "text": record.text,
        "length_ratio": round(ratio, 4),
        "operations": record.operations,
        "flags": flags,
    }
    return flags, detail


def audit_dataset(dataset_dir: Path, split_names: Sequence[str]) -> Dict[str, object]:
    parents = parent_lookup(dataset_dir)
    split_rows = {}
    all_ast: List[TextRecord] = []
    suspicious: List[Dict[str, object]] = []
    for split_name in split_names:
        records = load_split(dataset_dir, split_name)
        all_ast.extend(records)
        flag_counts = Counter()
        attack_counts = Counter(record.attack_type or "missing" for record in records)
        label_counts = Counter(record.label for record in records)
        flagged_records = 0
        for record in records:
            flags, detail = quality_flags(record, parents)
            flag_counts.update(flags)
            if flags:
                flagged_records += 1
                suspicious.append(detail)
        split_rows[split_name] = {
            "records": len(records),
            "label_counts": dict(label_counts),
            "attack_counts": dict(attack_counts.most_common()),
            "flag_counts": dict(flag_counts.most_common()),
            "flagged_records": flagged_records,
        }

    by_text = defaultdict(list)
    for record in all_ast:
        by_text[record.text].append(record.id)
    duplicate_texts = {text: ids for text, ids in by_text.items() if len(ids) > 1}
    duplicate_records = sum(len(ids) for ids in duplicate_texts.values())
    flagged_record_ids = {row["id"] for row in suspicious}
    return {
        "dataset_dir": str(dataset_dir),
        "splits": split_rows,
        "total_ast_records": len(all_ast),
        "unique_ast_texts": len(by_text),
        "duplicate_ast_text_groups": len(duplicate_texts),
        "duplicate_ast_records": duplicate_records,
        "flagged_unique_records": len(flagged_record_ids),
        "flag_rate": len(flagged_record_ids) / max(len(all_ast), 1),
        "duplicate_text_examples": [
            {"text": text, "ids": ids[:8], "count": len(ids)}
            for text, ids in list(duplicate_texts.items())[:30]
        ],
        "suspicious_examples": suspicious[:80],
        "note": "Programmatic audit only; semantic label preservation still needs human review for final submission.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AST quality without training or inference.")
    parser.add_argument("--dataset-dir", default="data/ast_experiment_strong")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="AST split name without .jsonl. Can be repeated.",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--review-jsonl", default="", help="Optional JSONL file with suspicious records.")
    args = parser.parse_args()

    dataset_dir = resolve_path(args.dataset_dir)
    split_names = args.split or ["train_ast", "val_ast", "test_ast"]
    payload = audit_dataset(dataset_dir, split_names)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        out_path = resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.review_jsonl:
        review_path = resolve_path(args.review_jsonl)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", encoding="utf-8") as handle:
            for row in payload["suspicious_examples"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
