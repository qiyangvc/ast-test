#!/usr/bin/env python
"""Audit trained vocabulary coverage and OOV rates on clean/AST data."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adversarial_text import ast_lexicon_terms, segment_for_project
from src.ast_dataset import TextRecord, load_ast_jsonl, load_canonical_jsonl


def resolve_path(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def tokens_for_record(record: TextRecord) -> List[str]:
    segmented = record.segmented or segment_for_project(record.text)
    return [token for token in segmented.split() if token]


def load_dataset_records(dataset_dir: Path, splits: Sequence[str]) -> Dict[str, List[TextRecord]]:
    records = {}
    for split in splits:
        path = dataset_dir / "canonical" / f"{split}.jsonl"
        records[split] = load_ast_jsonl(path) if path.exists() else []
    return records


def parse_extra_jsonl(items: Sequence[str]) -> Dict[str, List[TextRecord]]:
    result = {}
    for item in items:
        if "=" in item:
            name, path_text = item.split("=", 1)
        else:
            path = Path(item)
            name, path_text = path.stem, item
        path = resolve_path(path_text)
        if path.exists():
            result[name] = load_canonical_jsonl(path)
    return result


def load_vocab(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(word): int(idx) for word, idx in payload["word2idx"].items()}


def infer_max_len(output_dir: Path, fallback: int) -> int:
    metrics_path = output_dir / "metrics/all_results.json"
    if not metrics_path.exists():
        return fallback
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        return int((payload.get("config") or {}).get("max_len") or fallback)
    except (TypeError, ValueError):
        return fallback


def audit_records(records: Sequence[TextRecord], word2idx: Dict[str, int], max_len: int) -> Dict[str, object]:
    unknown = Counter()
    by_label = defaultdict(lambda: {"records": 0, "tokens": 0, "unknown_tokens": 0})
    total_tokens = 0
    total_unknown = 0
    truncated_tokens = 0
    empty_records = 0
    for record in records:
        tokens = tokens_for_record(record)
        if not tokens:
            empty_records += 1
        truncated_tokens += max(len(tokens) - max_len, 0)
        model_tokens = tokens[:max_len]
        total_tokens += len(model_tokens)
        row = by_label[record.label]
        row["records"] += 1
        row["tokens"] += len(model_tokens)
        for token in model_tokens:
            if token not in word2idx:
                unknown[token] += 1
                total_unknown += 1
                row["unknown_tokens"] += 1

    by_label_out = {}
    for label, row in by_label.items():
        by_label_out[label] = {
            **row,
            "unknown_rate": row["unknown_tokens"] / max(row["tokens"], 1),
        }

    return {
        "records": len(records),
        "empty_records": empty_records,
        "model_tokens": total_tokens,
        "unknown_tokens": total_unknown,
        "unknown_rate": total_unknown / max(total_tokens, 1),
        "truncated_tokens": truncated_tokens,
        "by_label": by_label_out,
        "top_unknown_tokens": [{"token": token, "count": count} for token, count in unknown.most_common(30)],
    }


def audit_lexicon_terms(word2idx: Dict[str, int], terms: Sequence[str]) -> Dict[str, object]:
    exact_hits = [term for term in terms if term in word2idx]
    segmented_hits = []
    segmented_misses = []
    for term in terms:
        tokens = [token for token in segment_for_project(term).split() if token]
        if tokens and all(token in word2idx for token in tokens):
            segmented_hits.append(term)
        else:
            segmented_misses.append({"term": term, "tokens": tokens})
    return {
        "terms": len(terms),
        "exact_hits": len(exact_hits),
        "exact_hit_rate": len(exact_hits) / max(len(terms), 1),
        "segmented_hits": len(segmented_hits),
        "segmented_hit_rate": len(segmented_hits) / max(len(terms), 1),
        "segmented_miss_examples": segmented_misses[:40],
    }


def audit_vocab(
    vocab_path: Path,
    records_by_name: Dict[str, List[TextRecord]],
    max_len: int,
    lexicon_terms: Sequence[str],
) -> Dict[str, object]:
    word2idx = load_vocab(vocab_path)
    return {
        "vocab_path": str(vocab_path),
        "vocab_size": len(word2idx),
        "max_len": max_len,
        "lexicon_coverage": audit_lexicon_terms(word2idx, lexicon_terms),
        "splits": {
            name: audit_records(records, word2idx, max_len)
            for name, records in records_by_name.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OOV rates for saved vocabularies without training.")
    parser.add_argument("--dataset-dir", default="data/ast_experiment_strong")
    parser.add_argument("--output-dir", default="output/submission_strong_ast_20260706_full")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Canonical split name without .jsonl. Can be repeated.",
    )
    parser.add_argument(
        "--extra-jsonl",
        action="append",
        default=None,
        help="Extra canonical JSONL as NAME=PATH. Can be repeated.",
    )
    parser.add_argument("--max-len", type=int, default=0, help="Override model max_len. Default reads metrics config.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    max_len = args.max_len or infer_max_len(output_dir, fallback=64)
    splits = args.split or ["train_clean", "train_clean_ast", "val_clean", "test_clean", "test_ast"]
    extra_jsonl = args.extra_jsonl or ["uci_en=data/external/canonical/uci_sms_spam_collection.jsonl"]
    records_by_name = load_dataset_records(dataset_dir, splits)
    records_by_name.update(parse_extra_jsonl(extra_jsonl))
    terms = ast_lexicon_terms()

    vocab_paths = sorted((output_dir / "word2vec").glob("vocab_*.json"))
    payload = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "vocab_count": len(vocab_paths),
        "records": {name: len(records) for name, records in records_by_name.items()},
        "audits": [audit_vocab(path, records_by_name, max_len, terms) for path in vocab_paths],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        out_path = resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
