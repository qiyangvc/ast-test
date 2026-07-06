"""Dataset construction utilities for AST experiments.

This module enforces the experiment protocol used for adversarial spam text:

1. load and canonicalize clean records;
2. deduplicate before splitting;
3. split clean records by label;
4. generate adversarial variants inside each split only;
5. write both metadata-rich JSONL files and legacy ``msg*.log.seg`` folders.

The split-before-generation order is important. If variants are generated first
and then randomly split, near-duplicate texts can leak across train/test and
inflate adversarial robustness metrics.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.adversarial_text import ChineseSpamTextAttacker, compact_segmented_text, segment_for_project


LABEL_TO_PROJECT_ID = {
    "spam": 0,
    "normal": 1,
    "ham": 1,
    "pass": 1,
}

PROJECT_ID_TO_LABEL = {
    0: "spam",
    1: "normal",
}

LEGACY_FILE_BY_LABEL = {
    "spam": "msgspam.log.seg",
    "normal": "msgpass.log.seg",
}


@dataclass
class TextRecord:
    """Canonical text record used by AST experiments."""

    id: str
    source: str
    label: str
    label_id: int
    text: str
    segmented: str
    split: str = "unsplit"
    is_adversarial: bool = False
    attack_type: Optional[str] = None
    parent_id: Optional[str] = None
    operations: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class BuildStats:
    """Summary of a dataset build."""

    loaded: int = 0
    kept_after_dedup: int = 0
    dropped_duplicates: int = 0
    dropped_conflicting_labels: int = 0
    split_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    ast_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    attack_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)


def normalize_label(label: object) -> str:
    """Normalize common label encodings to ``spam`` or ``normal``."""
    if isinstance(label, bool):
        return "spam" if label else "normal"
    if isinstance(label, int):
        if label in PROJECT_ID_TO_LABEL:
            return PROJECT_ID_TO_LABEL[label]
        raise ValueError(f"Unsupported numeric label: {label}")
    text = str(label).strip().lower()
    if text in {"1", "spam", "junk", "positive", "pos", "垃圾", "垃圾短信"}:
        return "spam"
    if text in {"0", "normal", "ham", "pass", "negative", "neg", "正常", "正常短信"}:
        return "normal"
    raise ValueError(f"Unsupported label: {label!r}")


def stable_id(source: str, label: str, text: str, prefix: str = "clean") -> str:
    digest = sha1(f"{source}\0{label}\0{text}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{source}_{label}_{digest}"


def make_record(source: str, label: str, segmented: str, metadata: Optional[Dict[str, object]] = None) -> TextRecord:
    label = normalize_label(label)
    text = compact_segmented_text(segmented)
    record_id = stable_id(source, label, text)
    return TextRecord(
        id=record_id,
        source=source,
        label=label,
        label_id=LABEL_TO_PROJECT_ID[label],
        text=text,
        segmented=segmented.strip() if segmented.strip() else segment_for_project(text),
        metadata=metadata or {},
    )


def load_project_msglog(input_dir: Path, source: str = "project_msglog") -> List[TextRecord]:
    """Load this project's ``msgspam.log.seg`` / ``msgpass.log.seg`` format."""
    input_dir = Path(input_dir)
    records: List[TextRecord] = []
    for label, filename in LEGACY_FILE_BY_LABEL.items():
        path = input_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                records.append(
                    make_record(
                        source=source,
                        label=label,
                        segmented=line,
                        metadata={"input_path": str(path), "line_no": line_no},
                    )
                )
    return records


def load_canonical_jsonl(path: Path, source: Optional[str] = None) -> List[TextRecord]:
    """Load JSONL records with at least ``text`` and ``label`` fields."""
    records: List[TextRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            label = normalize_label(payload["label"])
            dataset_source = source or str(payload.get("source") or Path(path).stem)
            text = str(payload.get("text") or payload.get("segmented") or "")
            segmented = str(payload.get("segmented") or segment_for_project(text))
            metadata = {}
            if payload.get("id"):
                metadata["original_id"] = payload["id"]
            if isinstance(payload.get("metadata"), dict):
                metadata.update(payload["metadata"])
            for key, value in payload.items():
                if key in {
                    "id",
                    "source",
                    "text",
                    "segmented",
                    "label",
                    "label_id",
                    "split",
                    "is_adversarial",
                    "attack_type",
                    "parent_id",
                    "operations",
                    "metadata",
                }:
                    continue
                metadata[key] = value
            metadata.update({"input_path": str(path), "line_no": line_no})
            records.append(make_record(dataset_source, label, segmented, metadata=metadata))
    return records


def load_ast_jsonl(path: Path) -> List[TextRecord]:
    """Load JSONL files written by :func:`build_ast_dataset` without loss."""
    records: List[TextRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            label = normalize_label(payload["label"])
            records.append(
                TextRecord(
                    id=str(payload["id"]),
                    source=str(payload.get("source", "unknown")),
                    label=label,
                    label_id=LABEL_TO_PROJECT_ID[label],
                    text=str(payload.get("text", "")),
                    segmented=str(payload.get("segmented", "")),
                    split=str(payload.get("split", "unsplit")),
                    is_adversarial=bool(payload.get("is_adversarial", False)),
                    attack_type=payload.get("attack_type"),
                    parent_id=payload.get("parent_id"),
                    operations=list(payload.get("operations") or []),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
    return records


def deduplicate_records(records: Sequence[TextRecord]) -> Tuple[List[TextRecord], int, int]:
    """Deduplicate by compact text while dropping conflicting labels."""
    by_text: Dict[str, TextRecord] = {}
    conflicted_texts = set()
    duplicates = 0
    conflicts = 0
    for record in records:
        key = record.text
        if key in conflicted_texts:
            conflicts += 1
            continue
        existing = by_text.get(key)
        if existing is None:
            by_text[key] = record
            continue
        if existing.label != record.label:
            conflicts += 1
            # Ambiguous texts are not useful for clean experiments. Mark the
            # first occurrence as conflicted and remove it entirely.
            by_text.pop(key, None)
            conflicted_texts.add(key)
        else:
            duplicates += 1
    return list(by_text.values()), duplicates, conflicts


def stratified_split(
    records: Sequence[TextRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[TextRecord]]:
    """Split records by label with deterministic shuffling."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    by_label: Dict[str, List[TextRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    splits = {"train": [], "val": [], "test": []}
    for label, label_records in by_label.items():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        split_items = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
        for split, items in split_items.items():
            for item in items:
                item.split = split
            splits[split].extend(items)
    return splits


def generate_ast_records(
    clean_records: Sequence[TextRecord],
    attacker: ChineseSpamTextAttacker,
    max_variants_spam: int = 2,
    max_variants_normal: int = 1,
    ast_strength: str = "mild",
) -> List[TextRecord]:
    """Generate adversarial records for one already-isolated split."""
    ast_records: List[TextRecord] = []
    for record in clean_records:
        max_variants = max_variants_spam if record.label == "spam" else max_variants_normal
        for idx, result in enumerate(
            attacker.generate(
                record.segmented,
                record.label,
                max_variants=max_variants,
                strength=ast_strength,
            )
        ):
            adv_id = f"ast_{record.split}_{record.id}_{idx}"
            ast_records.append(
                TextRecord(
                    id=adv_id,
                    source=record.source,
                    label=record.label,
                    label_id=record.label_id,
                    text=result.adversarial,
                    segmented=segment_for_project(result.adversarial),
                    split=record.split,
                    is_adversarial=True,
                    attack_type=result.attack_type,
                    parent_id=record.id,
                    operations=result.operations,
                    metadata={"parent_text": record.text},
                )
            )
    return ast_records


def count_by_label(records: Iterable[TextRecord]) -> Dict[str, int]:
    return dict(Counter(record.label for record in records))


def count_by_attack(records: Iterable[TextRecord]) -> Dict[str, int]:
    return dict(Counter(record.attack_type or "clean" for record in records))


def write_jsonl(path: Path, records: Iterable[TextRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")


def write_legacy_msglog(path: Path, records: Iterable[TextRecord]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[TextRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)
    for label, filename in LEGACY_FILE_BY_LABEL.items():
        with (path / filename).open("w", encoding="utf-8") as handle:
            for record in grouped.get(label, []):
                handle.write(record.segmented.strip() + "\n")


def write_manifest(path: Path, stats: BuildStats, settings: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": dict(settings),
        "stats": asdict(stats),
        "label_mapping": LABEL_TO_PROJECT_ID,
        "protocol": [
            "deduplicate clean records",
            "split clean records by label",
            "generate adversarial records inside each split",
            "keep test_ast isolated from training",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_ast_dataset(
    input_dirs: Sequence[Tuple[str, Path]],
    output_dir: Path,
    canonical_jsonl: Optional[Sequence[Tuple[str, Path]]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    max_variants_spam: int = 2,
    max_variants_normal: int = 1,
    ast_strength: str = "mild",
) -> BuildStats:
    """Build a full clean/AST dataset directory without training models."""
    output_dir = Path(output_dir)
    all_records: List[TextRecord] = []

    for source, input_dir in input_dirs:
        all_records.extend(load_project_msglog(Path(input_dir), source=source))

    for source, path in canonical_jsonl or []:
        all_records.extend(load_canonical_jsonl(Path(path), source=source))

    kept_records, duplicates, conflicts = deduplicate_records(all_records)
    splits = stratified_split(kept_records, train_ratio, val_ratio, test_ratio, seed=seed)

    attacker = ChineseSpamTextAttacker(seed=seed)
    ast_splits = {
        split: generate_ast_records(
            records,
            attacker,
            max_variants_spam=max_variants_spam,
            max_variants_normal=max_variants_normal,
            ast_strength=ast_strength,
        )
        for split, records in splits.items()
    }

    stats = BuildStats(
        loaded=len(all_records),
        kept_after_dedup=len(kept_records),
        dropped_duplicates=duplicates,
        dropped_conflicting_labels=conflicts,
    )

    canonical_dir = output_dir / "canonical"
    legacy_dir = output_dir / "legacy"

    for split, clean_records in splits.items():
        ast_records = ast_splits[split]
        clean_ast_records = list(clean_records) + list(ast_records)

        write_jsonl(canonical_dir / f"{split}_clean.jsonl", clean_records)
        write_jsonl(canonical_dir / f"{split}_ast.jsonl", ast_records)
        write_jsonl(canonical_dir / f"{split}_clean_ast.jsonl", clean_ast_records)

        write_legacy_msglog(legacy_dir / f"{split}_clean", clean_records)
        write_legacy_msglog(legacy_dir / f"{split}_ast", ast_records)
        write_legacy_msglog(legacy_dir / f"{split}_clean_ast", clean_ast_records)

        stats.split_counts[split] = count_by_label(clean_records)
        stats.ast_counts[split] = count_by_label(ast_records)
        stats.attack_counts[split] = count_by_attack(ast_records)

    settings = {
        "input_dirs": [(source, str(path)) for source, path in input_dirs],
        "canonical_jsonl": [(source, str(path)) for source, path in canonical_jsonl or []],
        "output_dir": str(output_dir),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "max_variants_spam": max_variants_spam,
        "max_variants_normal": max_variants_normal,
        "ast_strength": ast_strength,
    }
    write_manifest(output_dir / "manifest.json", stats, settings)
    return stats
