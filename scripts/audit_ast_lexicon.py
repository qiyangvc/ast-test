#!/usr/bin/env python
"""Audit the external AST lexicon against canonical clean data.

This script does not train models and does not rebuild datasets. It checks
whether the text-level AST lexicon is broad enough for the current corpus.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.adversarial_text import ChineseSpamTextAttacker, DEFAULT_LEXICON_PATH
from src.ast_dataset import TextRecord, load_ast_jsonl


def load_clean_records(dataset_dir: Path) -> List[TextRecord]:
    records: List[TextRecord] = []
    canonical_dir = dataset_dir / "canonical"
    for split in ("train", "val", "test"):
        records.extend(load_ast_jsonl(canonical_dir / f"{split}_clean.jsonl"))
    return records


def lexicon_keys(attacker: ChineseSpamTextAttacker) -> List[str]:
    terms = set()
    terms.update(attacker.PHRASE_VARIANTS)
    terms.update(attacker.STRONG_PHRASE_VARIANTS)
    terms.update(attacker.PINYIN_ABBREVIATIONS)
    terms.update(attacker.MULTI_KEYWORD_OBFUSCATION_KEYWORDS)
    terms.update(key for key, _ in attacker.BENEFIT_HINTS)
    terms.update(key for key, _ in attacker.AMOUNT_FALLBACKS)
    terms.update(key for key, _ in attacker.URL_KEYWORD_REPLACEMENTS)
    return sorted(terms, key=len, reverse=True)


def record_has_any_term(text: str, terms: Sequence[str]) -> bool:
    return any(term and term in text for term in terms)


def generation_audit(
    attacker: ChineseSpamTextAttacker,
    records: Sequence[TextRecord],
    sample_size: int,
    strength: str,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    sample = [record for record in records if record.label == "spam"]
    rng.shuffle(sample)
    if sample_size > 0:
        sample = sample[:sample_size]

    no_variant: List[str] = []
    attack_counts: Counter[str] = Counter()
    variant_counts: Counter[int] = Counter()
    for record in sample:
        variants = attacker.generate(record.segmented, record.label, max_variants=8, strength=strength)
        variant_counts[len(variants)] += 1
        if not variants:
            no_variant.append(record.id)
        for item in variants:
            attack_counts[item.attack_type] += 1

    generated = len(sample) - len(no_variant)
    return {
        "sample_size": len(sample),
        "generated_records": generated,
        "no_variant_records": len(no_variant),
        "generation_rate": generated / max(len(sample), 1),
        "variant_count_histogram": dict(sorted(variant_counts.items())),
        "attack_type_counts": dict(attack_counts.most_common()),
        "no_variant_example_ids": no_variant[:20],
    }


def uncovered_terms(records: Sequence[TextRecord], terms: Sequence[str], min_spam_count: int) -> List[Dict[str, object]]:
    counts = {"spam": Counter(), "normal": Counter()}
    docs = {"spam": 0, "normal": 0}
    for record in records:
        docs[record.label] += 1
        tokens = {token for token in record.segmented.split() if 2 <= len(token) <= 12}
        counts[record.label].update(tokens)

    rows = []
    known = set(terms)
    for token, spam_count in counts["spam"].items():
        if spam_count < min_spam_count:
            continue
        if any(term in token or token in term for term in known):
            continue
        normal_count = counts["normal"][token]
        spam_rate = (spam_count + 1) / (docs["spam"] + 2)
        normal_rate = (normal_count + 1) / (docs["normal"] + 2)
        score = math.log(spam_rate / normal_rate) * math.log1p(spam_count)
        rows.append(
            {
                "term": token,
                "spam_docs": spam_count,
                "normal_docs": normal_count,
                "score": round(score, 4),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the external AST lexicon without training models.")
    parser.add_argument("--dataset-dir", default="data/ast_experiment_strong", help="AST dataset directory.")
    parser.add_argument("--lexicon", default=str(DEFAULT_LEXICON_PATH), help="External AST lexicon JSON path.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Spam records sampled for generation audit. 0 means all.")
    parser.add_argument("--strength", choices=["mild", "balanced", "strong"], default="strong")
    parser.add_argument("--min-spam-count", type=int, default=50, help="Minimum spam document frequency for uncovered terms.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    lexicon_path = Path(args.lexicon)
    if not lexicon_path.is_absolute():
        lexicon_path = PROJECT_ROOT / lexicon_path

    attacker = ChineseSpamTextAttacker(seed=args.seed, lexicon_path=lexicon_path)
    records = load_clean_records(dataset_dir)
    terms = lexicon_keys(attacker)
    spam_records = [record for record in records if record.label == "spam"]
    normal_records = [record for record in records if record.label == "normal"]

    coverage = {
        "spam_records": len(spam_records),
        "normal_records": len(normal_records),
        "spam_with_lexicon_term": sum(record_has_any_term(record.text, terms) for record in spam_records),
        "normal_with_lexicon_term": sum(record_has_any_term(record.text, terms) for record in normal_records),
    }
    coverage["spam_term_coverage"] = coverage["spam_with_lexicon_term"] / max(len(spam_records), 1)
    coverage["normal_term_coverage"] = coverage["normal_with_lexicon_term"] / max(len(normal_records), 1)

    payload = {
        "dataset_dir": str(dataset_dir),
        "lexicon_path": str(lexicon_path),
        "lexicon_summary": attacker.lexicon_summary(),
        "term_coverage": coverage,
        "generation_audit": generation_audit(attacker, records, args.sample_size, args.strength, args.seed),
        "uncovered_high_frequency_spam_terms": uncovered_terms(records, terms, args.min_spam_count),
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
