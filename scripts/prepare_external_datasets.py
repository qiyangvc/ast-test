#!/usr/bin/env python
"""Download and normalize external spam-text datasets.

The script prepares data only. It does not build AST splits, train models, or
run evaluation. Outputs live under ``data/external`` by default, which is
already ignored by git through the repository's ``data/`` rule.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ast_dataset import TextRecord, make_record, write_jsonl
from src.adversarial_text import segment_for_project
from src.config import Config


TENSORLAYER_MSGLOG_URL = (
    "https://raw.githubusercontent.com/tensorlayer/text-antispam/master/word2vec/data/msglog.tar.gz"
)
SPAM_MESSAGES_LR_URL = "https://raw.githubusercontent.com/x-hacker/SpamMessagesLR/master/train.txt"
FBS_REPO_URL = "https://github.com/Cypher-Z/FBS_SMS_Dataset.git"
UCI_SMS_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_file(url: str, path: Path, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def clone_repo(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {url}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)


def prepare_tensorlayer_msglog(raw_dir: Path, canonical_dir: Path, force: bool = False) -> Path:
    """Download original project-compatible msglog data and write JSONL."""
    archive = raw_dir / "tensorlayer_text_antispam" / "msglog.tar.gz"
    extract_root = raw_dir / "tensorlayer_text_antispam"
    msglog_dir = extract_root / "msglog"
    download_file(TENSORLAYER_MSGLOG_URL, archive, force=force)
    if force and msglog_dir.exists():
        shutil.rmtree(msglog_dir)
    if not msglog_dir.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_root)

    records: List[TextRecord] = []
    for label, filename in (("normal", "msgpass.log.seg"), ("spam", "msgspam.log.seg")):
        path = msglog_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    records.append(
                        make_record(
                            "tensorlayer_text_antispam",
                            label,
                            line,
                            metadata={
                                "dataset_url": TENSORLAYER_MSGLOG_URL,
                                "input_path": str(path),
                                "line_no": line_no,
                            },
                        )
                    )
    output = canonical_dir / "tensorlayer_text_antispam.jsonl"
    write_jsonl(output, records)
    print(f"Prepared tensorlayer_text_antispam: {len(records)} records -> {output}")
    return msglog_dir


def prepare_spam_messages_lr(raw_dir: Path, canonical_dir: Path, force: bool = False) -> Path:
    """Download SpamMessagesLR and normalize label<TAB>text lines."""
    path = raw_dir / "spam_messages_lr" / "train.txt"
    download_file(SPAM_MESSAGES_LR_URL, path, force=force)
    records: List[TextRecord] = []
    skipped_empty = 0
    skipped_malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                skipped_empty += 1
                continue
            if "\t" not in line:
                skipped_malformed += 1
                continue
            raw_label, text = line.split("\t", 1)
            if not text.strip():
                skipped_empty += 1
                continue
            # Dataset convention observed in train.txt: 1=spam, 0=normal.
            label = "spam" if raw_label.strip() == "1" else "normal"
            records.append(
                make_record(
                    "spam_messages_lr",
                    label,
                    segment_for_project(text),
                    metadata={
                        "dataset_url": SPAM_MESSAGES_LR_URL,
                        "raw_label": raw_label,
                        "input_path": str(path),
                        "line_no": line_no,
                    },
                )
            )
    output = canonical_dir / "spam_messages_lr.jsonl"
    write_jsonl(output, records)
    print(f"Prepared spam_messages_lr: {len(records)} records -> {output}")
    if skipped_empty or skipped_malformed:
        print(
            "SpamMessagesLR skipped lines: "
            f"empty_text={skipped_empty}, malformed={skipped_malformed}"
        )
    return output


def prepare_fbs_sms(raw_dir: Path, canonical_dir: Path, force: bool = False) -> Path:
    """Clone FBS SMS dataset and normalize all category files as spam."""
    repo_dir = raw_dir / "fbs_sms_dataset" / "repo"
    clone_repo(FBS_REPO_URL, repo_dir, force=force)
    records: List[TextRecord] = []
    for path in sorted(repo_dir.iterdir()):
        if path.name.startswith(".") or path.name == "README.md" or path.is_dir():
            continue
        if not path.is_file():
            continue
        category = path.name
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                records.append(
                    make_record(
                        "fbs_sms_dataset",
                        "spam",
                        line,
                        metadata={
                            "dataset_url": FBS_REPO_URL,
                            "category": category,
                            "input_path": str(path),
                            "line_no": line_no,
                        },
                    )
                )
    output = canonical_dir / "fbs_sms_dataset.jsonl"
    write_jsonl(output, records)
    print(f"Prepared fbs_sms_dataset: {len(records)} records -> {output}")
    return output


def prepare_uci_sms(raw_dir: Path, canonical_dir: Path, force: bool = False) -> Path:
    """Download UCI SMS Spam Collection as optional English external data."""
    archive = raw_dir / "uci_sms_spam_collection" / "sms_spam_collection.zip"
    extract_root = raw_dir / "uci_sms_spam_collection" / "unzipped"
    download_file(UCI_SMS_ZIP_URL, archive, force=force)
    if force and extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_root)

    data_path = extract_root / "SMSSpamCollection"
    records: List[TextRecord] = []
    with data_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            raw_label, text = line.split("\t", 1)
            label = "spam" if raw_label == "spam" else "normal"
            records.append(
                make_record(
                    "uci_sms_spam_collection",
                    label,
                    segment_for_project(text),
                    metadata={
                        "dataset_url": UCI_SMS_ZIP_URL,
                        "language": "en",
                        "raw_label": raw_label,
                        "input_path": str(data_path),
                        "line_no": line_no,
                    },
                )
            )
    output = canonical_dir / "uci_sms_spam_collection.jsonl"
    write_jsonl(output, records)
    print(f"Prepared uci_sms_spam_collection: {len(records)} records -> {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare external datasets without training or testing.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(Config.DATA_DIR) / "external"),
        help="Base directory for raw downloads and canonical JSONL files.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["tensorlayer", "spam_lr", "fbs", "uci_en"],
        default=["tensorlayer", "spam_lr", "fbs"],
        help="Datasets to prepare. uci_en is optional English external data.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download/re-clone sources.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.output_dir).expanduser()
    raw_dir = base_dir / "raw"
    canonical_dir = base_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    if "tensorlayer" in args.sources:
        msglog_dir = prepare_tensorlayer_msglog(raw_dir, canonical_dir, force=args.force)
        prepared.append(("tensorlayer_msglog_dir", msglog_dir))
    if "spam_lr" in args.sources:
        prepared.append(("spam_messages_lr_jsonl", prepare_spam_messages_lr(raw_dir, canonical_dir, force=args.force)))
    if "fbs" in args.sources:
        prepared.append(("fbs_sms_dataset_jsonl", prepare_fbs_sms(raw_dir, canonical_dir, force=args.force)))
    if "uci_en" in args.sources:
        prepared.append(("uci_sms_spam_collection_jsonl", prepare_uci_sms(raw_dir, canonical_dir, force=args.force)))

    print("\nPrepared sources:")
    for name, path in prepared:
        print(f"- {name}: {path}")
    print("\nNext step: build AST splits with scripts/build_ast_dataset.py.")


if __name__ == "__main__":
    main()
