"""Evaluation metrics for clean and adversarial spam-text experiments."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from src.ast_dataset import PROJECT_ID_TO_LABEL, TextRecord


@dataclass
class BinaryClassificationMetrics:
    """Binary metrics with spam treated as the risk-positive class."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    spam_precision: float
    spam_recall: float
    spam_f1: float
    normal_precision: float
    normal_recall: float
    normal_f1: float
    false_positive_rate: float
    support_spam: int
    support_normal: int
    confusion_matrix: List[List[int]]


@dataclass
class RobustnessMetrics:
    """Difference between clean and adversarial evaluation results."""

    clean_accuracy: float
    ast_accuracy: float
    robust_drop: float
    clean_spam_recall: float
    ast_spam_recall: float
    spam_recall_drop: float
    attack_success_rate: float


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> BinaryClassificationMetrics:
    """Compute metrics for project labels: spam=0, normal=1."""
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    labels = [0, 1]
    matrix = np.zeros((2, 2), dtype=np.int64)
    for true_label, pred_label in zip(y_true_arr, y_pred_arr):
        if int(true_label) not in labels or int(pred_label) not in labels:
            raise ValueError("Only binary project labels 0=spam and 1=normal are supported")
        matrix[int(true_label), int(pred_label)] += 1

    # Matrix layout:
    # [[spam->spam, spam->normal],
    #  [normal->spam, normal->normal]]
    spam_tp = int(matrix[0, 0])
    spam_fn = int(matrix[0, 1])
    spam_fp = int(matrix[1, 0])
    normal_tp = int(matrix[1, 1])
    normal_fn = int(matrix[1, 0])
    normal_fp = int(matrix[0, 1])

    spam_precision = _safe_div(spam_tp, spam_tp + spam_fp)
    spam_recall = _safe_div(spam_tp, spam_tp + spam_fn)
    spam_f1 = _safe_div(2 * spam_precision * spam_recall, spam_precision + spam_recall)

    normal_precision = _safe_div(normal_tp, normal_tp + normal_fp)
    normal_recall = _safe_div(normal_tp, normal_tp + normal_fn)
    normal_f1 = _safe_div(2 * normal_precision * normal_recall, normal_precision + normal_recall)

    support_spam = spam_tp + spam_fn
    support_normal = normal_tp + normal_fn
    total = support_spam + support_normal

    accuracy = _safe_div(spam_tp + normal_tp, total)
    macro_f1 = (spam_f1 + normal_f1) / 2
    weighted_f1 = _safe_div(spam_f1 * support_spam + normal_f1 * support_normal, total)
    false_positive_rate = _safe_div(spam_fp, spam_fp + normal_tp)

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        spam_precision=spam_precision,
        spam_recall=spam_recall,
        spam_f1=spam_f1,
        normal_precision=normal_precision,
        normal_recall=normal_recall,
        normal_f1=normal_f1,
        false_positive_rate=false_positive_rate,
        support_spam=support_spam,
        support_normal=support_normal,
        confusion_matrix=matrix.astype(int).tolist(),
    )


def robustness_metrics(
    clean: BinaryClassificationMetrics,
    ast: BinaryClassificationMetrics,
) -> RobustnessMetrics:
    """Compare clean and AST performance."""
    return RobustnessMetrics(
        clean_accuracy=clean.accuracy,
        ast_accuracy=ast.accuracy,
        robust_drop=clean.accuracy - ast.accuracy,
        clean_spam_recall=clean.spam_recall,
        ast_spam_recall=ast.spam_recall,
        spam_recall_drop=clean.spam_recall - ast.spam_recall,
        attack_success_rate=1.0 - ast.spam_recall,
    )


def metrics_by_attack_type(
    records: Sequence[TextRecord],
    y_pred: Sequence[int],
) -> Dict[str, BinaryClassificationMetrics]:
    """Compute per-attack metrics from AST records and predictions."""
    if len(records) != len(y_pred):
        raise ValueError("records and y_pred must have the same length")
    grouped_true: Dict[str, List[int]] = defaultdict(list)
    grouped_pred: Dict[str, List[int]] = defaultdict(list)
    for record, pred in zip(records, y_pred):
        group = record.attack_type or "clean"
        grouped_true[group].append(record.label_id)
        grouped_pred[group].append(int(pred))
    return {group: binary_metrics(grouped_true[group], grouped_pred[group]) for group in grouped_true}


def prediction_distribution(y_pred: Iterable[int]) -> Dict[str, int]:
    counts = Counter(PROJECT_ID_TO_LABEL.get(int(label), str(label)) for label in y_pred)
    return dict(counts)


def write_metrics_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in payload.items():
        if hasattr(value, "__dataclass_fields__"):
            serializable[key] = asdict(value)
        elif isinstance(value, dict):
            serializable[key] = {
                sub_key: asdict(sub_value) if hasattr(sub_value, "__dataclass_fields__") else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            serializable[key] = value
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(
    path: Path,
    title: str,
    clean: Optional[BinaryClassificationMetrics] = None,
    ast: Optional[BinaryClassificationMetrics] = None,
    robustness: Optional[RobustnessMetrics] = None,
    by_attack: Optional[Mapping[str, BinaryClassificationMetrics]] = None,
) -> None:
    """Write a compact experiment report for later presentation use."""
    lines = [f"# {title}", ""]
    if clean:
        lines.extend(_metrics_section("Clean Test", clean))
    if ast:
        lines.extend(_metrics_section("AST Test", ast))
    if robustness:
        lines.extend(
            [
                "## Robustness",
                "",
                f"- Robust accuracy drop: {robustness.robust_drop:.4f}",
                f"- Spam recall drop: {robustness.spam_recall_drop:.4f}",
                f"- Attack success rate: {robustness.attack_success_rate:.4f}",
                "",
            ]
        )
    if by_attack:
        lines.extend(["## By Attack Type", "", "| Attack | Accuracy | Spam Recall | Macro F1 | FPR |", "|---|---:|---:|---:|---:|"])
        for attack_type, metrics in sorted(by_attack.items()):
            lines.append(
                f"| {attack_type} | {metrics.accuracy:.4f} | {metrics.spam_recall:.4f} | "
                f"{metrics.macro_f1:.4f} | {metrics.false_positive_rate:.4f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _metrics_section(name: str, metrics: BinaryClassificationMetrics) -> List[str]:
    return [
        f"## {name}",
        "",
        f"- Accuracy: {metrics.accuracy:.4f}",
        f"- Macro F1: {metrics.macro_f1:.4f}",
        f"- Weighted F1: {metrics.weighted_f1:.4f}",
        f"- Spam precision / recall / F1: {metrics.spam_precision:.4f} / {metrics.spam_recall:.4f} / {metrics.spam_f1:.4f}",
        f"- Normal precision / recall / F1: {metrics.normal_precision:.4f} / {metrics.normal_recall:.4f} / {metrics.normal_f1:.4f}",
        f"- False positive rate: {metrics.false_positive_rate:.4f}",
        f"- Confusion matrix: {metrics.confusion_matrix}",
        "",
    ]
