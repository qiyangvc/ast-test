"""Orchestration helpers for AST experiments.

The runner defaults to dry-run planning. It only trains or evaluates when the
caller explicitly sets ``execute=True`` (or uses the CLI ``--execute`` flag).
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from src.ast_dataset import LABEL_TO_PROJECT_ID, load_ast_jsonl
from src.ast_metrics import (
    binary_metrics,
    metrics_by_attack_type,
    robustness_metrics,
    write_markdown_report,
    write_metrics_json,
)
from src.config import Config


VALID_MODES = {"baseline", "text_ast", "embedding_fgm", "text_ast_fgm"}
VALID_MODELS = {"rnn", "mlp", "cnn"}


TRAIN_LEGACY_DIR_BY_MODE = {
    "baseline": "train_clean",
    "text_ast": "train_clean_ast",
    "embedding_fgm": "train_clean",
    "text_ast_fgm": "train_clean_ast",
}


@dataclass
class ASTExperimentConfig:
    """Runtime settings for one AST experiment run."""

    dataset_dir: Path
    work_dir: Path
    mode: str = "baseline"
    models: List[str] = field(default_factory=lambda: ["rnn", "cnn"])
    execute: bool = False
    seed: int = 42
    train_word2vec: bool = True
    prepare_features: bool = True
    display_step: int = 50

    def normalized(self) -> "ASTExperimentConfig":
        mode = self.mode.strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode {self.mode!r}. Valid modes: {sorted(VALID_MODES)}")
        models = [model.strip().lower() for model in self.models]
        invalid = [model for model in models if model not in VALID_MODELS]
        if invalid:
            raise ValueError(f"Unsupported models {invalid}. Valid models: {sorted(VALID_MODELS)}")
        self.mode = mode
        self.models = models
        self.dataset_dir = Path(self.dataset_dir)
        self.work_dir = Path(self.work_dir)
        return self


class ConfigOverride:
    """Temporarily point the original training code at an experiment run dir."""

    def __init__(self, msg_log_dir: Path, output_dir: Path, weights_dir: Path, logs_dir: Path):
        self.overrides = {
            "MSG_LOG_DIR": str(msg_log_dir),
            "OUTPUT_DIR": str(output_dir),
            "WEIGHTS_DIR": str(weights_dir),
            "LOGS_DIR": str(logs_dir),
        }
        self.previous: Dict[str, str] = {}

    def __enter__(self):
        for key, value in self.overrides.items():
            self.previous[key] = getattr(Config, key)
            setattr(Config, key, value)
        Config.ensure_dirs()
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.previous.items():
            setattr(Config, key, value)
        return False


def validate_ast_dataset(dataset_dir: Path) -> List[str]:
    """Return missing required files/directories for a built AST dataset."""
    dataset_dir = Path(dataset_dir)
    required = [
        dataset_dir / "manifest.json",
        dataset_dir / "legacy" / "train_clean",
        dataset_dir / "legacy" / "train_clean_ast",
        dataset_dir / "legacy" / "test_clean",
        dataset_dir / "legacy" / "test_ast",
        dataset_dir / "canonical" / "test_clean.jsonl",
        dataset_dir / "canonical" / "test_ast.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return missing


def make_experiment_plan(config: ASTExperimentConfig) -> List[str]:
    config = config.normalized()
    train_dir_name = TRAIN_LEGACY_DIR_BY_MODE[config.mode]
    plan = [
        f"dataset_dir: {config.dataset_dir}",
        f"work_dir: {config.work_dir}",
        f"mode: {config.mode}",
        f"models: {', '.join(config.models)}",
        f"train source: legacy/{train_dir_name}",
        "test sources: canonical/test_clean.jsonl + canonical/test_ast.jsonl",
        "metrics: clean accuracy, AST accuracy, robust drop, spam recall, FPR, attack success rate, per-attack metrics",
    ]
    if config.mode in {"embedding_fgm", "text_ast_fgm"}:
        ast_cfg = Config.AST_TRAINING_CONFIG
        plan.append(
            "embedding adversarial training: "
            f"{ast_cfg['method']} epsilon={ast_cfg['epsilon']} alpha={ast_cfg['alpha']}"
        )
    else:
        plan.append("embedding adversarial training: disabled")
    if not config.execute:
        plan.append("execution: dry-run only; no training or testing will run")
    return plan


def run_ast_experiment(config: ASTExperimentConfig) -> Dict[str, object]:
    """Run or dry-run an AST experiment."""
    config = config.normalized()
    plan = make_experiment_plan(config)
    missing = validate_ast_dataset(config.dataset_dir)

    if not config.execute:
        return {"status": "dry_run", "plan": plan, "missing": missing}

    if missing:
        raise FileNotFoundError("AST dataset is incomplete:\n" + "\n".join(missing))

    np.random.seed(config.seed)
    results: Dict[str, object] = {"mode": config.mode, "models": {}}
    for model_type in config.models:
        model_result = _run_single_model(config, model_type)
        results["models"][model_type] = model_result
    return results


def _run_single_model(config: ASTExperimentConfig, model_type: str) -> Dict[str, object]:
    train_dir_name = TRAIN_LEGACY_DIR_BY_MODE[config.mode]
    run_dir = config.work_dir / config.mode / model_type
    runtime_msglog = run_dir / "data" / "msglog"
    runtime_output = run_dir / "output"
    runtime_weights = run_dir / "weights"
    runtime_logs = run_dir / "logs"

    _copy_legacy_msglog(config.dataset_dir / "legacy" / train_dir_name, runtime_msglog)

    with ConfigOverride(runtime_msglog, runtime_output, runtime_weights, runtime_logs):
        if config.train_word2vec:
            from src.word2vec import train_word2vec

            train_word2vec()
        if config.prepare_features:
            from src.text_features import prepare_mlp_features, prepare_sequence_features

            if model_type == "mlp":
                prepare_mlp_features()
            else:
                prepare_sequence_features()

        if config.mode in {"embedding_fgm", "text_ast_fgm"}:
            _train_embedding_adversarial_classifier(model_type, display_step=config.display_step)
        else:
            _train_standard_classifier(model_type)

        clean_records = load_ast_jsonl(config.dataset_dir / "canonical" / "test_clean.jsonl")
        ast_records = load_ast_jsonl(config.dataset_dir / "canonical" / "test_ast.jsonl")
        clean_metrics, clean_pred = _evaluate_records(model_type, clean_records)
        ast_metrics, ast_pred = _evaluate_records(model_type, ast_records)
        robust = robustness_metrics(clean_metrics, ast_metrics)
        by_attack = metrics_by_attack_type(ast_records, ast_pred)

        report_dir = run_dir / "reports"
        payload = {
            "clean": clean_metrics,
            "ast": ast_metrics,
            "robustness": robust,
            "by_attack": by_attack,
            "clean_prediction_count": len(clean_pred),
            "ast_prediction_count": len(ast_pred),
        }
        write_metrics_json(report_dir / "metrics.json", payload)
        write_markdown_report(
            report_dir / "summary.md",
            title=f"AST Experiment - {config.mode} - {model_type}",
            clean=clean_metrics,
            ast=ast_metrics,
            robustness=robust,
            by_attack=by_attack,
        )

    return {
        "run_dir": str(run_dir),
        "metrics_json": str(run_dir / "reports" / "metrics.json"),
        "summary_md": str(run_dir / "reports" / "summary.md"),
    }


def _copy_legacy_msglog(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing training legacy directory: {src_dir}")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst_dir)


def _train_standard_classifier(model_type: str) -> None:
    from src.classifier import train_cnn_classifier, train_mlp_classifier, train_rnn_classifier

    if model_type == "rnn":
        train_rnn_classifier()
    elif model_type == "mlp":
        train_mlp_classifier()
    elif model_type == "cnn":
        train_cnn_classifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _train_embedding_adversarial_classifier(model_type: str, display_step: int = 50) -> None:
    from src.adversarial_training import AdversarialTrainingConfig, EmbeddingAdversarialTrainer
    from src.classifier import CNNClassifier, MLPClassifier, RNNClassifier
    from src.data_loader import DataLoader

    if model_type == "mlp":
        feature_path = Path(Config.OUTPUT_DIR) / "features_mlp.npz"
        learning_rate = Config.MLP_CONFIG["learning_rate"]
        epochs = Config.MLP_CONFIG["n_epoch"]
        model = MLPClassifier(
            hidden_units=Config.MLP_CONFIG["hidden_units"],
            dropout_keep=Config.MLP_CONFIG["dropout_keep"],
        )
        model_path = Path(Config.WEIGHTS_DIR) / "mlp_classifier"
    elif model_type == "rnn":
        feature_path = Path(Config.OUTPUT_DIR) / "features_seq_20.npz"
        learning_rate = Config.RNN_CONFIG["learning_rate"]
        epochs = Config.RNN_CONFIG["n_epoch"]
        model = RNNClassifier(
            lstm_units=Config.RNN_CONFIG["lstm_units"],
            recurrent_dropout=Config.RNN_CONFIG["recurrent_dropout"],
        )
        model_path = Path(Config.WEIGHTS_DIR) / "rnn_classifier"
    elif model_type == "cnn":
        feature_path = Path(Config.OUTPUT_DIR) / "features_seq_20.npz"
        learning_rate = Config.CNN_CONFIG["learning_rate"]
        epochs = Config.CNN_CONFIG["n_epoch"]
        model = CNNClassifier(
            n_filter=Config.CNN_CONFIG["n_filter"],
            filter_size=Config.CNN_CONFIG["filter_size"],
            stride=Config.CNN_CONFIG["stride"],
            pool_size=Config.CNN_CONFIG["pool_size"],
            pool_strides=Config.CNN_CONFIG["pool_strides"],
            dropout_keep=Config.CNN_CONFIG["dropout_keep"],
        )
        model_path = Path(Config.WEIGHTS_DIR) / "cnn_classifier"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    x_train, y_train, _, _ = DataLoader(Config.OUTPUT_DIR).load_classifier_data(
        [str(feature_path)],
        test_size=Config.CLASSIFIER_CONFIG["test_size"],
    )
    val_size = max(1, int(len(x_train) * 0.15))
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train_sub, y_train_sub = x_train[val_size:], y_train[val_size:]

    ast_cfg = AdversarialTrainingConfig(**Config.AST_TRAINING_CONFIG)
    trainer = EmbeddingAdversarialTrainer(model, ast_cfg)
    trainer.fit(
        x_train_sub,
        y_train_sub,
        x_val,
        y_val,
        epochs=epochs,
        batch_size=Config.CLASSIFIER_CONFIG["batch_size"],
        learning_rate=learning_rate,
        display_step=display_step,
    )
    model.save(str(model_path))


def _evaluate_records(model_type: str, records: Iterable[object]):
    from src.serving import TextClassifierService

    service = TextClassifierService(model_type=model_type)
    y_true: List[int] = []
    y_pred: List[int] = []
    for record in records:
        result = service.predict(record.segmented)
        y_true.append(record.label_id)
        y_pred.append(LABEL_TO_PROJECT_ID[result["label"]])
    return binary_metrics(y_true, y_pred), y_pred


def write_dry_run_plan(path: Path, result: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
