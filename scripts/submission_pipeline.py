#!/usr/bin/env python
"""End-to-end submission pipeline for AST spam-text experiments.

This script runs the complete Python 3.12 submission workflow. It trains real
Word2Vec embeddings with Gensim and real MLP/CNN/RNN classifiers with PyTorch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.adversarial_text import ChineseSpamTextAttacker, segment_for_project
from src.ast_dataset import LABEL_TO_PROJECT_ID, TextRecord, load_ast_jsonl
from src.ast_metrics import binary_metrics, robustness_metrics, write_metrics_json


MODE_TRAIN_SPLIT = {
    "baseline": "train_clean",
    "focal": "train_clean",
    "text_ast": "train_clean_ast",
    "text_ast_focal": "train_clean_ast",
    "embedding_fgm": "train_clean",
    "embedding_fgm_focal": "train_clean",
    "text_ast_fgm": "train_clean_ast",
    "text_ast_fgm_focal": "train_clean_ast",
}

MODE_USES_FGM = {
    "baseline": False,
    "focal": False,
    "text_ast": False,
    "text_ast_focal": False,
    "embedding_fgm": True,
    "embedding_fgm_focal": True,
    "text_ast_fgm": True,
    "text_ast_fgm_focal": True,
}

MODE_USES_FOCAL = {
    "baseline": False,
    "focal": True,
    "text_ast": False,
    "text_ast_focal": True,
    "embedding_fgm": False,
    "embedding_fgm_focal": True,
    "text_ast_fgm": False,
    "text_ast_fgm_focal": True,
}

DEFAULT_MODES = ["baseline", "text_ast", "embedding_fgm", "text_ast_fgm"]
FULL_MODES = [
    "baseline",
    "focal",
    "text_ast",
    "text_ast_focal",
    "embedding_fgm",
    "embedding_fgm_focal",
    "text_ast_fgm",
    "text_ast_fgm_focal",
]
DEFAULT_MODELS = ["mlp", "cnn", "rnn"]
FULL_MODELS = ["mlp", "cnn", "rnn", "bilstm_attn"]
DEFAULT_ENSEMBLE_MODELS = ["mlp", "cnn", "rnn", "bilstm_attn"]


@dataclass
class RunConfig:
    dataset_dir: Path
    output_dir: Path
    modes: List[str]
    models: List[str]
    vector_size: int
    max_vocab: int
    max_len: int
    w2v_epochs: int
    clf_epochs: int
    batch_size: int
    seed: int
    fgm_epsilon: float
    focal_gamma: float
    learning_rate: float
    confidence_attack_limit: int
    confidence_attack_strength: str
    review_sample_size: int
    run_ensemble: bool
    ensemble_models: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokens_from_segmented(segmented: str) -> List[str]:
    tokens = [tok for tok in segmented.strip().split() if tok]
    if not tokens:
        tokens = [tok for tok in segment_for_project(segmented).split() if tok]
    return tokens


def load_records(dataset_dir: Path, split_name: str) -> List[TextRecord]:
    return load_ast_jsonl(dataset_dir / "canonical" / f"{split_name}.jsonl")


def train_word2vec(records: Sequence[TextRecord], output_dir: Path, cfg: RunConfig, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"word2vec_{name}.model"
    if model_path.exists():
        return model_path

    sentences = [tokens_from_segmented(record.segmented) for record in records]
    model = Word2Vec(
        sentences=sentences,
        vector_size=cfg.vector_size,
        window=5,
        min_count=2,
        sg=1,
        negative=8,
        workers=max(1, min(4, os.cpu_count() or 1)),
        epochs=cfg.w2v_epochs,
        seed=cfg.seed,
    )
    model.save(str(model_path))
    return model_path


def build_vocab_and_matrix(w2v_path: Path, records: Sequence[TextRecord], cfg: RunConfig):
    w2v = Word2Vec.load(str(w2v_path))
    counter = Counter()
    for record in records:
        counter.update(tokens_from_segmented(record.segmented))

    words = ["<PAD>", "<UNK>"]
    for word, _ in counter.most_common(cfg.max_vocab - 2):
        if word in w2v.wv:
            words.append(word)

    word2idx = {word: idx for idx, word in enumerate(words)}
    matrix = np.zeros((len(words), cfg.vector_size), dtype=np.float32)
    rng = np.random.default_rng(cfg.seed)
    matrix[1] = rng.normal(0, 0.05, cfg.vector_size)
    for word, idx in word2idx.items():
        if word in {"<PAD>", "<UNK>"}:
            continue
        matrix[idx] = w2v.wv[word]
    return word2idx, matrix


def vectorize_records(records: Sequence[TextRecord], word2idx: Dict[str, int], max_len: int):
    x = np.zeros((len(records), max_len), dtype=np.int64)
    y = np.zeros((len(records),), dtype=np.int64)
    for row, record in enumerate(records):
        tokens = tokens_from_segmented(record.segmented)
        for col, token in enumerate(tokens[:max_len]):
            x[row, col] = word2idx.get(token, 1)
        y[row] = record.label_id
    return x, y


class MLPClassifierTorch(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_matrix.shape[1], hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def pooled(self, ids: torch.Tensor, embeds: Optional[torch.Tensor] = None):
        if embeds is None:
            embeds = self.embedding(ids)
        mask = (ids != 0).float().unsqueeze(-1)
        summed = (embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, ids: torch.Tensor):
        return self.forward_embeds(ids, self.embedding(ids))

    def forward_embeds(self, ids: torch.Tensor, embeds: torch.Tensor):
        pooled = self.pooled(ids, embeds)
        hidden = torch.relu(self.fc1(self.dropout(pooled)))
        return self.fc2(self.dropout(hidden))


class CNNClassifierTorch(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, n_filters: int = 64, dropout: float = 0.35):
        super().__init__()
        emb_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, kernel_size=k, padding=0) for k in (2, 3, 4, 5)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * 4, 2)

    def forward(self, ids: torch.Tensor):
        return self.forward_embeds(ids, self.embedding(ids))

    def forward_embeds(self, ids: torch.Tensor, embeds: torch.Tensor):
        x = embeds.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            z = torch.relu(conv(x))
            pooled.append(torch.max(z, dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class RNNClassifierTorch(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        emb_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, ids: torch.Tensor):
        return self.forward_embeds(ids, self.embedding(ids))

    def forward_embeds(self, ids: torch.Tensor, embeds: torch.Tensor):
        out, _ = self.lstm(embeds)
        mask = (ids != 0).float().unsqueeze(-1)
        summed = (out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return self.fc(self.dropout(pooled))


class BiLSTMAttentionClassifierTorch(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        emb_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.attention = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, ids: torch.Tensor):
        return self.forward_embeds(ids, self.embedding(ids))

    def forward_embeds(self, ids: torch.Tensor, embeds: torch.Tensor):
        out, _ = self.lstm(embeds)
        mask = ids != 0
        scores = self.attention(torch.tanh(out)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (out * weights).sum(dim=1)
        return self.fc(self.dropout(pooled))


def make_model(model_name: str, embedding_matrix: np.ndarray) -> nn.Module:
    if model_name == "mlp":
        return MLPClassifierTorch(embedding_matrix)
    if model_name == "cnn":
        return CNNClassifierTorch(embedding_matrix)
    if model_name == "rnn":
        return RNNClassifierTorch(embedding_matrix)
    if model_name in {"bilstm_attn", "bilstm_attention", "attention"}:
        return BiLSTMAttentionClassifierTorch(embedding_matrix)
    raise ValueError(f"Unsupported model: {model_name}")


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = Counter(int(v) for v in y)
    total = len(y)
    weights = [total / (2 * max(counts.get(label, 1), 1)) for label in (0, 1)]
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - target_probs).clamp(min=1e-8) ** self.gamma) * target_log_probs
        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, targets)
        return loss.mean()


def make_criterion(train_y: np.ndarray, cfg: RunConfig, uses_focal: bool, device: torch.device) -> nn.Module:
    weights = class_weights(train_y).to(device)
    if uses_focal:
        return FocalLoss(alpha=weights, gamma=cfg.focal_gamma).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def train_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: RunConfig,
    uses_fgm: bool,
    uses_focal: bool,
    model_path: Path,
    history_path: Path,
) -> Dict[str, List[float]]:
    if model_path.exists() and history_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return json.loads(history_path.read_text(encoding="utf-8"))

    device = torch.device("cpu")
    model.to(device)
    criterion = make_criterion(train_y, cfg, uses_focal, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    val_ds = TensorDataset(torch.tensor(val_x), torch.tensor(val_y))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = None
    best_val = -1.0

    for epoch in range(cfg.clf_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.clf_epochs}", leave=False)
        for ids, labels in progress:
            ids = ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(ids)
            loss = criterion(logits, labels)
            if uses_fgm:
                embeds = model.embedding(ids).detach().requires_grad_(True)
                adv_logits = model.forward_embeds(ids, embeds)
                adv_loss = criterion(adv_logits, labels)
                grad = torch.autograd.grad(adv_loss, embeds, retain_graph=False, create_graph=False)[0]
                norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).reshape(-1, 1, 1).clamp(min=1e-8)
                adv_embeds = embeds + cfg.fgm_epsilon * grad / norm
                loss = 0.5 * loss + 0.5 * criterion(model.forward_embeds(ids, adv_embeds.detach()), labels)

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
            progress.set_postfix(loss=total_loss / max(seen, 1), acc=correct / max(seen, 1))

        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        train_loss = total_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return history


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for ids, labels in loader:
            ids = ids.to(device)
            labels = labels.to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def predict_model(model: nn.Module, x: np.ndarray, batch_size: int):
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    preds = []
    probs = []
    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=batch_size)
    with torch.no_grad():
        for (ids,) in loader:
            logits = model(ids.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))
    return np.concatenate(preds), np.vstack(probs)


def predict_ensemble(models: Sequence[nn.Module], x: np.ndarray, batch_size: int):
    if not models:
        raise ValueError("Ensemble prediction requires at least one model.")
    prob_items = []
    for model in models:
        _, prob = predict_model(model, x, batch_size)
        prob_items.append(prob)
    avg_prob = np.mean(np.stack(prob_items, axis=0), axis=0)
    return avg_prob.argmax(axis=1), avg_prob


def evaluate_records(
    model: nn.Module,
    records: Sequence[TextRecord],
    word2idx: Dict[str, int],
    cfg: RunConfig,
):
    x, y = vectorize_records(records, word2idx, cfg.max_len)
    pred, prob = predict_model(model, x, cfg.batch_size)
    return {
        "metrics": asdict(binary_metrics(y, pred)),
        "classification_report": classification_report(y, pred, digits=4, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).astype(int).tolist(),
    }, pred, prob


def evaluate_ensemble_records(
    models: Sequence[nn.Module],
    records: Sequence[TextRecord],
    word2idx: Dict[str, int],
    cfg: RunConfig,
):
    x, y = vectorize_records(records, word2idx, cfg.max_len)
    pred, prob = predict_ensemble(models, x, cfg.batch_size)
    return {
        "metrics": asdict(binary_metrics(y, pred)),
        "classification_report": classification_report(y, pred, digits=4, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).astype(int).tolist(),
    }, pred, prob


def save_vocab(path: Path, word2idx: Dict[str, int], embedding_matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"word2idx": word2idx, "embedding_shape": list(embedding_matrix.shape)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all_splits(dataset_dir: Path):
    return {
        "train_clean": load_records(dataset_dir, "train_clean"),
        "train_clean_ast": load_records(dataset_dir, "train_clean_ast"),
        "val_clean": load_records(dataset_dir, "val_clean"),
        "test_clean": load_records(dataset_dir, "test_clean"),
        "test_ast": load_records(dataset_dir, "test_ast"),
    }


def train_and_evaluate(cfg: RunConfig) -> Dict[str, object]:
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    validate_modes_and_models(cfg.modes, cfg.models)
    records = load_all_splits(cfg.dataset_dir)
    external_uci_path = PROJECT_ROOT / "data/external/canonical/uci_sms_spam_collection.jsonl"
    external_uci = load_ast_jsonl(external_uci_path) if external_uci_path.exists() else []

    results = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "runs": {},
    }

    w2v_paths: Dict[str, Path] = {}
    for w2v_name, train_split in {"clean": "train_clean", "clean_ast": "train_clean_ast"}.items():
        w2v_paths[w2v_name] = train_word2vec(records[train_split], cfg.output_dir / "word2vec", cfg, w2v_name)

    for mode in cfg.modes:
        train_split = MODE_TRAIN_SPLIT[mode]
        w2v_name = "clean_ast" if "text_ast" in mode else "clean"
        word2idx, embedding_matrix = build_vocab_and_matrix(w2v_paths[w2v_name], records[train_split], cfg)
        save_vocab(cfg.output_dir / "word2vec" / f"vocab_{mode}.json", word2idx, embedding_matrix)

        train_x, train_y = vectorize_records(records[train_split], word2idx, cfg.max_len)
        val_x, val_y = vectorize_records(records["val_clean"], word2idx, cfg.max_len)
        mode_result = {}
        trained_models: Dict[str, nn.Module] = {}

        for model_name in cfg.models:
            print(f"\n=== Training {mode}/{model_name} ===")
            model = make_model(model_name, embedding_matrix)
            model_dir = cfg.output_dir / "models" / mode
            history = train_classifier(
                model,
                train_x,
                train_y,
                val_x,
                val_y,
                cfg,
                uses_fgm=MODE_USES_FGM[mode],
                uses_focal=MODE_USES_FOCAL[mode],
                model_path=model_dir / f"{model_name}.pt",
                history_path=model_dir / f"{model_name}_history.json",
            )
            trained_models[model_name] = model

            clean_eval, _, _ = evaluate_records(model, records["test_clean"], word2idx, cfg)
            ast_eval, ast_pred, _ = evaluate_records(model, records["test_ast"], word2idx, cfg)
            clean_metrics_dc = metrics_dict_to_dataclass(clean_eval["metrics"])
            ast_metrics_dc = metrics_dict_to_dataclass(ast_eval["metrics"])
            robust = asdict(robustness_metrics(clean_metrics_dc, ast_metrics_dc))

            by_attack = evaluate_by_attack(records["test_ast"], ast_pred)
            uci_eval = None
            if external_uci:
                uci_eval, _, _ = evaluate_records(model, external_uci, word2idx, cfg)

            mode_result[model_name] = {
                "history": history,
                "clean": clean_eval,
                "ast": ast_eval,
                "robustness": robust,
                "by_attack": by_attack,
                "uci_en": uci_eval,
            }

            write_metrics_json(
                cfg.output_dir / "metrics" / mode / f"{model_name}.json",
                {
                    "clean": clean_metrics_dc,
                    "ast": ast_metrics_dc,
                    "robustness": robustness_metrics(clean_metrics_dc, ast_metrics_dc),
                    "by_attack": {k: metrics_dict_to_dataclass(v) for k, v in by_attack.items()},
                    "uci_en": uci_eval,
                },
            )

        if cfg.run_ensemble:
            ensemble_members = [name for name in cfg.ensemble_models if name in trained_models]
            if len(ensemble_members) >= 2:
                print(f"\n=== Evaluating {mode}/ensemble_vote ({', '.join(ensemble_members)}) ===")
                ensemble_models = [trained_models[name] for name in ensemble_members]
                clean_eval, _, _ = evaluate_ensemble_records(ensemble_models, records["test_clean"], word2idx, cfg)
                ast_eval, ast_pred, _ = evaluate_ensemble_records(ensemble_models, records["test_ast"], word2idx, cfg)
                clean_metrics_dc = metrics_dict_to_dataclass(clean_eval["metrics"])
                ast_metrics_dc = metrics_dict_to_dataclass(ast_eval["metrics"])
                robust = asdict(robustness_metrics(clean_metrics_dc, ast_metrics_dc))
                by_attack = evaluate_by_attack(records["test_ast"], ast_pred)
                uci_eval = None
                if external_uci:
                    uci_eval, _, _ = evaluate_ensemble_records(ensemble_models, external_uci, word2idx, cfg)

                mode_result["ensemble_vote"] = {
                    "ensemble_members": ensemble_members,
                    "clean": clean_eval,
                    "ast": ast_eval,
                    "robustness": robust,
                    "by_attack": by_attack,
                    "uci_en": uci_eval,
                }

                write_metrics_json(
                    cfg.output_dir / "metrics" / mode / "ensemble_vote.json",
                    {
                        "ensemble_members": ensemble_members,
                        "clean": clean_metrics_dc,
                        "ast": ast_metrics_dc,
                        "robustness": robustness_metrics(clean_metrics_dc, ast_metrics_dc),
                        "by_attack": {k: metrics_dict_to_dataclass(v) for k, v in by_attack.items()},
                        "uci_en": uci_eval,
                    },
                )

        results["runs"][mode] = mode_result

    (cfg.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "metrics" / "all_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def metrics_dict_to_dataclass(metrics: Dict[str, object]):
    from src.ast_metrics import BinaryClassificationMetrics

    return BinaryClassificationMetrics(**metrics)


def validate_modes_and_models(modes: Sequence[str], models: Sequence[str]) -> None:
    unknown_modes = sorted(set(modes).difference(MODE_TRAIN_SPLIT))
    if unknown_modes:
        raise ValueError(f"Unsupported modes: {unknown_modes}. Supported: {sorted(MODE_TRAIN_SPLIT)}")
    supported_models = {"mlp", "cnn", "rnn", "bilstm_attn", "bilstm_attention", "attention"}
    unknown_models = sorted(set(models).difference(supported_models))
    if unknown_models:
        raise ValueError(f"Unsupported models: {unknown_models}. Supported: {sorted(supported_models)}")


def evaluate_by_attack(records: Sequence[TextRecord], pred: Sequence[int]) -> Dict[str, Dict[str, object]]:
    grouped_y = defaultdict(list)
    grouped_pred = defaultdict(list)
    for record, p in zip(records, pred):
        key = record.attack_type or "clean"
        grouped_y[key].append(record.label_id)
        grouped_pred[key].append(int(p))
    return {key: asdict(binary_metrics(grouped_y[key], grouped_pred[key])) for key in sorted(grouped_y)}


def load_trained_model_for_attack(cfg: RunConfig, mode: str, model_name: str):
    train_split = MODE_TRAIN_SPLIT[mode]
    records = load_all_splits(cfg.dataset_dir)
    w2v_name = "clean_ast" if "text_ast" in mode else "clean"
    w2v_path = cfg.output_dir / "word2vec" / f"word2vec_{w2v_name}.model"
    word2idx, embedding_matrix = build_vocab_and_matrix(w2v_path, records[train_split], cfg)
    model = make_model(model_name, embedding_matrix)
    model.load_state_dict(torch.load(cfg.output_dir / "models" / mode / f"{model_name}.pt", map_location="cpu"))
    return model, word2idx, records


def confidence_search_attack(cfg: RunConfig, mode: str = "text_ast_fgm", model_name: str = "cnn") -> Dict[str, object]:
    model, word2idx, records = load_trained_model_for_attack(cfg, mode, model_name)
    attacker = ChineseSpamTextAttacker(seed=cfg.seed)
    spam_records = [r for r in records["test_clean"] if r.label == "spam"]
    if cfg.confidence_attack_limit > 0:
        spam_records = spam_records[: cfg.confidence_attack_limit]

    output_path = cfg.output_dir / "attacks" / f"confidence_search_{mode}_{model_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    searched = 0
    rows = []
    strength = cfg.confidence_attack_strength
    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(spam_records, desc="confidence-search"):
            candidates = []
            for seed_offset in range(4):
                attacker.rng.seed(cfg.seed + seed_offset)
                candidates.extend(
                    attacker.generate(
                        record.segmented,
                        record.label,
                        max_variants=8,
                        strength=strength,
                    )
                )
            unique = {item.adversarial: item for item in candidates}
            if not unique:
                continue
            candidate_texts = [record.segmented] + [segment_for_project(item.adversarial) for item in unique.values()]
            temp_records = [
                TextRecord(
                    id=f"candidate_{i}",
                    source=record.source,
                    label=record.label,
                    label_id=record.label_id,
                    text=text,
                    segmented=text,
                )
                for i, text in enumerate(candidate_texts)
            ]
            x, _ = vectorize_records(temp_records, word2idx, cfg.max_len)
            pred, prob = predict_model(model, x, cfg.batch_size)
            normal_probs = prob[:, 1]
            best_idx = int(normal_probs.argmax())
            best_item = None if best_idx == 0 else list(unique.values())[best_idx - 1]
            success = int(pred[best_idx] == LABEL_TO_PROJECT_ID["normal"])
            successes += success
            searched += 1
            row = {
                "id": record.id,
                "source": record.source,
                "original": record.text,
                "best_text": record.text if best_item is None else best_item.adversarial,
                "best_segmented": candidate_texts[best_idx],
                "attack_type": None if best_item is None else best_item.attack_type,
                "operations": [] if best_item is None else best_item.operations,
                "original_normal_prob": float(normal_probs[0]),
                "best_normal_prob": float(normal_probs[best_idx]),
                "pred_label": int(pred[best_idx]),
                "success": bool(success),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": mode,
        "model": model_name,
        "strength": strength,
        "searched": searched,
        "successes": successes,
        "attack_success_rate": successes / max(searched, 1),
        "output": str(output_path),
    }
    summary_path = cfg.output_dir / "attacks" / f"confidence_search_{mode}_{model_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def ast_quality_review(cfg: RunConfig) -> Dict[str, object]:
    records = load_records(cfg.dataset_dir, "test_ast")
    rng = random.Random(cfg.seed)
    sample = records[:]
    rng.shuffle(sample)
    if cfg.review_sample_size > 0:
        sample = sample[: cfg.review_sample_size]
    output_path = cfg.output_dir / "review" / "ast_quality_review.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_counter = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for record in sample:
            parent = str(record.metadata.get("parent_text", ""))
            text = record.text
            label_preserved = bool(text.strip()) and record.label in {"spam", "normal"}
            changed = parent != text
            risky = record.label == "normal" and record.attack_type in {"mixed", "contact_obfuscation"}
            readable = len(text.strip()) >= 2
            verdict = "pass" if label_preserved and changed and readable and not risky else "needs_review"
            summary_counter[verdict] += 1
            row = {
                "id": record.id,
                "label": record.label,
                "attack_type": record.attack_type,
                "parent_text": parent,
                "adversarial_text": text,
                "operations": record.operations,
                "label_preserved_check": label_preserved,
                "changed_check": changed,
                "readability_check": readable,
                "risk_note": "normal sample has aggressive attack" if risky else "",
                "review_verdict": verdict,
                "check_method": "rule_based_quality_check",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "sample_size": len(sample),
        "verdict_counts": dict(summary_counter),
        "pass_rate": summary_counter["pass"] / max(len(sample), 1),
        "output": str(output_path),
        "note": "This is a rule-based programmatic quality check; final human review should be done if required.",
    }
    summary_path = cfg.output_dir / "review" / "ast_quality_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def attempt_huggingface_gated(cfg: RunConfig) -> Dict[str, object]:
    datasets = [
        "paulkm/chinese_conversation_and_spam",
        "reatiny/chinese-spam-10000",
    ]
    results = {}
    api = HfApi()
    out_dir = cfg.output_dir / "external_access"
    out_dir.mkdir(parents=True, exist_ok=True)
    for repo_id in datasets:
        try:
            info = api.dataset_info(repo_id)
            siblings = [s.rfilename for s in info.siblings]
            target = next((name for name in siblings if name.endswith((".csv", ".json", ".jsonl", ".parquet"))), None)
            if target is None:
                results[repo_id] = {"status": "metadata_ok_no_supported_file", "siblings": siblings[:20]}
                continue
            path = hf_hub_download(repo_id=repo_id, filename=target, repo_type="dataset", local_dir=out_dir / repo_id.replace("/", "__"))
            results[repo_id] = {"status": "downloaded", "file": path}
        except Exception as exc:
            results[repo_id] = {
                "status": "blocked_or_unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
    path = out_dir / "huggingface_gated_attempts.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def write_submission_report(cfg: RunConfig, results: Dict[str, object], attack_summary, review_summary, hf_results) -> Path:
    report_path = cfg.output_dir / "SUBMISSION_REPORT.md"
    lines = [
        "# 文本检测实践 AST 作业报告",
        "",
        "## 实验数据",
        "",
        f"- AST 数据目录：`{cfg.dataset_dir}`",
        "- 训练/验证/测试划分已先于 AST 生成完成，避免同源变体泄漏。",
        "- 已包含 TensorLayer text-antispam、SpamMessagesLR、FBS_SMS_Dataset 和 UCI 英文外部测试集。",
        "",
        "## 训练配置",
        "",
        f"- Word2Vec: vector_size={cfg.vector_size}, epochs={cfg.w2v_epochs}, skip-gram, negative sampling",
        f"- 分类器：{', '.join(cfg.models)}",
        f"- 实验模式：{', '.join(cfg.modes)}",
        f"- 分类器训练 epoch：{cfg.clf_epochs}",
        f"- max_len={cfg.max_len}, max_vocab={cfg.max_vocab}, batch_size={cfg.batch_size}",
        f"- Focal Loss gamma：{cfg.focal_gamma}",
        f"- 模型集成：{'启用' if cfg.run_ensemble else '关闭'}"
        + (f"，soft voting 成员={', '.join(cfg.ensemble_models)}" if cfg.run_ensemble else ""),
        "",
        "## 结果摘要",
        "",
        "| Mode | Model | Clean Acc | AST Acc | Robust Drop | AST Spam Recall | UCI Acc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode, mode_result in results["runs"].items():
        for model_name, payload in mode_result.items():
            clean = payload["clean"]["metrics"]
            ast = payload["ast"]["metrics"]
            robust = payload["robustness"]
            uci = payload.get("uci_en")
            uci_acc = "" if not uci else f"{uci['metrics']['accuracy']:.4f}"
            lines.append(
                f"| {mode} | {model_name} | {clean['accuracy']:.4f} | {ast['accuracy']:.4f} | "
                f"{robust['robust_drop']:.4f} | {ast['spam_recall']:.4f} | {uci_acc} |"
            )
    lines.extend(
        [
            "",
            "## 基于模型置信度搜索的攻击",
            "",
            f"- 搜索模型：`{attack_summary.get('mode')}/{attack_summary.get('model')}`",
            f"- 攻击强度：`{attack_summary.get('strength')}`",
            f"- 搜索样本数：{attack_summary.get('searched')}",
            f"- 攻击成功率：{attack_summary.get('attack_success_rate'):.4f}",
            f"- 输出：`{attack_summary.get('output')}`",
            "",
            "## AST 样本质量检查",
            "",
            f"- 抽检样本数：{review_summary.get('sample_size')}",
            f"- 通过率：{review_summary.get('pass_rate'):.4f}",
            f"- 输出：`{review_summary.get('output')}`",
            "- 注：该文件为规则程序质检结果；如果课程要求严格的人类签名审核，需要学生最后确认。",
            "",
            "## Hugging Face gated 数据集尝试",
            "",
        ]
    )
    for repo_id, result in hf_results.items():
        lines.append(f"- `{repo_id}`: {result.get('status')} ({result.get('error_type', '')})")
    lines.extend(
        [
            "",
            "## 产物位置",
            "",
            f"- Word2Vec：`{cfg.output_dir / 'word2vec'}`",
            f"- 模型权重：`{cfg.output_dir / 'models'}`",
            f"- 指标：`{cfg.output_dir / 'metrics'}`",
            f"- 置信度攻击：`{cfg.output_dir / 'attacks'}`",
            f"- 样本质检：`{cfg.output_dir / 'review'}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run full submission training/evaluation pipeline.")
    parser.add_argument("--dataset-dir", default="data/ast_experiment")
    parser.add_argument("--output-dir", default="output/submission")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Train the expanded matrix with Focal Loss modes, BiLSTM-Attention, and ensemble_vote.",
    )
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--w2v-epochs", type=int, default=5)
    parser.add_argument("--clf-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fgm-epsilon", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble_vote evaluation.")
    parser.add_argument("--ensemble-models", nargs="+", default=DEFAULT_ENSEMBLE_MODELS)
    parser.add_argument("--confidence-attack-limit", type=int, default=1000, help="0 means search all clean spam test samples.")
    parser.add_argument(
        "--confidence-attack-strength",
        choices=["mild", "balanced", "strong"],
        default="mild",
        help="Perturbation profile used by the post-training confidence search attack.",
    )
    parser.add_argument("--review-sample-size", type=int, default=200, help="0 means review all AST test samples.")
    args = parser.parse_args()
    modes = FULL_MODES if args.full_matrix and args.modes == DEFAULT_MODES else args.modes
    models = FULL_MODELS if args.full_matrix and args.models == DEFAULT_MODELS else args.models
    return RunConfig(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        modes=modes,
        models=models,
        vector_size=args.vector_size,
        max_vocab=args.max_vocab,
        max_len=args.max_len,
        w2v_epochs=args.w2v_epochs,
        clf_epochs=args.clf_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        fgm_epsilon=args.fgm_epsilon,
        focal_gamma=args.focal_gamma,
        learning_rate=args.learning_rate,
        confidence_attack_limit=args.confidence_attack_limit,
        confidence_attack_strength=args.confidence_attack_strength,
        review_sample_size=args.review_sample_size,
        run_ensemble=not args.no_ensemble,
        ensemble_models=args.ensemble_models,
    )


def main() -> None:
    start = time.time()
    cfg = parse_args()
    results = train_and_evaluate(cfg)
    attack_summary = confidence_search_attack(cfg)
    review_summary = ast_quality_review(cfg)
    hf_results = attempt_huggingface_gated(cfg)
    report_path = write_submission_report(cfg, results, attack_summary, review_summary, hf_results)
    print(f"\nSubmission pipeline completed in {(time.time() - start) / 60:.2f} min")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
