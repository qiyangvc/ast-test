"""Inference helpers for the AST submission models.

This module serves the PyTorch models produced by ``scripts/submission_pipeline.py``.
Final submission artifacts are PyTorch ``.pt`` checkpoints plus JSON vocab files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.adversarial_text import ChineseSpamTextAttacker, segment_for_project


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/submission_full_20260706_full"
LABEL_ID_TO_NAME = {0: "spam", 1: "normal"}
LABEL_NAME_TO_ID = {"spam": 0, "normal": 1}
DEFAULT_MAX_LEN = 64
DEFAULT_BATCH_SIZE = 128
TRAINING_AST_STRENGTH = "mild"
TRAINING_MAX_VARIANTS_BY_LABEL = {"spam": 2, "normal": 1}


def tokens_from_segmented(segmented: str) -> List[str]:
    tokens = [tok for tok in segmented.strip().split() if tok]
    if not tokens:
        tokens = [tok for tok in segment_for_project(segmented).split() if tok]
    return tokens


class MLPClassifierTorch(nn.Module):
    def __init__(self, embedding_shape: Sequence[int], hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        embedding_matrix = torch.zeros(tuple(embedding_shape), dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_shape[1], hidden)
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
    def __init__(self, embedding_shape: Sequence[int], n_filters: int = 64, dropout: float = 0.35):
        super().__init__()
        emb_dim = int(embedding_shape[1])
        embedding_matrix = torch.zeros(tuple(embedding_shape), dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
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
    def __init__(self, embedding_shape: Sequence[int], hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        emb_dim = int(embedding_shape[1])
        embedding_matrix = torch.zeros(tuple(embedding_shape), dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
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


def make_model(model_name: str, embedding_shape: Sequence[int]) -> nn.Module:
    if model_name == "mlp":
        return MLPClassifierTorch(embedding_shape)
    if model_name == "cnn":
        return CNNClassifierTorch(embedding_shape)
    if model_name == "rnn":
        return RNNClassifierTorch(embedding_shape)
    raise ValueError(f"Unsupported model: {model_name}")


@dataclass(frozen=True)
class LoadedModel:
    mode: str
    model_name: str
    model: nn.Module
    word2idx: Dict[str, int]
    embedding_shape: Tuple[int, int]


class SubmissionModelService:
    """Load final submission artifacts and run single-text inference."""

    def __init__(self, output_dir: Path | str = DEFAULT_OUTPUT_DIR, max_len: Optional[int] = None) -> None:
        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / "metrics/all_results.json"
        self._metrics = self._load_metrics()
        self.max_len = int(max_len or self._config_value("max_len", DEFAULT_MAX_LEN))
        self.batch_size = int(self._config_value("batch_size", DEFAULT_BATCH_SIZE))
        self._cache: Dict[Tuple[str, str], LoadedModel] = {}

    def _load_metrics(self) -> Dict[str, object]:
        if self.metrics_path.exists():
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        return {"config": {}, "runs": {}}

    def _config_value(self, key: str, default: object) -> object:
        value = (self._metrics.get("config") or {}).get(key, default)
        return default if value in {None, ""} else value

    def available_models(self) -> Dict[str, List[str]]:
        from_metrics = self._metrics.get("runs") or {}
        if from_metrics:
            return {mode: sorted(models.keys()) for mode, models in from_metrics.items()}

        models_dir = self.output_dir / "models"
        result: Dict[str, List[str]] = {}
        if not models_dir.exists():
            return result
        for mode_dir in sorted(path for path in models_dir.iterdir() if path.is_dir()):
            result[mode_dir.name] = sorted(path.stem for path in mode_dir.glob("*.pt"))
        return result

    def metrics_summary(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for mode, models in (self._metrics.get("runs") or {}).items():
            for model_name, payload in models.items():
                clean = payload["clean"]["metrics"]
                ast = payload["ast"]["metrics"]
                robust = payload["robustness"]
                uci = payload.get("uci_en") or {}
                rows.append(
                    {
                        "mode": mode,
                        "model": model_name,
                        "clean_accuracy": clean["accuracy"],
                        "clean_spam_recall": clean["spam_recall"],
                        "ast_accuracy": ast["accuracy"],
                        "ast_spam_recall": ast["spam_recall"],
                        "robust_drop": robust["robust_drop"],
                        "uci_accuracy": (uci.get("metrics") or {}).get("accuracy"),
                    }
                )
        return rows

    def load_model(self, mode: str, model_name: str) -> LoadedModel:
        key = (mode, model_name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        vocab_path = self.output_dir / "word2vec" / f"vocab_{mode}.json"
        model_path = self.output_dir / "models" / mode / f"{model_name}.pt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

        vocab_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        word2idx = {str(word): int(idx) for word, idx in vocab_payload["word2idx"].items()}
        embedding_shape = tuple(int(v) for v in vocab_payload["embedding_shape"])
        if len(embedding_shape) != 2:
            raise ValueError(f"Invalid embedding shape in {vocab_path}: {embedding_shape}")

        model = make_model(model_name, embedding_shape)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        loaded = LoadedModel(mode=mode, model_name=model_name, model=model, word2idx=word2idx, embedding_shape=embedding_shape)
        self._cache[key] = loaded
        return loaded

    def vectorize_texts(self, texts: Sequence[str], word2idx: Dict[str, int]) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        x = np.zeros((len(texts), self.max_len), dtype=np.int64)
        meta: List[Dict[str, object]] = []
        for row, text in enumerate(texts):
            segmented = segment_for_project(text)
            tokens = tokens_from_segmented(segmented)
            used_tokens = tokens[: self.max_len]
            unknown = 0
            for col, token in enumerate(used_tokens):
                idx = word2idx.get(token, 1)
                unknown += int(idx == 1)
                x[row, col] = idx
            meta.append(
                {
                    "text": text,
                    "segmented": segmented,
                    "tokens": used_tokens,
                    "token_count": len(tokens),
                    "used_token_count": len(used_tokens),
                    "unknown_count": unknown,
                    "truncated": len(tokens) > self.max_len,
                }
            )
        return x, meta

    def predict_many(self, texts: Sequence[str], mode: str = "text_ast_fgm", model_name: str = "cnn") -> List[Dict[str, object]]:
        loaded = self.load_model(mode, model_name)
        x, meta = self.vectorize_texts(texts, loaded.word2idx)
        ids = torch.tensor(x)
        outputs: List[Dict[str, object]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                logits = loaded.model(ids[start : start + self.batch_size])
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for offset, row in enumerate(probs):
                    idx = start + offset
                    pred_id = int(row.argmax())
                    outputs.append(
                        {
                            **meta[idx],
                            "mode": mode,
                            "model": model_name,
                            "label_id": pred_id,
                            "label": LABEL_ID_TO_NAME[pred_id],
                            "confidence": float(row[pred_id]),
                            "probabilities": {
                                "spam": float(row[LABEL_NAME_TO_ID["spam"]]),
                                "normal": float(row[LABEL_NAME_TO_ID["normal"]]),
                            },
                        }
                    )
        return outputs

    def predict(self, text: str, mode: str = "text_ast_fgm", model_name: str = "cnn") -> Dict[str, object]:
        return self.predict_many([text], mode=mode, model_name=model_name)[0]

    def compare_models(self, text: str) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for mode, model_names in self.available_models().items():
            for model_name in model_names:
                item = self.predict(text, mode=mode, model_name=model_name)
                rows.append(
                    {
                        "mode": mode,
                        "model": model_name,
                        "label": item["label"],
                        "confidence": item["confidence"],
                        "spam_probability": item["probabilities"]["spam"],
                        "normal_probability": item["probabilities"]["normal"],
                    }
                )
        return rows

    def generate_ast_candidates(
        self,
        text: str,
        label: str = "spam",
        max_variants: Optional[int] = None,
        strength: str = TRAINING_AST_STRENGTH,
    ) -> List[Dict[str, object]]:
        strength = (strength or TRAINING_AST_STRENGTH).lower()
        if max_variants is None:
            max_variants = TRAINING_MAX_VARIANTS_BY_LABEL.get(label, 1)
        is_strong = strength == "strong"
        attacker = ChineseSpamTextAttacker(
            seed=42,
            max_char_replacements=4 if is_strong else 2,
            max_symbol_insertions=4 if is_strong else 2,
        )
        candidates = attacker.generate(text, label, max_variants=max_variants, strength=strength)

        unique = {}
        for item in candidates:
            unique.setdefault(item.adversarial, item)

        rows: List[Dict[str, object]] = []
        for item in list(unique.values())[:max_variants]:
            rows.append(
                {
                    "text": item.adversarial,
                    "attack_type": item.attack_type,
                    "operations": item.operations,
                    "strength": strength,
                }
            )
        return rows

    def attack_search(
        self,
        text: str,
        mode: str = "text_ast_fgm",
        model_name: str = "cnn",
        label: str = "spam",
        max_variants: Optional[int] = None,
        strength: str = TRAINING_AST_STRENGTH,
    ) -> Dict[str, object]:
        original = self.predict(text, mode=mode, model_name=model_name)
        candidates = self.generate_ast_candidates(text, label=label, max_variants=max_variants, strength=strength)
        if not candidates:
            return {"original": original, "candidates": [], "best": None, "strength": strength}

        predictions = self.predict_many([row["text"] for row in candidates], mode=mode, model_name=model_name)
        enriched: List[Dict[str, object]] = []
        for row, pred in zip(candidates, predictions):
            enriched.append({**row, "prediction": pred})

        best = max(enriched, key=lambda row: row["prediction"]["probabilities"]["normal"])
        return {
            "original": original,
            "candidates": enriched,
            "best": best,
            "success": bool(best["prediction"]["label"] == "normal"),
            "strength": strength,
        }
