from __future__ import annotations

from pathlib import Path
from typing import Any


class Config:
    """Shared paths and experiment defaults for the AST spam-text project."""

    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"

    EXTERNAL_DIR = DATA_DIR / "external"
    MSG_LOG_DIR = EXTERNAL_DIR / "raw" / "tensorlayer_text_antispam" / "msglog"

    AST_DATASET_CONFIG = {
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "seed": 42,
        "max_variants_spam": 2,
        "max_variants_normal": 1,
        "ast_strength": "mild",
        "output_dir_name": "ast_experiment",
    }

    STRONG_AST_DATASET_CONFIG = {
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "seed": 42,
        "max_variants_spam": 4,
        "max_variants_normal": 1,
        "ast_strength": "strong",
        "output_dir_name": "ast_experiment_strong",
    }

    SUBMISSION_TRAINING_CONFIG = {
        "modes": ["baseline", "text_ast", "embedding_fgm", "text_ast_fgm"],
        "models": ["mlp", "cnn", "rnn"],
        "vector_size": 200,
        "max_vocab": 50000,
        "max_len": 64,
        "w2v_epochs": 20,
        "clf_epochs": 10,
        "batch_size": 512,
        "fgm_epsilon": 0.5,
        "focal_gamma": 2.0,
        "learning_rate": 1e-3,
        "run_ensemble": True,
        "ensemble_models": ["mlp", "cnn", "rnn"],
    }

    FULL_SUBMISSION_TRAINING_CONFIG = {
        **SUBMISSION_TRAINING_CONFIG,
        "modes": [
            "baseline",
            "focal",
            "text_ast",
            "text_ast_focal",
            "embedding_fgm",
            "embedding_fgm_focal",
            "text_ast_fgm",
            "text_ast_fgm_focal",
        ],
        "models": ["mlp", "cnn", "rnn", "bilstm_attn"],
        "ensemble_models": ["mlp", "cnn", "rnn", "bilstm_attn"],
    }

    @classmethod
    def get(cls, key: str) -> Any:
        return getattr(cls, key, None)

    @classmethod
    def ensure_dirs(cls) -> None:
        for path in (cls.DATA_DIR, cls.OUTPUT_DIR, cls.EXTERNAL_DIR):
            Path(path).mkdir(parents=True, exist_ok=True)
