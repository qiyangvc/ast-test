"""Data-driven vocabulary expansion for text-level AST.

The fixed domain lexicon covers known spam patterns. This module adds a second
layer that is learned from the training split only: discriminative spam terms
are mined with TF-IDF, then expanded with pinyin, symbol insertion, traditional
Chinese conversion, optional Word2Vec similarity, and optional glyph seeds.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pypinyin
except ImportError:  # pragma: no cover - dependency is declared for full runs
    pypinyin = None

try:
    import opencc
except ImportError:  # pragma: no cover - dependency is declared for full runs
    opencc = None

try:
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover - optional for dataset-only builds
    Word2Vec = None


CHAR_SEED_ENV = "AST_CHAR_SEED_JSON"
DEFAULT_CHAR_SEED: Dict[str, List[str]] = {}


def _record_text_label(record: object) -> Tuple[str, str]:
    if isinstance(record, dict):
        return str(record.get("text", "") or ""), str(record.get("label", "") or "").lower()
    return str(getattr(record, "text", "") or ""), str(getattr(record, "label", "") or "").lower()


def _load_char_seed() -> Dict[str, List[str]]:
    seed_path = os.environ.get(CHAR_SEED_ENV)
    if seed_path and Path(seed_path).exists():
        payload = json.loads(Path(seed_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {
                str(key): [str(value) for value in values if str(value).strip()]
                for key, values in payload.items()
                if isinstance(values, list)
            }
    return DEFAULT_CHAR_SEED


def _tokenize_for_tfidf(text: str) -> List[str]:
    return [word.strip() for word in jieba.cut(text) if len(word.strip()) > 1]


def extract_spam_keywords(
    records: Iterable[object],
    top_k: int = 80,
    min_freq: int = 3,
    max_features: int = 5000,
) -> List[str]:
    """Mine spam-specific terms with a spam-vs-normal TF-IDF contrast."""
    spam_texts: List[str] = []
    normal_texts: List[str] = []
    for record in records:
        text, label = _record_text_label(record)
        if not text:
            continue
        if label == "spam":
            spam_texts.append(text)
        elif label == "normal":
            normal_texts.append(text)

    if not spam_texts:
        return []

    all_texts = spam_texts + normal_texts
    labels = ["spam"] * len(spam_texts) + ["normal"] * len(normal_texts)
    min_df = min(len(all_texts), max(2, min(5, len(spam_texts) // 10 + 1)))
    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize_for_tfidf,
        token_pattern=None,
        max_features=max_features,
        min_df=min_df,
    )

    try:
        matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return []

    spam_indices = [idx for idx, label in enumerate(labels) if label == "spam"]
    normal_indices = [idx for idx, label in enumerate(labels) if label == "normal"]
    spam_mean = matrix[spam_indices].mean(axis=0).A1
    normal_mean = matrix[normal_indices].mean(axis=0).A1 if normal_indices else 0.0
    diff = spam_mean - normal_mean

    names = vectorizer.get_feature_names_out()
    word_freq = Counter()
    for text in spam_texts:
        word_freq.update(_tokenize_for_tfidf(text))

    ranked = [
        str(names[idx])
        for idx in diff.argsort()[::-1]
        if diff[idx] > 0 and word_freq.get(str(names[idx]), 0) >= min_freq
    ]
    return ranked[:top_k]


def build_pinyin_variants(word: str) -> List[str]:
    if pypinyin is None or len(word) < 2:
        return []
    pinyin_items = pypinyin.lazy_pinyin(word)
    variants = {
        "".join(pinyin_items),
        "".join(item[0] for item in pinyin_items if item),
        " ".join(pinyin_items),
    }
    if len(pinyin_items) >= 2 and pinyin_items[0]:
        variants.add(pinyin_items[0][0] + "".join(pinyin_items[1:]))
    return sorted(value for value in variants if value and value != word)


def build_symbol_variants(
    word: str,
    symbols: Sequence[str] = (" ", "-", "_", ".", "*", "·"),
) -> List[str]:
    if len(word) < 2:
        return []
    return [symbol.join(list(word)) for symbol in symbols]


def build_traditional_variants(word: str) -> List[str]:
    if opencc is None:
        return []
    converter = opencc.OpenCC("s2t")
    traditional = converter.convert(word)
    return [traditional] if traditional != word else []


def build_word2vec_variants(
    word: str,
    w2v_path: Optional[Path],
    topn: int = 5,
    min_score: float = 0.5,
) -> List[str]:
    if w2v_path is None or Word2Vec is None or not Path(w2v_path).exists():
        return []
    try:
        model = Word2Vec.load(str(w2v_path))
    except Exception:
        return []
    if word not in model.wv:
        return []
    return [
        str(candidate)
        for candidate, score in model.wv.most_similar(word, topn=topn * 2)
        if score >= min_score and str(candidate) != word
    ][:topn]


def build_glyph_variants(word: str) -> List[str]:
    seed = _load_char_seed()
    variants: set[str] = set()
    for idx, char in enumerate(word):
        for replacement in seed.get(char, []):
            chars = list(word)
            chars[idx] = replacement
            variants.add("".join(chars))
    return sorted(variants)


def build_keyword_variants(word: str, w2v_path: Optional[Path] = None) -> List[str]:
    candidates: set[str] = set()
    candidates.update(build_pinyin_variants(word))
    candidates.update(build_symbol_variants(word))
    candidates.update(build_traditional_variants(word))
    candidates.update(build_word2vec_variants(word, w2v_path))
    candidates.update(build_glyph_variants(word))
    candidates.discard(word)
    return sorted(candidate for candidate in candidates if candidate.strip())


def build_dynamic_attack_vocab(
    records: Iterable[object],
    w2v_path: Optional[Path] = None,
    top_k: int = 80,
) -> Dict[str, List[str]]:
    """Build a keyword-to-variant table from training records."""
    vocab: Dict[str, List[str]] = {}
    for keyword in extract_spam_keywords(records, top_k=top_k):
        variants = build_keyword_variants(keyword, w2v_path=w2v_path)
        if variants:
            vocab[keyword] = variants
    return vocab


def summarize_vocab(vocab: Dict[str, List[str]]) -> Dict[str, object]:
    return {
        "n_keywords": len(vocab),
        "n_candidates": sum(len(values) for values in vocab.values()),
        "examples": [
            {"keyword": key, "candidates": values[:5]}
            for key, values in sorted(vocab.items(), key=lambda item: -len(item[1]))[:10]
        ],
    }
