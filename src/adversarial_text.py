"""Text-level adversarial sample generation for Chinese spam detection.

The generators in this module are intentionally rule based. They model common
spam evasion patterns found in Chinese short messages: glyph/phonetic variants,
spacing, symbol insertion, traditional characters, contact obfuscation,
digit-letter confusion, pinyin abbreviation, keyword reordering, and stronger
multi-strategy rewrites. Each generated sample keeps metadata so experiments can
report performance by attack type instead of only overall accuracy.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import jieba
except ImportError:  # pragma: no cover - handled at runtime for minimal envs
    jieba = None


AttackType = str
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "lexicons" / "chinese_spam_ast_lexicon.json"
_JIEBA_LEXICON_LOADED = False
_REGISTERED_JIEBA_TERMS: set[str] = set()


@dataclass
class AttackResult:
    """A generated adversarial text and the operations used to create it."""

    original: str
    adversarial: str
    attack_type: AttackType
    operations: List[str] = field(default_factory=list)


def _tuple_map(payload: object, field_name: str) -> Dict[str, Tuple[str, ...]]:
    if not isinstance(payload, dict):
        raise ValueError(f"Lexicon field {field_name!r} must be an object.")
    result: Dict[str, Tuple[str, ...]] = {}
    for key, values in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Lexicon field {field_name!r} contains an invalid key: {key!r}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Lexicon field {field_name!r}.{key!r} must be a non-empty list.")
        cleaned = tuple(str(value).strip() for value in values if str(value).strip())
        if not cleaned:
            raise ValueError(f"Lexicon field {field_name!r}.{key!r} has no usable variants.")
        result[key] = cleaned
    return result


def _string_map(payload: object, field_name: str) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"Lexicon field {field_name!r} must be an object.")
    result: Dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Lexicon field {field_name!r} contains an invalid key: {key!r}")
        value_text = str(value).strip()
        if not value_text:
            raise ValueError(f"Lexicon field {field_name!r}.{key!r} has an empty replacement.")
        result[key] = value_text
    return result


def _pair_list(payload: object, field_name: str) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(payload, list):
        raise ValueError(f"Lexicon field {field_name!r} must be a list.")
    pairs: List[Tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Lexicon field {field_name!r} contains an invalid pair: {item!r}")
        key, value = (str(item[0]).strip(), str(item[1]).strip())
        if not key or not value:
            raise ValueError(f"Lexicon field {field_name!r} contains an empty pair: {item!r}")
        pairs.append((key, value))
    return tuple(pairs)


def _string_tuple(payload: object, field_name: str) -> Tuple[str, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"Lexicon field {field_name!r} must be a list.")
    values = tuple(str(item).strip() for item in payload if str(item).strip())
    if not values:
        raise ValueError(f"Lexicon field {field_name!r} must contain at least one value.")
    return values


def _string_tuple_map(payload: object, field_name: str) -> Dict[str, Tuple[str, ...]]:
    return _tuple_map(payload, field_name)


def _pattern_specs(payload: object, field_name: str) -> Tuple[Dict[str, object], ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Lexicon field {field_name!r} must be a non-empty list.")
    specs: List[Dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Lexicon field {field_name!r} contains an invalid pattern spec: {item!r}")
        name = str(item.get("name") or "").strip()
        pattern = str(item.get("pattern") or "").strip()
        if not name or not pattern:
            raise ValueError(f"Lexicon field {field_name!r} pattern specs need name and pattern.")
        spec: Dict[str, object] = {"name": name, "pattern": pattern}
        for key in ("obfuscation_replacements", "split_replacements", "hint_replacements"):
            if key in item:
                spec[key] = _string_tuple(item[key], f"{field_name}.{name}.{key}")
        spec["number_obfuscation"] = bool(item.get("number_obfuscation", False))
        specs.append(spec)
    return tuple(specs)


def _special_replacements(payload: object, field_name: str) -> Dict[str, Dict[str, object]]:
    if not isinstance(payload, list):
        raise ValueError(f"Lexicon field {field_name!r} must be a list.")
    result: Dict[str, Dict[str, object]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Lexicon field {field_name!r} contains an invalid item: {item!r}")
        keyword = str(item.get("keyword") or "").strip()
        if not keyword:
            raise ValueError(f"Lexicon field {field_name!r} item has empty keyword.")
        result[keyword] = {
            "probability": float(item.get("probability", 1.0)),
            "replacements": _string_tuple(item.get("replacements"), f"{field_name}.{keyword}.replacements"),
        }
    return result


def _attack_profiles(payload: object, field_name: str) -> Dict[str, Tuple[str, ...]]:
    required = {"spam_mild", "normal_mild", "spam_balanced_extra", "spam_strong_extra"}
    result = _tuple_map(payload, field_name)
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Lexicon field {field_name!r} missing profiles: {missing}")
    return result


def _runtime_config(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Lexicon field 'runtime_config' must be an object.")
    required = {
        "digit_variants",
        "insert_symbols",
        "strong_insert_symbols",
        "number_separators",
        "amount_regex",
        "url_regex",
        "contact_number_regex",
        "contact_split_number_regex",
        "amount_unit_variants",
        "amount_insert_symbols",
        "contact_patterns",
        "multi_keyword_special_replacements",
        "placeholder_context_hints",
        "default_context_hints",
        "contact_action_hints",
        "attack_profiles",
        "attack_priority",
        "mixed_attack_sequence",
        "strong_mixed_attack_sequence",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"AST lexicon runtime_config missing fields: {missing}")
    return {
        "digit_variants": _string_tuple_map(payload["digit_variants"], "runtime_config.digit_variants"),
        "insert_symbols": _string_tuple(payload["insert_symbols"], "runtime_config.insert_symbols"),
        "strong_insert_symbols": _string_tuple(payload["strong_insert_symbols"], "runtime_config.strong_insert_symbols"),
        "number_separators": _string_tuple(payload["number_separators"], "runtime_config.number_separators"),
        "amount_regex": str(payload["amount_regex"]),
        "url_regex": str(payload["url_regex"]),
        "contact_number_regex": str(payload["contact_number_regex"]),
        "contact_split_number_regex": str(payload["contact_split_number_regex"]),
        "amount_unit_variants": _string_tuple_map(payload["amount_unit_variants"], "runtime_config.amount_unit_variants"),
        "amount_insert_symbols": _string_tuple(payload["amount_insert_symbols"], "runtime_config.amount_insert_symbols"),
        "contact_patterns": _pattern_specs(payload["contact_patterns"], "runtime_config.contact_patterns"),
        "multi_keyword_special_replacements": _special_replacements(
            payload["multi_keyword_special_replacements"],
            "runtime_config.multi_keyword_special_replacements",
        ),
        "placeholder_context_hints": _pair_list(
            payload["placeholder_context_hints"],
            "runtime_config.placeholder_context_hints",
        ),
        "default_context_hints": _string_tuple(payload["default_context_hints"], "runtime_config.default_context_hints"),
        "contact_action_hints": _string_map(payload["contact_action_hints"], "runtime_config.contact_action_hints"),
        "attack_profiles": _attack_profiles(payload["attack_profiles"], "runtime_config.attack_profiles"),
        "attack_priority": _string_tuple(payload["attack_priority"], "runtime_config.attack_priority"),
        "mixed_attack_sequence": _string_tuple(payload["mixed_attack_sequence"], "runtime_config.mixed_attack_sequence"),
        "strong_mixed_attack_sequence": _string_tuple(
            payload["strong_mixed_attack_sequence"],
            "runtime_config.strong_mixed_attack_sequence",
        ),
    }


def load_ast_lexicon(path: Path | str = DEFAULT_LEXICON_PATH) -> Dict[str, object]:
    """Load and validate the external AST lexicon.

    The lexicon is intentionally outside this Python module so keyword coverage
    can be audited and extended without changing attack logic.
    """
    lexicon_path = Path(path)
    payload = json.loads(lexicon_path.read_text(encoding="utf-8"))
    required = {
        "phrase_variants",
        "strong_phrase_variants",
        "pinyin_abbreviations",
        "char_variants",
        "traditional_variants",
        "multi_keyword_obfuscation_keywords",
        "amount_fallbacks",
        "url_keyword_replacements",
        "benefit_hints",
        "default_benefits",
        "default_contact_hints",
        "semantic_templates",
        "urgency_hints",
        "runtime_config",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"AST lexicon missing required fields: {missing}")
    return {
        "phrase_variants": _tuple_map(payload["phrase_variants"], "phrase_variants"),
        "strong_phrase_variants": _tuple_map(payload["strong_phrase_variants"], "strong_phrase_variants"),
        "pinyin_abbreviations": _tuple_map(payload["pinyin_abbreviations"], "pinyin_abbreviations"),
        "char_variants": _tuple_map(payload["char_variants"], "char_variants"),
        "traditional_variants": _string_map(payload["traditional_variants"], "traditional_variants"),
        "multi_keyword_obfuscation_keywords": _string_tuple(
            payload["multi_keyword_obfuscation_keywords"],
            "multi_keyword_obfuscation_keywords",
        ),
        "amount_fallbacks": _pair_list(payload["amount_fallbacks"], "amount_fallbacks"),
        "url_keyword_replacements": _pair_list(payload["url_keyword_replacements"], "url_keyword_replacements"),
        "benefit_hints": _pair_list(payload["benefit_hints"], "benefit_hints"),
        "default_benefits": _string_tuple(payload["default_benefits"], "default_benefits"),
        "default_contact_hints": _string_tuple(payload["default_contact_hints"], "default_contact_hints"),
        "semantic_templates": _string_tuple(payload["semantic_templates"], "semantic_templates"),
        "urgency_hints": _string_tuple(payload["urgency_hints"], "urgency_hints"),
        "runtime_config": _runtime_config(payload["runtime_config"]),
    }


def _terms_from_tuple_map(payload: Dict[str, Tuple[str, ...]]) -> Iterable[str]:
    for key, values in payload.items():
        yield key
        yield from values


def _normalize_variant_map(
    payload: Optional[Mapping[str, Sequence[str]]],
    field_name: str,
) -> Dict[str, Tuple[str, ...]]:
    if not payload:
        return {}
    result: Dict[str, Tuple[str, ...]] = {}
    for key, values in payload.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError(f"{field_name} contains an empty keyword.")
        if isinstance(values, str):
            raw_values = [values]
        else:
            raw_values = list(values)
        cleaned = tuple(
            str(value).strip()
            for value in raw_values
            if str(value).strip() and str(value).strip() != key_text
        )
        if cleaned:
            result[key_text] = cleaned
    return result


def _merge_variant_maps(
    base: Dict[str, Tuple[str, ...]],
    extra: Optional[Mapping[str, Sequence[str]]],
    field_name: str,
) -> Dict[str, Tuple[str, ...]]:
    merged: Dict[str, List[str]] = {key: list(values) for key, values in base.items()}
    for key, values in _normalize_variant_map(extra, field_name).items():
        bucket = merged.setdefault(key, [])
        seen = set(bucket)
        for value in values:
            if value not in seen:
                bucket.append(value)
                seen.add(value)
    return {key: tuple(values) for key, values in merged.items()}


def register_jieba_terms(terms: Iterable[str], freq: int = 200000) -> None:
    """Register runtime vocabulary with jieba for stable segmentation."""
    if jieba is None:
        return
    for term in terms:
        term = str(term).strip()
        if (
            len(term) < 2
            or any(char.isspace() for char in term)
            or "{" in term
            or "}" in term
            or term in _REGISTERED_JIEBA_TERMS
        ):
            continue
        jieba.add_word(term, freq=freq)
        _REGISTERED_JIEBA_TERMS.add(term)


def ast_lexicon_terms(path: Path | str = DEFAULT_LEXICON_PATH) -> List[str]:
    """Return AST domain terms that should be visible to segmentation/audits."""
    lexicon = load_ast_lexicon(path)
    terms = set()
    for field in ("phrase_variants", "strong_phrase_variants", "pinyin_abbreviations"):
        terms.update(_terms_from_tuple_map(lexicon[field]))
    terms.update(lexicon["multi_keyword_obfuscation_keywords"])
    terms.update(lexicon["traditional_variants"].keys())
    terms.update(lexicon["traditional_variants"].values())
    for field in ("amount_fallbacks", "url_keyword_replacements", "benefit_hints"):
        for key, value in lexicon[field]:
            terms.add(key)
            terms.add(value)
    for field in ("default_benefits", "default_contact_hints", "semantic_templates", "urgency_hints"):
        terms.update(lexicon[field])
    runtime = lexicon["runtime_config"]
    terms.update(runtime["insert_symbols"])
    terms.update(runtime["strong_insert_symbols"])
    terms.update(runtime["number_separators"])
    terms.update(runtime["amount_insert_symbols"])
    terms.update(runtime["default_context_hints"])
    terms.update(runtime["contact_action_hints"].keys())
    terms.update(runtime["contact_action_hints"].values())
    for values in runtime["digit_variants"].values():
        terms.update(values)
    for values in runtime["amount_unit_variants"].values():
        terms.update(values)
    for _, hint in runtime["placeholder_context_hints"]:
        terms.add(hint)
    for spec in runtime["contact_patterns"]:
        for key in ("obfuscation_replacements", "split_replacements", "hint_replacements"):
            terms.update(spec.get(key, ()))
    for item in runtime["multi_keyword_special_replacements"].values():
        terms.update(item["replacements"])

    cleaned = {
        term.strip()
        for term in terms
        if isinstance(term, str)
        and len(term.strip()) >= 2
        and "{" not in term
        and "}" not in term
    }
    return sorted(cleaned, key=len, reverse=True)


def _ensure_jieba_lexicon_loaded() -> None:
    """Register AST domain vocabulary with jieba once per process."""
    global _JIEBA_LEXICON_LOADED
    if jieba is None or _JIEBA_LEXICON_LOADED:
        return
    register_jieba_terms(ast_lexicon_terms(DEFAULT_LEXICON_PATH))
    _JIEBA_LEXICON_LOADED = True


def compact_segmented_text(text: str) -> str:
    """Remove token separators from a segmented line while preserving content."""
    return re.sub(r"\s+", "", text.strip())


def segment_for_project(text: str) -> str:
    """Segment text into the whitespace-separated format used by this project."""
    text = text.strip()
    if not text:
        return ""
    if jieba is None:
        # Fallback keeps the pipeline usable in environments where jieba is not
        # installed yet. It is less linguistically accurate, so the manifest
        # records the generator settings for reproducibility.
        return " ".join([ch for ch in text if not ch.isspace()])
    _ensure_jieba_lexicon_loaded()
    words = [w.strip() for w in jieba.cut(text) if w.strip()]
    return " ".join(words)


class ChineseSpamTextAttacker:
    """Generate realistic Chinese spam-text adversarial variants."""

    # Domain lexicons are loaded from lexicons/chinese_spam_ast_lexicon.json
    # in __init__, so dataset building, confidence search, and the web UI use
    # one auditable source of AST vocabulary.

    def __init__(
        self,
        seed: int = 42,
        max_char_replacements: int = 2,
        max_symbol_insertions: int = 2,
        lexicon_path: Path | str = DEFAULT_LEXICON_PATH,
        phrase_variants: Optional[Mapping[str, Sequence[str]]] = None,
        strong_phrase_variants: Optional[Mapping[str, Sequence[str]]] = None,
        pinyin_abbreviations: Optional[Mapping[str, Sequence[str]]] = None,
        char_variants: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_char_replacements = max_char_replacements
        self.max_symbol_insertions = max_symbol_insertions
        self.lexicon_path = Path(lexicon_path)
        lexicon = load_ast_lexicon(self.lexicon_path)
        runtime = lexicon["runtime_config"]
        strong_extra = strong_phrase_variants if strong_phrase_variants is not None else phrase_variants
        self.PHRASE_VARIANTS = _merge_variant_maps(lexicon["phrase_variants"], phrase_variants, "phrase_variants")
        self.STRONG_PHRASE_VARIANTS = _merge_variant_maps(
            lexicon["strong_phrase_variants"],
            strong_extra,
            "strong_phrase_variants",
        )
        self.PINYIN_ABBREVIATIONS = _merge_variant_maps(
            lexicon["pinyin_abbreviations"],
            pinyin_abbreviations,
            "pinyin_abbreviations",
        )
        self.CHAR_VARIANTS = _merge_variant_maps(lexicon["char_variants"], char_variants, "char_variants")
        self.TRADITIONAL_VARIANTS = lexicon["traditional_variants"]
        self.MULTI_KEYWORD_OBFUSCATION_KEYWORDS = lexicon["multi_keyword_obfuscation_keywords"]
        self.AMOUNT_FALLBACKS = lexicon["amount_fallbacks"]
        self.URL_KEYWORD_REPLACEMENTS = lexicon["url_keyword_replacements"]
        self.BENEFIT_HINTS = lexicon["benefit_hints"]
        self.DEFAULT_BENEFITS = lexicon["default_benefits"]
        self.DEFAULT_CONTACT_HINTS = lexicon["default_contact_hints"]
        self.SEMANTIC_TEMPLATES = lexicon["semantic_templates"]
        self.URGENCY_HINTS = lexicon["urgency_hints"]
        self.DIGIT_VARIANTS = runtime["digit_variants"]
        self.INSERT_SYMBOLS = runtime["insert_symbols"]
        self.STRONG_INSERT_SYMBOLS = runtime["strong_insert_symbols"]
        self.NUMBER_SEPARATORS = runtime["number_separators"]
        self.AMOUNT_PATTERN = re.compile(runtime["amount_regex"])
        self.URL_PATTERN = re.compile(runtime["url_regex"])
        self.CONTACT_NUMBER_PATTERN = re.compile(runtime["contact_number_regex"])
        self.CONTACT_SPLIT_NUMBER_PATTERN = re.compile(runtime["contact_split_number_regex"])
        self.AMOUNT_UNIT_VARIANTS = runtime["amount_unit_variants"]
        self.AMOUNT_INSERT_SYMBOLS = runtime["amount_insert_symbols"]
        self.CONTACT_PATTERNS = [(re.compile(spec["pattern"]), spec) for spec in runtime["contact_patterns"]]
        self.MULTI_KEYWORD_SPECIAL_REPLACEMENTS = runtime["multi_keyword_special_replacements"]
        self.PLACEHOLDER_CONTEXT_HINTS = runtime["placeholder_context_hints"]
        self.DEFAULT_CONTEXT_HINTS = runtime["default_context_hints"]
        self.CONTACT_ACTION_HINTS = runtime["contact_action_hints"]
        self.ATTACK_PROFILES = runtime["attack_profiles"]
        self.ATTACK_PRIORITY = {attack_type: idx for idx, attack_type in enumerate(runtime["attack_priority"])}
        self.MIXED_ATTACK_SEQUENCE = runtime["mixed_attack_sequence"]
        self.STRONG_MIXED_ATTACK_SEQUENCE = runtime["strong_mixed_attack_sequence"]
        register_jieba_terms(_terms_from_tuple_map(self.PHRASE_VARIANTS))
        register_jieba_terms(_terms_from_tuple_map(self.STRONG_PHRASE_VARIANTS))
        register_jieba_terms(_terms_from_tuple_map(self.PINYIN_ABBREVIATIONS))

    def lexicon_summary(self) -> Dict[str, int]:
        """Return coverage counts for audit scripts and reports."""
        return {
            "phrase_variants": len(self.PHRASE_VARIANTS),
            "strong_phrase_variants": len(self.STRONG_PHRASE_VARIANTS),
            "pinyin_abbreviations": len(self.PINYIN_ABBREVIATIONS),
            "char_variants": len(self.CHAR_VARIANTS),
            "traditional_variants": len(self.TRADITIONAL_VARIANTS),
            "multi_keyword_obfuscation_keywords": len(self.MULTI_KEYWORD_OBFUSCATION_KEYWORDS),
            "amount_fallbacks": len(self.AMOUNT_FALLBACKS),
            "url_keyword_replacements": len(self.URL_KEYWORD_REPLACEMENTS),
            "benefit_hints": len(self.BENEFIT_HINTS),
            "semantic_templates": len(self.SEMANTIC_TEMPLATES),
            "runtime_contact_patterns": len(self.CONTACT_PATTERNS),
            "runtime_attack_profiles": len(self.ATTACK_PROFILES),
        }

    def generate(
        self,
        text: str,
        label: str,
        max_variants: int = 2,
        attack_types: Optional[Sequence[AttackType]] = None,
        strength: str = "mild",
    ) -> List[AttackResult]:
        """Generate adversarial variants for one record.

        Spam samples receive semantic evasion attacks. Normal samples only get
        conservative format perturbations by default so their label remains
        credible.
        """
        original = compact_segmented_text(text)
        if not original:
            return []

        if attack_types is None:
            attack_types = self.attack_types_for_strength(label, strength)

        candidates: List[AttackResult] = []
        for attack_type in attack_types:
            result = self.apply_attack(original, attack_type)
            if result and result.adversarial != original:
                candidates.append(result)

        unique: Dict[str, AttackResult] = {}
        for item in candidates:
            unique.setdefault(item.adversarial, item)

        results = list(unique.values())
        if (strength or "mild").lower() in {"balanced", "strong"}:
            results.sort(key=lambda item: self.ATTACK_PRIORITY.get(item.attack_type, 20))
        else:
            self.rng.shuffle(results)
        return results[:max(0, max_variants)]

    def attack_types_for_strength(self, label: str, strength: str = "mild") -> Sequence[AttackType]:
        """Return attack types by perturbation strength.

        ``mild`` preserves the original assignment setting. ``balanced`` and
        ``strong`` add larger, spam-specific rewrites for the browser demo and
        future stress tests.
        """
        base = tuple(self.default_attack_types(label))
        if label != "spam":
            return base

        strength = (strength or "mild").lower()
        if strength == "strong":
            return (*self.ATTACK_PROFILES["spam_strong_extra"], *base)
        if strength == "balanced":
            return (*self.ATTACK_PROFILES["spam_balanced_extra"], *base)
        return base

    def default_attack_types(self, label: str) -> Sequence[AttackType]:
        if label == "spam":
            return self.ATTACK_PROFILES["spam_mild"]
        return self.ATTACK_PROFILES["normal_mild"]

    def _handler_for_attack_type(self, attack_type: AttackType):
        handlers = {
            "phrase_variant": self._phrase_variant,
            "strong_phrase_variant": self._strong_phrase_variant,
            "glyph_variant": self._char_variant,
            "phonetic_variant": self._char_variant,
            "symbol_insertion": self._symbol_insertion,
            "traditional_variant": self._traditional_variant,
            "digit_letter_mix": self._digit_letter_mix,
            "contact_obfuscation": self._contact_obfuscation,
            "pinyin_abbreviation": self._pinyin_abbreviation,
            "multi_keyword_obfuscation": self._multi_keyword_obfuscation,
            "contact_split": self._contact_split,
            "url_obfuscation": self._url_obfuscation,
            "amount_obfuscation": self._amount_obfuscation,
            "semantic_rewrite": self._semantic_rewrite,
            "mixed": self._mixed_attack,
            "strong_mixed": self._strong_mixed_attack,
        }
        handler = handlers.get(attack_type)
        if handler is None:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        return handler

    def apply_attack(self, text: str, attack_type: AttackType) -> Optional[AttackResult]:
        handler = self._handler_for_attack_type(attack_type)
        attacked, operations = handler(text)
        if attacked == text:
            return None
        return AttackResult(text, attacked, attack_type, operations)

    def _phrase_variant(self, text: str) -> Tuple[str, List[str]]:
        keys = [key for key in self.PHRASE_VARIANTS if key in text]
        if not keys:
            return text, []
        key = self.rng.choice(sorted(keys, key=len, reverse=True))
        replacement = self.rng.choice(tuple(self.PHRASE_VARIANTS[key]))
        return text.replace(key, replacement, 1), [f"{key}->{replacement}"]

    def _strong_phrase_variant(self, text: str) -> Tuple[str, List[str]]:
        keys = [key for key in self.STRONG_PHRASE_VARIANTS if key in text]
        if not keys:
            return text, []
        key = self.rng.choice(sorted(keys, key=len, reverse=True))
        replacement = self.rng.choice(tuple(self.STRONG_PHRASE_VARIANTS[key]))
        return text.replace(key, replacement, 1), [f"{key}->{replacement}"]

    def _pinyin_abbreviation(self, text: str) -> Tuple[str, List[str]]:
        keys = [key for key in self.PINYIN_ABBREVIATIONS if key in text]
        if not keys:
            return text, []

        current = text
        operations: List[str] = []
        self.rng.shuffle(keys)
        for key in keys[:3]:
            if key not in current:
                continue
            replacement = self.rng.choice(tuple(self.PINYIN_ABBREVIATIONS[key]))
            current = current.replace(key, replacement, 1)
            operations.append(f"{key}->{replacement}")
        return current, operations

    def _char_variant(self, text: str) -> Tuple[str, List[str]]:
        positions = [idx for idx, ch in enumerate(text) if ch in self.CHAR_VARIANTS]
        if not positions:
            return text, []

        chars = list(text)
        self.rng.shuffle(positions)
        operations: List[str] = []
        for idx in positions[: self.max_char_replacements]:
            old = chars[idx]
            new = self.rng.choice(tuple(self.CHAR_VARIANTS[old]))
            chars[idx] = new
            operations.append(f"{old}@{idx}->{new}")
        return "".join(chars), operations

    def _symbol_insertion(self, text: str) -> Tuple[str, List[str]]:
        if len(text) < 2:
            return text, []

        chars = list(text)
        possible = list(range(1, len(chars)))
        self.rng.shuffle(possible)
        insertions = sorted(possible[: self.max_symbol_insertions], reverse=True)
        operations: List[str] = []
        for idx in insertions:
            symbol = self.rng.choice(tuple(self.INSERT_SYMBOLS))
            chars.insert(idx, symbol)
            operations.append(f"insert({symbol})@{idx}")
        return "".join(chars), operations

    def _multi_keyword_obfuscation(self, text: str) -> Tuple[str, List[str]]:
        keys = [key for key in self.MULTI_KEYWORD_OBFUSCATION_KEYWORDS if key in text]
        if not keys:
            return text, []

        current = text
        operations: List[str] = []
        keys = sorted(keys, key=len, reverse=True)
        for key in keys[:4]:
            if key not in current or len(key) < 2:
                continue
            symbol = self.rng.choice(tuple(self.STRONG_INSERT_SYMBOLS))
            replacement = symbol.join(list(key))
            special = self.MULTI_KEYWORD_SPECIAL_REPLACEMENTS.get(key)
            if special and self.rng.random() < float(special["probability"]):
                replacement = self.rng.choice(tuple(special["replacements"]))
            current = current.replace(key, replacement, 1)
            operations.append(f"{key}->{replacement}")
        return current, operations

    def _traditional_variant(self, text: str) -> Tuple[str, List[str]]:
        chars = list(text)
        operations: List[str] = []
        for idx, ch in enumerate(chars):
            if ch in self.TRADITIONAL_VARIANTS and len(operations) < self.max_char_replacements:
                new = self.TRADITIONAL_VARIANTS[ch]
                chars[idx] = new
                operations.append(f"{ch}@{idx}->{new}")
        return "".join(chars), operations

    def _amount_obfuscation(self, text: str) -> Tuple[str, List[str]]:
        match = self.AMOUNT_PATTERN.search(text)
        if match:
            value = match.group(0)
            replacement = self._obfuscate_amount(value)
            return text[: match.start()] + replacement + text[match.end() :], [f"{value}->{replacement}"]

        fallback = []
        for key, replacement in self.AMOUNT_FALLBACKS:
            if key in text:
                fallback.append((key, replacement))
        if not fallback:
            return text, []
        key, replacement = self.rng.choice(fallback)
        return text.replace(key, replacement, 1), [f"{key}->{replacement}"]

    def _digit_letter_mix(self, text: str) -> Tuple[str, List[str]]:
        positions = [idx for idx, ch in enumerate(text) if ch in self.DIGIT_VARIANTS]
        if not positions:
            return text, []
        chars = list(text)
        self.rng.shuffle(positions)
        operations: List[str] = []
        for idx in positions[: self.max_char_replacements]:
            old = chars[idx]
            new = self.rng.choice(tuple(self.DIGIT_VARIANTS[old]))
            chars[idx] = new
            operations.append(f"{old}@{idx}->{new}")
        return "".join(chars), operations

    def _contact_obfuscation(self, text: str) -> Tuple[str, List[str]]:
        for pattern, spec in self.CONTACT_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            matched = match.group(0)
            choices = spec.get("obfuscation_replacements")
            if choices:
                replacement = self.rng.choice(tuple(choices))
            elif spec.get("number_obfuscation"):
                replacement = self._obfuscate_number(matched)
            else:
                continue
            return (
                text[: match.start()] + replacement + text[match.end() :],
                [f"{matched}->{replacement}"],
            )
        return text, []

    def _contact_split(self, text: str) -> Tuple[str, List[str]]:
        for pattern, spec in self.CONTACT_PATTERNS:
            choices = spec.get("split_replacements")
            if not choices:
                continue
            match = pattern.search(text)
            if not match:
                continue
            matched = match.group(0)
            replacement = self.rng.choice(tuple(choices))
            return text[: match.start()] + replacement + text[match.end() :], [f"{matched}->{replacement}"]

        match = self.CONTACT_SPLIT_NUMBER_PATTERN.search(text)
        if match:
            matched = match.group(0)
            replacement = self._group_number(matched)
            return text[: match.start()] + replacement + text[match.end() :], [f"{matched}->{replacement}"]
        return text, []

    def _url_obfuscation(self, text: str) -> Tuple[str, List[str]]:
        url_match = self.URL_PATTERN.search(text)
        if url_match:
            value = url_match.group(0)
            replacement = value.replace("http://", "hxxp://").replace("https://", "hxxps://")
            replacement = replacement.replace("www.", "w w w点").replace(".", "点").replace("/", "/ ")
            return text[: url_match.start()] + replacement + text[url_match.end() :], [f"{value}->{replacement}"]

        for key, replacement in self.URL_KEYWORD_REPLACEMENTS:
            if key in text:
                return text.replace(key, replacement, 1), [f"{key}->{replacement}"]
        return text, []

    def _semantic_rewrite(self, text: str) -> Tuple[str, List[str]]:
        contact = self._extract_contact_hint(text)
        benefit = self._extract_benefit_hint(text)
        context = self._extract_context_hint(text)
        urgency = self.rng.choice(tuple(self.URGENCY_HINTS))
        contact_action = self._contact_action_hint(contact)
        template = self.rng.choice(tuple(self.SEMANTIC_TEMPLATES))
        rewritten = template.format(
            benefit=benefit,
            contact_action=contact_action,
            urgency=urgency,
            context=context,
        )
        return rewritten, [f"semantic_rewrite(benefit={benefit},contact={contact},context={context})"]

    def _mixed_attack(self, text: str) -> Tuple[str, List[str]]:
        operations: List[str] = []
        current = text
        for attack_type in self.MIXED_ATTACK_SEQUENCE:
            handler = self._handler_for_attack_type(attack_type)
            current, ops = handler(current)
            operations.extend(ops)
        return current, operations

    def _strong_mixed_attack(self, text: str) -> Tuple[str, List[str]]:
        operations: List[str] = []
        current = text
        for attack_type in self.STRONG_MIXED_ATTACK_SEQUENCE:
            handler = self._handler_for_attack_type(attack_type)
            current, ops = handler(current)
            operations.extend(ops)
        if not operations:
            current, operations = self._semantic_rewrite(text)
        return current, operations

    def _obfuscate_number(self, value: str) -> str:
        if len(value) < 4:
            return value
        chars = list(value)
        idx = self.rng.randrange(1, len(chars) - 1)
        chars.insert(idx, self.rng.choice(tuple(self.NUMBER_SEPARATORS)))
        return "".join(chars)

    def _group_number(self, value: str) -> str:
        groups = []
        current = ""
        for idx, ch in enumerate(value):
            current += self.rng.choice(tuple(self.DIGIT_VARIANTS.get(ch, (ch,)))) if self.rng.random() < 0.35 else ch
            if idx % 3 == 2 and idx != len(value) - 1:
                groups.append(current)
                current = ""
        if current:
            groups.append(current)
        return self.rng.choice(tuple(self.NUMBER_SEPARATORS)).join(groups)

    def _obfuscate_amount(self, value: str) -> str:
        chars = []
        for ch in value:
            if ch.isdigit() and ch in self.DIGIT_VARIANTS and self.rng.random() < 0.55:
                chars.append(self.rng.choice(tuple(self.DIGIT_VARIANTS[ch])))
            elif ch in self.AMOUNT_UNIT_VARIANTS:
                chars.append(self.rng.choice(tuple(self.AMOUNT_UNIT_VARIANTS[ch])))
            else:
                chars.append(ch)
        if len(chars) > 2 and self.rng.random() < 0.7:
            idx = self.rng.randrange(1, len(chars))
            chars.insert(idx, self.rng.choice(tuple(self.AMOUNT_INSERT_SYMBOLS)))
        return "".join(chars)

    def _extract_contact_hint(self, text: str) -> str:
        for pattern, spec in self.CONTACT_PATTERNS:
            choices = spec.get("hint_replacements")
            if choices and pattern.search(text):
                return self.rng.choice(tuple(choices))
        match = self.CONTACT_SPLIT_NUMBER_PATTERN.search(text)
        if match:
            return self._group_number(match.group(0))
        return self.rng.choice(tuple(self.DEFAULT_CONTACT_HINTS))

    def _contact_action_hint(self, contact: str) -> str:
        return self.CONTACT_ACTION_HINTS.get(contact, contact)

    def _extract_context_hint(self, text: str) -> str:
        candidates: List[str] = []
        for key, hint in self.PLACEHOLDER_CONTEXT_HINTS:
            if key in text:
                candidates.append(hint)

        amount = self.AMOUNT_PATTERN.search(text)
        if amount:
            candidates.append(f"{self._obfuscate_amount(amount.group(0))}档")

        for key, replacement in self.BENEFIT_HINTS:
            if key in text:
                candidates.append(replacement)
                if len(candidates) >= 5:
                    break

        if candidates:
            return self.rng.choice(tuple(candidates))
        return self.rng.choice(tuple(self.DEFAULT_CONTEXT_HINTS))

    def _extract_benefit_hint(self, text: str) -> str:
        for key, replacement in self.BENEFIT_HINTS:
            if key in text:
                return replacement
        return self.rng.choice(tuple(self.DEFAULT_BENEFITS))


def attack_type_summary(results: Iterable[AttackResult]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for item in results:
        summary[item.attack_type] = summary.get(item.attack_type, 0) + 1
    return summary
