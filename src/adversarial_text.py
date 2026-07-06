"""Text-level adversarial sample generation for Chinese spam detection.

The generators in this module are intentionally rule based. They model common
spam evasion patterns found in Chinese short messages: glyph/phonetic variants,
spacing, symbol insertion, traditional characters, contact obfuscation,
digit-letter confusion, pinyin abbreviation, keyword reordering, and stronger
multi-strategy rewrites. Each generated sample keeps metadata so experiments can
report performance by attack type instead of only overall accuracy.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import jieba
except ImportError:  # pragma: no cover - handled at runtime for minimal envs
    jieba = None


AttackType = str


@dataclass
class AttackResult:
    """A generated adversarial text and the operations used to create it."""

    original: str
    adversarial: str
    attack_type: AttackType
    operations: List[str] = field(default_factory=list)


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
    words = [w.strip() for w in jieba.cut(text) if w.strip()]
    return " ".join(words)


class ChineseSpamTextAttacker:
    """Generate realistic Chinese spam-text adversarial variants."""

    # Phrase-level replacements cover high-value spam keywords and contact terms.
    PHRASE_VARIANTS: Dict[str, Sequence[str]] = {
        "微信": ("微 信", "薇信", "威信", "维信", "胃星", "卫星", "V信", "vx"),
        "加微信": ("加薇信", "加V信", "加vx", "加 微 信", "家维信"),
        "加我": ("家我", "佳我", "加 我", "jia我"),
        "QQ": ("扣扣", "Q Q", "企鹅号", "q号"),
        "qq": ("扣扣", "q q", "企鹅号", "q号"),
        "电话": ("电 话", "垫话", "tel", "dian话"),
        "中奖": ("中 奖", "中獎", "仲奖", "zhong奖"),
        "免费": ("免 费", "免沸", "0元", "mian费"),
        "领取": ("领 取", "領取", "ling取", "领渠"),
        "红包": ("红 包", "紅包", "hong包"),
        "优惠": ("优 惠", "優惠", "you惠"),
        "充值": ("充 值", "沖值", "chong值"),
        "贷款": ("贷 款", "貸款", "代款", "dai款"),
        "兼职": ("兼 职", "兼職", "蒹职", "jian职"),
        "赚钱": ("赚 钱", "賺钱", "zhuan钱", "掙钱"),
        "博彩": ("博 彩", "搏彩", "bo彩"),
        "发票": ("发 票", "發票", "fa票"),
        "链接": ("链 接", "連结", "lian接"),
        "点击": ("点 击", "點击", "dian击"),
        "客服": ("客 服", "ke服", "喀服"),
        "活动": ("活 动", "活動", "huo动"),
        "提现": ("提 现", "提现", "ti现"),
        "返现": ("返 现", "返現", "fan现"),
    }

    STRONG_PHRASE_VARIANTS: Dict[str, Sequence[str]] = {
        "微信": ("w信", "w x", "v/x", "微丨信", "薇 x", "wx"),
        "加微信": ("+v/x", "加 w x", "私我wx", "走薇x", "找我看资料"),
        "加我": ("私我", "找我", "戳我", "看头像"),
        "QQ": ("Q/ Q", "扣 扣", "q鹅", "q/号"),
        "电话": ("电丨话", "tel.", "来电", "号 码"),
        "中奖": ("中jiang", "中·奖", "有喜", "抽中名额"),
        "免费": ("0门槛", "免米", "无偿", "0元拿"),
        "领取": ("点开拿", "去拿", "领·取", "lq"),
        "红包": ("红苞", "福袋", "福利包", "h b"),
        "优惠": ("优会", "折扣码", "you惠", "券包"),
        "充值": ("冲值", "充 值", "chong值", "到账"),
        "贷款": ("放款", "周转金", "dai款", "额度"),
        "兼职": ("兼zhi", "副业", "日结活", "线上活"),
        "赚钱": ("zhuan米", "进账", "收益", "搞钱"),
        "发票": ("票据", "fa票", "开飘", "专票"),
        "链接": ("入口", "地止", "链·接", "lian接"),
        "点击": ("戳开", "点一下", "dian开", "点·击"),
        "客服": ("k服", "专员", "小助手", "客fu"),
        "提现": ("ti现", "到卡", "秒到", "提·现"),
        "返现": ("fan现", "返钱", "返米", "回款"),
    }

    PINYIN_ABBREVIATIONS: Dict[str, Sequence[str]] = {
        "微信": ("wx", "vx", "w信"),
        "客服": ("kf", "k服"),
        "免费": ("mf", "0费"),
        "领取": ("lq", "领q"),
        "红包": ("hb", "红b"),
        "点击": ("dj", "点j"),
        "链接": ("lj", "链j"),
        "电话": ("dh", "tel"),
        "充值": ("cz", "充z"),
        "贷款": ("dk", "贷k"),
        "中奖": ("zj", "中j"),
        "发票": ("fp", "发p"),
        "优惠": ("yh", "优h"),
        "提现": ("tx", "提x"),
        "返现": ("fx", "返x"),
        "赚钱": ("zq", "赚q"),
        "兼职": ("jz", "兼z"),
    }

    CHAR_VARIANTS: Dict[str, Sequence[str]] = {
        "微": ("薇", "威", "维", "胃", "卫"),
        "信": ("心", "欣", "新", "伩", "星"),
        "奖": ("獎", "浆", "奨"),
        "领": ("領", "令"),
        "取": ("渠", "娶"),
        "赚": ("賺", "攥", "转"),
        "钱": ("銭", "前"),
        "贷": ("貸", "代"),
        "款": ("欵",),
        "充": ("沖", "冲"),
        "值": ("直", "植"),
        "票": ("漂", "飘"),
        "击": ("擊", "机"),
        "客": ("喀",),
        "服": ("俯",),
        "万": ("萬",),
        "轻": ("輕",),
        "松": ("鬆",),
        "家": ("佳",),
        "号": ("號",),
    }

    TRADITIONAL_VARIANTS: Dict[str, str] = {
        "万": "萬",
        "与": "與",
        "业": "業",
        "东": "東",
        "乐": "樂",
        "买": "買",
        "云": "雲",
        "优": "優",
        "传": "傳",
        "体": "體",
        "价": "價",
        "会": "會",
        "伪": "偽",
        "贷": "貸",
        "赚": "賺",
        "轻": "輕",
        "过": "過",
        "远": "遠",
        "选": "選",
        "连": "連",
        "话": "話",
        "费": "費",
        "账": "賬",
        "贴": "貼",
        "请": "請",
        "奖": "獎",
        "击": "擊",
        "联": "聯",
        "系": "係",
        "现": "現",
        "发": "發",
        "领": "領",
        "台": "臺",
    }

    DIGIT_VARIANTS: Dict[str, Sequence[str]] = {
        "0": ("O", "o", "零"),
        "1": ("l", "I", "壹"),
        "2": ("Z", "贰"),
        "3": ("E", "叁"),
        "4": ("A", "肆"),
        "5": ("S", "伍"),
        "6": ("G", "陆"),
        "7": ("T", "柒"),
        "8": ("B", "捌"),
        "9": ("g", "玖"),
    }

    CONTACT_PATTERNS: Sequence[Tuple[re.Pattern, str]] = (
        (re.compile(r"(?i)(vx|v信|微信|微 信|薇信|维信)"), "contact_wechat"),
        (re.compile(r"(?i)(qq|扣扣|q号|Q Q)"), "contact_qq"),
        (re.compile(r"\d{5,}"), "contact_number"),
    )

    INSERT_SYMBOLS: Sequence[str] = (" ", "-", "_", ".", "*")
    STRONG_INSERT_SYMBOLS: Sequence[str] = (" ", "-", "_", ".", "*", "·", "/", "丨", "+", "~")

    def __init__(
        self,
        seed: int = 42,
        max_char_replacements: int = 2,
        max_symbol_insertions: int = 2,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_char_replacements = max_char_replacements
        self.max_symbol_insertions = max_symbol_insertions

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
            priority = {
                "semantic_rewrite": 0,
                "strong_mixed": 1,
                "pinyin_abbreviation": 2,
                "multi_keyword_obfuscation": 3,
                "contact_split": 4,
                "amount_obfuscation": 5,
                "url_obfuscation": 6,
                "strong_phrase_variant": 7,
            }
            results.sort(key=lambda item: priority.get(item.attack_type, 20))
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
            return (
                "semantic_rewrite",
                "strong_mixed",
                "pinyin_abbreviation",
                "contact_split",
                "multi_keyword_obfuscation",
                "url_obfuscation",
                "amount_obfuscation",
                "strong_phrase_variant",
                *base,
            )
        if strength == "balanced":
            return (
                "pinyin_abbreviation",
                "contact_split",
                "multi_keyword_obfuscation",
                "strong_phrase_variant",
                *base,
            )
        return base

    def default_attack_types(self, label: str) -> Sequence[AttackType]:
        if label == "spam":
            return (
                "phrase_variant",
                "glyph_variant",
                "phonetic_variant",
                "symbol_insertion",
                "traditional_variant",
                "digit_letter_mix",
                "contact_obfuscation",
                "mixed",
            )
        return (
            "symbol_insertion",
            "traditional_variant",
            "digit_letter_mix",
        )

    def apply_attack(self, text: str, attack_type: AttackType) -> Optional[AttackResult]:
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
        attacked, operations = handler(text)
        if attacked == text:
            return None
        return AttackResult(text, attacked, attack_type, operations)

    def _phrase_variant(self, text: str) -> Tuple[str, List[str]]:
        keys = [key for key in self.PHRASE_VARIANTS if key in text]
        if not keys:
            return text, []
        key = self.rng.choice(keys)
        replacement = self.rng.choice(tuple(self.PHRASE_VARIANTS[key]))
        return text.replace(key, replacement, 1), [f"{key}->{replacement}"]

    def _strong_phrase_variant(self, text: str) -> Tuple[str, List[str]]:
        keys = [key for key in self.STRONG_PHRASE_VARIANTS if key in text]
        if not keys:
            return text, []
        key = self.rng.choice(keys)
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
        keywords = [
            "微信",
            "加微信",
            "领取",
            "红包",
            "免费",
            "点击",
            "链接",
            "客服",
            "电话",
            "贷款",
            "发票",
            "充值",
            "提现",
            "返现",
        ]
        keys = [key for key in keywords if key in text]
        if not keys:
            return text, []

        current = text
        operations: List[str] = []
        self.rng.shuffle(keys)
        for key in keys[:4]:
            if key not in current or len(key) < 2:
                continue
            symbol = self.rng.choice(tuple(self.STRONG_INSERT_SYMBOLS))
            replacement = symbol.join(list(key))
            if key == "加微信" and self.rng.random() < 0.5:
                replacement = self.rng.choice(("加 v/x", "加 w x", "+微丨信"))
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
        match = re.search(r"\d+(?:\.\d+)?\s*(?:元|块|万|折|%)?", text)
        if match:
            value = match.group(0)
            replacement = self._obfuscate_amount(value)
            return text[: match.start()] + replacement + text[match.end() :], [f"{value}->{replacement}"]

        fallback = []
        for key, replacement in (("免费", "0门槛"), ("红包", "福袋"), ("优惠", "券包"), ("提现", "到卡")):
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
        for pattern, name in self.CONTACT_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            matched = match.group(0)
            if name == "contact_wechat":
                replacement = self.rng.choice(("薇信", "V信", "v 信", "vx"))
            elif name == "contact_qq":
                replacement = self.rng.choice(("扣扣", "q q", "Q号"))
            else:
                replacement = self._obfuscate_number(matched)
            return (
                text[: match.start()] + replacement + text[match.end() :],
                [f"{matched}->{replacement}"],
            )
        return text, []

    def _contact_split(self, text: str) -> Tuple[str, List[str]]:
        replacements: Sequence[Tuple[re.Pattern, Sequence[str]]] = (
            (re.compile(r"(?i)(微信|微 信|薇信|维信|vx|v信|wx)"), ("w x", "v/x", "微丨信", "薇 x", "wx")),
            (re.compile(r"(?i)(QQ|qq|扣扣|q号|Q Q)"), ("Q/ Q", "扣 扣", "q鹅", "q/号")),
        )
        for pattern, choices in replacements:
            match = pattern.search(text)
            if not match:
                continue
            matched = match.group(0)
            replacement = self.rng.choice(tuple(choices))
            return text[: match.start()] + replacement + text[match.end() :], [f"{matched}->{replacement}"]

        match = re.search(r"\d{6,}", text)
        if match:
            matched = match.group(0)
            replacement = self._group_number(matched)
            return text[: match.start()] + replacement + text[match.end() :], [f"{matched}->{replacement}"]
        return text, []

    def _url_obfuscation(self, text: str) -> Tuple[str, List[str]]:
        url_match = re.search(r"(?i)(https?://|www\.)[a-z0-9./?=&_%:-]+", text)
        if url_match:
            value = url_match.group(0)
            replacement = value.replace("http://", "hxxp://").replace("https://", "hxxps://")
            replacement = replacement.replace("www.", "w w w点").replace(".", "点").replace("/", "/ ")
            return text[: url_match.start()] + replacement + text[url_match.end() :], [f"{value}->{replacement}"]

        for key, replacement in (("链接", "入口"), ("网址", "地止"), ("点击", "戳开"), ("登录", "登 录")):
            if key in text:
                return text.replace(key, replacement, 1), [f"{key}->{replacement}"]
        return text, []

    def _semantic_rewrite(self, text: str) -> Tuple[str, List[str]]:
        contact = self._extract_contact_hint(text)
        benefit = self._extract_benefit_hint(text)
        urgency = self.rng.choice(("过时失效", "名额不多", "今天截止", "系统保留一小时"))
        contact_action = self._contact_action_hint(contact)
        templates = (
            "{benefit}已开放，{contact_action}核对口令，{urgency}",
            "通知：{benefit}仍可领取，{contact_action}回复88，{urgency}",
            "不用排队，{contact_action}发口令，{benefit}秒到，{urgency}",
            "{benefit}入口已开，{contact_action}确认，错过不补",
        )
        template = self.rng.choice(templates)
        rewritten = template.format(benefit=benefit, contact_action=contact_action, urgency=urgency)
        return rewritten, [f"semantic_rewrite(benefit={benefit},contact={contact})"]

    def _mixed_attack(self, text: str) -> Tuple[str, List[str]]:
        operations: List[str] = []
        current = text
        for handler in (self._phrase_variant, self._digit_letter_mix, self._symbol_insertion):
            current, ops = handler(current)
            operations.extend(ops)
        return current, operations

    def _strong_mixed_attack(self, text: str) -> Tuple[str, List[str]]:
        operations: List[str] = []
        current = text
        handlers = (
            self._strong_phrase_variant,
            self._pinyin_abbreviation,
            self._amount_obfuscation,
            self._contact_split,
            self._multi_keyword_obfuscation,
        )
        for handler in handlers:
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
        chars.insert(idx, self.rng.choice(("-", " ", ".")))
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
        return self.rng.choice((" ", "-", "." )).join(groups)

    def _obfuscate_amount(self, value: str) -> str:
        chars = []
        for ch in value:
            if ch.isdigit() and ch in self.DIGIT_VARIANTS and self.rng.random() < 0.55:
                chars.append(self.rng.choice(tuple(self.DIGIT_VARIANTS[ch])))
            elif ch == "元":
                chars.append(self.rng.choice(("米", "圆", "rmb")))
            elif ch == "折":
                chars.append("zhe")
            else:
                chars.append(ch)
        if len(chars) > 2 and self.rng.random() < 0.7:
            idx = self.rng.randrange(1, len(chars))
            chars.insert(idx, self.rng.choice((".", "·", " ")))
        return "".join(chars)

    def _extract_contact_hint(self, text: str) -> str:
        if re.search(r"(?i)(微信|vx|wx|v信|薇信|微 信)", text):
            return self.rng.choice(("w x", "v/x", "微丨信", "wx"))
        if re.search(r"(?i)(qq|扣扣|q号)", text):
            return self.rng.choice(("Q/ Q", "扣 扣", "q鹅"))
        match = re.search(r"\d{6,}", text)
        if match:
            return self._group_number(match.group(0))
        return self.rng.choice(("私信", "头像", "小助手", "入口"))

    def _contact_action_hint(self, contact: str) -> str:
        if contact == "头像":
            return "看头像"
        if contact == "小助手":
            return "找小助手"
        if contact == "入口":
            return "从入口"
        return contact

    def _extract_benefit_hint(self, text: str) -> str:
        for key, replacement in (
            ("话费", "话费包"),
            ("红包", "福袋"),
            ("提现", "到卡名额"),
            ("贷款", "周转额度"),
            ("发票", "票据服务"),
            ("中奖", "抽中名额"),
            ("优惠", "券包"),
            ("充值", "到账福利"),
        ):
            if key in text:
                return replacement
        return self.rng.choice(("福利", "名额", "礼包", "补贴"))


def attack_type_summary(results: Iterable[AttackResult]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for item in results:
        summary[item.attack_type] = summary.get(item.attack_type, 0) + 1
    return summary
