"""장치명 → 모델 ID 매칭(설계서 §7-1).

원칙
- 정확 일치·별칭 일치만 자동 확정한다(score 1.0).
- 유사도 결과는 후보로만 내고 확정하지 않는다(점수 상한 candidate_score_cap < 1.0, 사람 확인).
- 판별 표시(Ti·SUPER·X3D·K·F ...)가 다르면 다른 모델이다.
- 숫자 토큰은 대칭 비교: 양쪽 모두 상대에게 없는 숫자를 가지면 다른 모델, 한쪽에만 더 있으면 후보.
- 토큰은 다중집합(반복 토큰이 하나로 합쳐져 1.0이 되는 일을 막는다).
- 한 이름에 모델 키가 둘 이상(겹쳐 포함이 아닌) 보이면 매칭하지 않는다(MULTI_MODEL).
- 노트북·모바일 표기는 데스크톱 모델에 매칭하지 않는다(MOBILE).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class MatchStatus(Enum):
    CONFIRMED = "CONFIRMED"   # 정확·별칭 일치(자동 확정)
    CANDIDATE = "CANDIDATE"   # 후보만(사람 확인 필요)
    NO_MATCH = "NO_MATCH"     # 후보 없음
    EXCLUDED = "EXCLUDED"     # 규칙상 매칭 대상 아님(reason: MOBILE, MULTI_MODEL)


@dataclass(frozen=True)
class Candidate:
    model_id: str
    score: float


@dataclass(frozen=True)
class MatchResult:
    status: MatchStatus
    model_id: str | None = None
    score: float = 0.0
    reason: str | None = None
    candidates: list[Candidate] = field(default_factory=list)


_SEPARATORS = re.compile(r"[\s\-_/+,;:()\[\]|&]+")
_MODEL_NUMBER = re.compile(r"^\d{3,}$")  # 모델 번호(4070, 7800). 등급 숫자(7·9)·단위(16gb)·i7은 제외


def _glued_pattern(suffixes: Iterable[str]) -> re.Pattern[str]:
    ordered = sorted(set(suffixes), key=len, reverse=True)  # 긴 접미사부터(x3d가 x보다 먼저)
    return re.compile(r"^(\d+)(" + "|".join(map(re.escape, ordered)) + r")$")


def _clean(name: str, cfg: dict[str, Any]) -> str:
    text = unicodedata.normalize("NFKC", name).lower()
    for pattern in cfg["noise_patterns"]:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def tokens(name: str, cfg: dict[str, Any]) -> list[str]:
    """정규화 토큰 목록(순서 유지). 브랜드 토큰은 빼고, 숫자에 붙은 판별 접미사는 분리한다."""
    glued = _glued_pattern(cfg["glued_suffixes"])
    brands = set(cfg["brand_tokens"])
    out: list[str] = []
    for raw in _SEPARATORS.split(_clean(name, cfg)):
        if not raw or raw in brands:
            continue
        m = glued.match(raw)
        out.extend([m.group(1), m.group(2)] if m else [raw])
    return out


def normalize(name: str, cfg: dict[str, Any]) -> str:
    return " ".join(tokens(name, cfg))


def is_mobile(name: str, cfg: dict[str, Any]) -> bool:
    text = _clean(name, cfg)
    for word in cfg["mobile_words"]:
        if re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text):
            return True
    mobile = _glued_pattern(cfg["mobile_glued_suffixes"])
    return any(mobile.match(t) for t in _SEPARATORS.split(text) if t)


@dataclass(frozen=True)
class _Key:
    model_id: str
    text: str
    bag: Counter
    words: frozenset[str]
    numbers: Counter


def _make_key(model_id: str, name: str, cfg: dict[str, Any]) -> _Key:
    toks = tokens(name, cfg)
    words = set(cfg["distinguishing_words"])
    return _Key(
        model_id=model_id,
        text=" ".join(toks),
        bag=Counter(toks),
        words=frozenset(t for t in toks if t in words),
        numbers=Counter(t for t in toks if t not in words and any(c.isdigit() for c in t)),
    )


class Catalog:
    """모델 ID별 표시명·별칭 키 모음."""

    def __init__(self, keys: Sequence[_Key]):
        texts = Counter(k.text for k in keys)
        clashes = {t for t, n in texts.items() if n > 1 and len({k.model_id for k in keys if k.text == t}) > 1}
        if clashes:
            raise ValueError(f"서로 다른 모델이 같은 정규화 이름을 가짐: {sorted(clashes)}")
        self.keys = list(keys)
        self.by_text = {k.text: k for k in keys}

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[str, str, Sequence[str]]], cfg: dict[str, Any]) -> "Catalog":
        keys = []
        for model_id, display_name, aliases in rows:
            for name in [display_name, *aliases]:
                keys.append(_make_key(model_id, name, cfg))
        return cls(keys)


def _multiset_jaccard(a: Counter, b: Counter) -> float:
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 0.0


def _contains(bag: Counter, key: _Key) -> bool:
    return all(bag[t] >= n for t, n in key.bag.items())


class NameMatcher:
    def __init__(self, catalog: Catalog, cfg: dict[str, Any]):
        self.catalog = catalog
        self.cfg = cfg

    def match(self, device_name: str) -> MatchResult:
        if is_mobile(device_name, self.cfg):
            return MatchResult(MatchStatus.EXCLUDED, reason="MOBILE")
        probe = _make_key("", device_name, self.cfg)
        exact = self.catalog.by_text.get(probe.text)
        if exact is not None:
            return MatchResult(MatchStatus.CONFIRMED, exact.model_id, 1.0, "EXACT")
        if self._has_multiple_models(probe.bag) or self._has_multiple_model_numbers(probe.bag):
            return MatchResult(MatchStatus.EXCLUDED, reason="MULTI_MODEL")
        candidates = self._candidates(probe)
        if not candidates:
            return MatchResult(MatchStatus.NO_MATCH, reason="NO_CANDIDATE")
        return MatchResult(MatchStatus.CANDIDATE, reason="SIMILARITY", candidates=candidates)

    def _has_multiple_models(self, bag: Counter) -> bool:
        """가장 긴 포함 키를 빼고도 다른 키가 남으면 모델이 둘 이상(4070 ⊂ 4070 Ti 같은 겹침은 제외)."""
        contained = sorted(
            (k for k in self.catalog.keys if _contains(bag, k)),
            key=lambda k: sum(k.bag.values()),
            reverse=True,
        )
        if not contained:
            return False
        rest = bag - contained[0].bag
        return any(_contains(rest, k) for k in self.catalog.keys)

    @staticmethod
    def _has_multiple_model_numbers(bag: Counter) -> bool:
        """'RTX 4070/4080'처럼 두 번째 모델을 번호만 적은 줄임 표기도 다중 모델로 본다."""
        return len({t for t in bag if _MODEL_NUMBER.match(t)}) >= 2

    def _admits(self, probe: _Key, key: _Key) -> bool:
        if probe.words != key.words:
            return False
        probe_only = probe.numbers - key.numbers
        key_only = key.numbers - probe.numbers
        return not (probe_only and key_only)  # 양쪽 모두 고유 숫자 → 다른 모델

    def _candidates(self, probe: _Key) -> list[Candidate]:
        cap = float(self.cfg["candidate_score_cap"])
        floor = float(self.cfg["candidate_min_score"])
        best: dict[str, float] = {}
        for key in self.catalog.keys:
            if not self._admits(probe, key):
                continue
            score = min(_multiset_jaccard(probe.bag, key.bag), cap)
            if score >= floor and score > best.get(key.model_id, -1.0):
                best[key.model_id] = score
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        return [Candidate(m, round(s, 4)) for m, s in ranked[: int(self.cfg["top_k"])]]
