"""위키 표 셀 문자열 → 사실값 파서. 못 읽으면 None(추정하지 않는다).

실제 셀 표본(2026-10-08 수집본)에서 확인한 형식만 다룬다: docs/sources.md
"""

from __future__ import annotations

import re
from datetime import date

_MISSING = re.compile(r"^(—|-|–|n/?a|none|tba|tbd|\?|unknown)?\s*(n/?a)?$", re.I)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _blank(s: str | None) -> bool:
    return s is None or bool(_MISSING.match(s.strip()))


def first_number(s: str | None) -> float | None:
    if _blank(s):
        return None
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s.replace("~", ""))
    return float(m.group(0).replace(",", "")) if m else None


def cores_threads(s: str | None) -> tuple[int | None, int | None]:
    """'16 (32)' → (16, 32). '8 (8)' → (8, 8)."""
    if _blank(s):
        return None, None
    m = re.match(r"\s*(\d+)\s*\(\s*(\d+)\s*\)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    n = first_number(s)
    return (int(n), None) if n is not None else (None, None)


def ghz_to_mhz(s: str | None) -> int | None:
    n = first_number(s)
    return int(round(n * 1000)) if n is not None and n < 20 else None


def boost_mhz(s: str | None) -> int | None:
    """GPU 클럭 셀: '1830 (2460)' → 2460, '1000 2040' → 2040, '2450' → 2450 (부스트 = 마지막 수)."""
    if _blank(s):
        return None
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s)]
    nums = [x for x in nums if x >= 100]  # MHz 값만
    return int(nums[-1]) if nums else None


def watts(s: str | None) -> int | None:
    n = first_number(s)
    return int(n) if n is not None else None


def usd(s: str | None) -> int | None:
    """'US $699', '$110', 'Jan 4, 2022 $199' → 699/110/199. 달러 표시가 없으면 None."""
    if _blank(s):
        return None
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", s)
    return int(round(float(m.group(1).replace(",", "")))) if m else None


def shader_count(s: str | None) -> int | None:
    """코어 구성 '2304:72:32: 18:72:' → 2304, '6 Xe cores 768:32:...' → 768 (콜론 앞 첫 수)."""
    if _blank(s):
        return None
    m = re.search(r"(\d[\d,]*)\s*:", s)
    return int(m.group(1).replace(",", "")) if m else None


def release(s: str | None) -> tuple[date | None, str | None]:
    """출시일과 정밀도(DAY/MONTH/QUARTER). 'Feb 28, 2023'·'April 22, 2026'·'April 2024'·'Aug, 2025'·'Q4 2021'."""
    if _blank(s):
        return None, None
    text = s.strip()
    # 'Apr 13, 2021 (OEM), Aug 5, 2021 (retail)': 리테일 데스크톱 범위이므로 (retail) 표시가 붙은 날짜를 쓴다
    retail = re.search(r"([A-Za-z]{3,9}\.?\s+\d{1,2},\s*\d{4}|[A-Za-z]{3,9}\.?,?\s+\d{4}|Q[1-4]\s*\d{4})\s*\(retail\)", text, re.I)
    if retail:
        text = retail.group(1)
    m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),\s*(\d{4})", text)
    if m and m.group(1)[:3].lower() in _MONTHS:
        return date(int(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2))), "DAY"
    m = re.search(r"\b([A-Za-z]{3,9})\.?,?\s+(\d{4})\b", text)
    if m and m.group(1)[:3].lower() in _MONTHS:
        return date(int(m.group(2)), _MONTHS[m.group(1)[:3].lower()], 1), "MONTH"
    m = re.search(r"\bQ([1-4])\s*(\d{4})\b", text, re.I)
    if m:
        return date(int(m.group(2)), 3 * int(m.group(1)) - 2, 1), "QUARTER"
    return None, None


def pcie(s: str | None) -> tuple[int | None, int | None]:
    """GPU 버스 인터페이스 → (PCIe 세대, 레인 수). 'PCIe 4.0 ×16'·'PCIe 4 x8'. 못 읽으면 (None, None)."""
    if _blank(s):
        return None, None
    m = re.search(r"PCIe\s*(\d)(?:\.\d)?\s*[x×]\s*(\d+)", s, re.I)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def size_mb(s: str | None) -> float | None:
    """'64 MB' → 64.0, '2.6 MB' → 2.6, '512 KB' → 0.5. 단위가 없으면 None(추정하지 않는다)."""
    if _blank(s):
        return None
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(KB|MB|KiB|MiB)\b", s, re.I)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return value / 1024 if m.group(2).lower().startswith("k") else value


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
