"""linuxhw DMI(CC BY 4.0) → CPU 모델별 캐시 L1·L2·L3 관찰값(최빈값).

방식(사용자 결정: A 기본 + B 교차 확인 + C 사람 대조)
- A: Desktop 보고서마다 Processor Version을 모델 ID에 매칭(names.py 확정 매칭만),
  dmidecode Cache Information의 "Configuration: … Level N"으로 수준을 읽는다.
  L1·L2 = 같은 수준 블록 합(하이브리드 인텔은 P·E 두 세트 합), L3 = 최댓값(공유 캐시라 합하지 않음).
  크기 0 보고는 버린다. 모델별 (L1, L2, L3) 최빈값을 쓰고, 보고서가 min_reports 미만이면 채택하지 않는다.
- B: 위키 L3(또는 L2)가 있으면 관찰값과 대조해 일치 여부를 남긴다(교차 확인).
- C: 채택 못 한 모델은 data/manual/cache_todo.csv로 내보내 사람이 공식 페이지와 대조한다.
원 보고서(개인 PC 덤프)는 공개하지 않고 모델별 집계만 쓴다.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from pccompat.config import PROJECT_ROOT, load_config
from pccompat.names import Catalog, MatchStatus, NameMatcher

DMI = PROJECT_ROOT / "data" / "raw" / "linuxhw" / "DMI"
INTERIM = PROJECT_ROOT / "data" / "interim"
MIN_REPORTS = 3  # 설계값

_SIZE = re.compile(r"Installed Size:\s*([\d.]+)\s*(kB|KB|MB|GB)", re.I)
_LEVEL = re.compile(r"Level\s*(\d)")


def _kb(num: str, unit: str) -> float:
    return float(num) * {"kb": 1, "mb": 1024, "gb": 1024 * 1024}[unit.lower()]


def parse_report(text: str) -> tuple[str | None, tuple[float, float, float] | None]:
    """보고서 한 건 → (Processor Version, (L1 kB, L2 kB, L3 kB)). 캐시가 없거나 0이면 None."""
    version = None
    levels: dict[int, list[float]] = defaultdict(list)
    for block in text.split("\n\n"):
        if "Processor Information" in block and version is None:
            m = re.search(r"^\s*Version:\s*(.+)$", block, re.M)
            if m:
                version = m.group(1).strip()
        elif "Cache Information" in block:
            size, level = _SIZE.search(block), _LEVEL.search(block)
            if size and level:
                levels[int(level.group(1))].append(_kb(*size.groups()))
    if not all(levels.get(i) for i in (1, 2, 3)):
        return version, None
    l1, l2, l3 = sum(levels[1]), sum(levels[2]), max(levels[3])
    if min(l1, l2, l3) <= 0:
        return version, None
    return version, (l1, l2, l3)


def _reports(root: Path) -> Iterator[str]:
    for dirpath, _, files in os.walk(root / "Desktop"):
        for f in files:
            try:
                yield (Path(dirpath) / f).read_text(errors="ignore")
            except OSError:
                continue


def aggregate(parts: pd.DataFrame, cfg: dict[str, Any], root: Path = DMI,
              min_reports: int = MIN_REPORTS) -> tuple[pd.DataFrame, Counter]:
    cpus = parts[parts["category"] == "CPU"]
    matcher = NameMatcher(Catalog.from_rows(((r.part_id, r.model_name, []) for r in cpus.itertuples()),
                                            cfg["matching"]), cfg["matching"])
    seen: dict[str, Any] = {}
    obs: dict[str, Counter] = defaultdict(Counter)
    stats: Counter = Counter()
    for text in _reports(root):
        version, cache = parse_report(text)
        if version is None:
            stats["no_cpu_version"] += 1
            continue
        if version not in seen:
            r = matcher.match(version)
            seen[version] = r.model_id if r.status is MatchStatus.CONFIRMED else None
        pid = seen[version]
        if pid is None:
            stats["cpu_not_in_catalog"] += 1
            continue
        if cache is None:
            stats["no_or_zero_cache"] += 1
            continue
        obs[pid][cache] += 1
        stats["used"] += 1
    rows = []
    for pid, counter in obs.items():
        (l1, l2, l3), k = counter.most_common(1)[0]
        n = sum(counter.values())
        rows.append({"part_id": pid, "l1_cache_kb": l1, "l2_cache_mb": round(l2 / 1024, 3),
                     "l3_cache_mb": round(l3 / 1024, 3), "n_reports": n, "mode_share": round(k / n, 3),
                     "n_variants": len(counter), "adopted": n >= min_reports})
    return pd.DataFrame(rows), stats


def cross_check(observed: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    """B: 위키 값이 있는 필드와 관찰 최빈값 대조(상대 오차 1% 이내 = 일치)."""
    merged = observed.merge(parts[["part_id", "model_name", "l2_cache_mb", "l3_cache_mb"]], on="part_id",
                            suffixes=("_obs", "_wiki"))
    out = []
    for f in ("l2_cache_mb", "l3_cache_mb"):
        sub = merged.dropna(subset=[f"{f}_wiki"])
        for r in sub.itertuples():
            a, b = getattr(r, f"{f}_obs"), getattr(r, f"{f}_wiki")
            out.append({"part_id": r.part_id, "field": f, "observed": a, "wikipedia": float(b),
                        "agree": abs(a - float(b)) <= 0.01 * max(a, float(b)), "n_reports": r.n_reports})
    return pd.DataFrame(out)


def provenance_rows(observed: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    """채택된 관찰값 중 위키에 없는 필드만 출처 행으로 만든다(위키 값이 있으면 위키가 1차, 관찰은 교차 확인용)."""
    wiki = parts.set_index("part_id")
    rows = []
    for r in observed[observed["adopted"]].itertuples():
        for f in ("l1_cache_kb", "l2_cache_mb", "l3_cache_mb"):
            if pd.notna(wiki.at[r.part_id, f]):
                continue
            rows.append({"entity_id": r.part_id, "field": f, "value": getattr(r, f), "source_url": r.source_url,
                         "source_tier": "OPEN_DATASET", "source_revision": r.source_url.rsplit("/", 1)[-1],
                         "license_scope": "CC_BY_4_0", "verified": "false",
                         "note": f"linuxhw 관찰 최빈값 n={r.n_reports} share={r.mode_share}"})
    return pd.DataFrame(rows)


def source_commit(root: Path = DMI) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    cfg = load_config()
    parts = pd.read_csv(INTERIM / "parts_wiki.csv")
    observed, stats = aggregate(parts, cfg)
    commit = source_commit()
    observed["source_url"] = f"https://github.com/linuxhw/DMI/tree/{commit}"
    observed.to_csv(INTERIM / "cache_linuxhw.csv", index=False)
    provenance_rows(observed, parts).to_csv(INTERIM / "provenance_cache.csv", index=False)
    check = cross_check(observed[observed["adopted"]], parts)
    check.to_csv(INTERIM / "cache_crosscheck.csv", index=False)
    adopted = set(observed.loc[observed["adopted"], "part_id"])
    todo = parts[(parts["category"] == "CPU") & ~parts["part_id"].isin(adopted)][["part_id", "model_name", "family"]]
    for c in ["l1_cache_kb", "l2_cache_mb", "l3_cache_mb", "official_url", "note"]:
        todo[c] = ""
    todo.to_csv(PROJECT_ROOT / "data" / "manual" / "cache_todo.csv", index=False)
    print("보고서 처리:", dict(stats))
    print(f"관찰 모델 {len(observed)} / 채택(n≥{MIN_REPORTS}) {len(adopted)} / 사람 대조 필요 {len(todo)}")
    if not check.empty:
        print("교차 확인 일치율:", check.groupby("field")["agree"].mean().round(3).to_dict(),
              "불일치:", check.loc[~check["agree"], ["part_id", "field", "observed", "wikipedia"]].to_dict("records"))


if __name__ == "__main__":
    main()
