"""Blender Open Data 스냅샷(사람이 받은 zip, CC0) → 실측 표본 → 모델별 중앙값.

1단계 extract: zip 안 jsonl을 스트리밍해 v4 레코드(samples_per_minute가 있는 형식)만
  (device_name, device_type, scene, blender_version, samples_per_minute) 행으로 뽑는다.
  - 한 제출에 서로 다른 장치가 둘 이상이면(멀티 GPU 등) 어느 장치 점수인지 알 수 없어 제외하고 센다.
2단계 aggregate: 장치명을 모델 ID로 매칭(names.py §7-1 규칙, CONFIRMED만 사용) →
  (모델, 장치 종류, 장면, Blender 버전)별 IQR 절단 → 중앙값·표본 수·제거 수.
  - 버전·장면이 다르면 점수를 섞지 않는다(벤치마크 조건이 다르다).
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from pccompat.clean import trim_samples_iqr
from pccompat.config import PROJECT_ROOT, load_config
from pccompat.names import Catalog, MatchStatus, NameMatcher

ZIP = PROJECT_ROOT / "data" / "raw" / "blender" / "opendata-latest.zip"
INTERIM = PROJECT_ROOT / "data" / "interim"


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".jsonl"))
        with z.open(name) as f:
            for line in f:
                yield json.loads(line)


def extract_samples(path: Path = ZIP) -> tuple[pd.DataFrame, Counter]:
    rows: list[tuple] = []
    skipped: Counter = Counter()
    for rec in _records(path):
        if rec.get("schema_version") != "v4":
            skipped[f"schema_{rec.get('schema_version')}"] += 1
            continue
        for item in rec.get("data") or []:
            stats = item.get("stats") or {}
            spm = stats.get("samples_per_minute")
            info = item.get("device_info") or {}
            names = {d.get("name") for d in info.get("compute_devices") or [] if isinstance(d, dict)}
            if spm is None:
                skipped["no_samples_per_minute"] += 1
                continue
            if len(names) != 1:
                skipped["multi_or_no_device"] += 1
                continue
            rows.append((
                names.pop(), info.get("device_type"), (item.get("scene") or {}).get("label"),
                (item.get("blender_version") or {}).get("version"), float(spm),
            ))
    df = pd.DataFrame(rows, columns=["device_name", "device_type", "scene", "blender_version", "samples_per_minute"])
    return df, skipped


def aggregate(samples: pd.DataFrame, parts: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (perf_measurements 행, 장치명 매칭 결과표). 매칭은 고유 장치명마다 한 번만 한다."""
    catalog = Catalog.from_rows(((r.part_id, r.model_name, []) for r in parts.itertuples()), cfg["matching"])
    matcher = NameMatcher(catalog, cfg["matching"])
    names = samples["device_name"].drop_duplicates()
    results = {n: matcher.match(n) for n in names}
    match_table = pd.DataFrame(
        [{"device_name": n, "status": r.status.value, "reason": r.reason, "model_id": r.model_id,
          "candidates": ";".join(f"{c.model_id}:{c.score}" for c in r.candidates)} for n, r in results.items()]
    )
    confirmed = {n: r.model_id for n, r in results.items() if r.status is MatchStatus.CONFIRMED}
    s = samples[samples["device_name"].isin(confirmed)].copy()
    s["part_id"] = s["device_name"].map(confirmed)
    s["group"] = s[["part_id", "device_type", "scene", "blender_version"]].astype(str).agg("|".join, axis=1)
    m = cfg["measurements"]
    kept, report = trim_samples_iqr(s, "group", "samples_per_minute", k=m["iqr_k"], min_n=m["low_n_threshold"])
    stats = kept.groupby("group")["samples_per_minute"].agg(
        median="median", n="size", iqr=lambda x: x.quantile(0.75) - x.quantile(0.25))
    names_by_group = s.groupby("group")["device_name"].agg(lambda x: ";".join(sorted(set(x))))
    out = stats.join(report.set_index("group")[["n_removed"]]).join(names_by_group).reset_index()
    out[["part_id", "device_type", "scene", "benchmark_version"]] = out["group"].str.split("|", expand=True)
    out = out.rename(columns={"median": "median_samples_per_min", "n": "n_samples", "n_removed": "n_removed_outliers",
                              "device_name": "matched_device_names"})
    out["source"] = "BLENDER_OPEN_DATA"
    out["low_n_flag"] = out["n_samples"] < m["low_n_threshold"]
    out["match_method"] = "EXACT"
    cols = ["part_id", "source", "benchmark_version", "device_type", "scene", "median_samples_per_min", "n_samples",
            "n_removed_outliers", "iqr", "low_n_flag", "matched_device_names", "match_method"]
    return out[cols], match_table


def main() -> None:
    cfg = load_config()
    INTERIM.mkdir(parents=True, exist_ok=True)
    cache = INTERIM / "blender_samples.parquet"
    if cache.exists():
        samples = pd.read_parquet(cache)
    else:
        samples, skipped = extract_samples()
        samples.to_parquet(cache, index=False)
        print("제외:", dict(skipped))
    parts = pd.read_csv(INTERIM / "parts_wiki.csv", dtype=str)
    perf, matches = aggregate(samples, parts, cfg)
    perf.to_csv(INTERIM / "perf_blender.csv", index=False)
    matches.to_csv(INTERIM / "blender_name_matches.csv", index=False)
    print(f"표본 {len(samples):,} / 고유 장치명 {len(matches):,} / 매칭 상태 {matches['status'].value_counts().to_dict()}")
    print(f"측정 행 {len(perf):,} / 매칭된 모델 {perf['part_id'].nunique()} / 전체 모델 {len(parts)}")


if __name__ == "__main__":
    main()
