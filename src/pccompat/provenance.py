"""WP0: 흩어진 출처 기록을 공개용 field_provenance.csv 하나로 합친다(논문 기여점: 값마다 출처·등급).

합치는 것
- 위키: interim/provenance_wiki.csv (값 셀·행·열 각주 / 각주 없음)
- 캐시 관찰값: interim/provenance_cache.csv (linuxhw, OPEN_DATASET, CC BY 4.0)
- 세대 사실(소켓·메모리 세대·최대 메모리, 칩셋 계열의 소켓·메모리 옵션): manual/family_facts.csv
  사람이 공식 URL로 검증하면 OFFICIAL·verified=true, 아니면 OTHER(Claude 사전 입력, 미검증)
- 수동 출시일: manual/release_dates_todo.csv 에 채워진 행 → OFFICIAL·verified=true
- GPU 사양 정정: manual/gpu_spec_check.csv 에서 공식 값과 달랐던 값 → OFFICIAL·verified=true(gpu_specs)
공개본의 값은 data/out/parts.csv·chipsets.csv와 일치해야 한다(불일치는 issues로 보고).
"""

from __future__ import annotations

import pandas as pd

from pccompat import gpu_specs
from pccompat.clean import coerce_types
from pccompat.config import PROJECT_ROOT
from pccompat.schema import FIELD_PROVENANCE

INTERIM = PROJECT_ROOT / "data" / "interim"
MANUAL = PROJECT_ROOT / "data" / "manual"
OUT = PROJECT_ROOT / "data" / "out"
COLS = [c.name for c in FIELD_PROVENANCE.columns]


def _facts_rows(parts: pd.DataFrame, chips: pd.DataFrame) -> list[dict]:
    facts = pd.read_csv(MANUAL / "family_facts.csv", dtype=str, keep_default_na=False)
    rows = []
    for f in facts.itertuples():
        verified = f.verified.strip().lower() == "true" and f.source_url.strip() != ""
        tier = "OFFICIAL" if verified else "OTHER"
        if f.kind == "cpu":
            ids, fields = parts.loc[parts["family"] == f.family, "part_id"], {"socket": f.socket,
                                                                              "memory_gens": f.memory_gens,
                                                                              "max_memory_gb": f.max_memory_gb}
        else:
            ids, fields = chips.loc[chips["family"] == f.family, "chipset_id"], {"socket": f.socket,
                                                                                 "memory_gen_options": f.memory_gens}
        # 최대 메모리만 다른(비공식) 곳에서 확인했으면 그 필드만 따로 출처를 단다
        other_mem = getattr(f, "max_memory_source_url", "").strip()
        for eid in ids:
            for field, value in fields.items():
                if not value.strip():
                    continue
                separate = field == "max_memory_gb" and other_mem
                rows.append({"entity_id": eid, "field": field, "value": value,
                             "source_url": other_mem if separate else (f.source_url or "manual/family_facts.csv"),
                             "source_tier": "OTHER" if separate else tier,
                             "license_scope": "FACT_CITED", "verified": False if separate else verified,
                             "source_revision": "family_facts"})
    return rows


def _release_rows() -> list[dict]:
    todo = pd.read_csv(MANUAL / "release_dates_todo.csv", dtype=str, keep_default_na=False)
    done = todo[todo["release_date"].str.strip() != ""]
    rows = []
    for r in done.itertuples():
        # 시트에 등급을 적었으면 그 등급(예: 위키 목록 문서), 없으면 URL 있는 사람 입력 = 공식
        given = getattr(r, "source_tier", "").strip()
        tier = given or ("OFFICIAL" if r.source_url else "OTHER")
        rows.append({"entity_id": r.part_id, "field": "release_date", "value": r.release_date,
                     "source_url": r.source_url, "source_tier": tier, "license_scope": "FACT_CITED",
                     "verified": tier == "OFFICIAL", "source_revision": "manual"})
    return rows


def expand_to_variants(prov: pd.DataFrame, chips: pd.DataFrame) -> pd.DataFrame:
    """분할 전 칩셋 ID로 기록된 출처를 보드 메모리 변형 노드에 복제한다.

    변형의 memory_gen_options 값은 그 변형의 보드 메모리 한 종류다(출처가 말하는 "칩셋이 DDR4·DDR5를 지원"의 부분집합).
    """
    split = chips[chips["board_memory"].notna()] if "board_memory" in chips else chips.iloc[0:0]
    variants = split.groupby("base_chipset_id")["chipset_id"].apply(list).to_dict()
    out = prov.assign(entity_id=prov["entity_id"].map(lambda e: variants.get(e, [e]))).explode("entity_id")
    memory = dict(zip(split["chipset_id"], split["board_memory"]))
    is_mem = out["field"].eq("memory_gen_options") & out["entity_id"].isin(memory)
    out.loc[is_mem, "value"] = out.loc[is_mem, "entity_id"].map(memory)
    return out.reset_index(drop=True)


def build() -> tuple[pd.DataFrame, list]:
    parts = pd.read_csv(OUT / "parts.csv")
    chips = pd.read_csv(OUT / "chipsets.csv")
    frames = [pd.read_csv(INTERIM / "provenance_wiki.csv", dtype=str, keep_default_na=False)]
    fam_dates = INTERIM / "provenance_family_dates.csv"
    if fam_dates.exists() and fam_dates.stat().st_size > 1:
        frames.append(pd.read_csv(fam_dates, dtype=str, keep_default_na=False))
    cache = INTERIM / "provenance_cache.csv"
    if cache.exists() and cache.stat().st_size > 1:
        frames.append(pd.read_csv(cache, dtype=str, keep_default_na=False))
    gpu_fix = gpu_specs.provenance_rows(gpu_specs.corrections(gpu_specs.load()))
    frames.append(pd.DataFrame(_facts_rows(parts, chips) + _release_rows() + gpu_fix).astype(str))
    prov = expand_to_variants(pd.concat(frames, ignore_index=True), chips)
    released = set(parts["part_id"]) | set(chips["chipset_id"])
    prov = prov[prov["entity_id"].isin(released)]
    # 수동 입력이 위키보다 우선: 같은 (값 주체, 필드)는 마지막(수동) 행만 남긴다
    prov = prov.drop_duplicates(subset=["entity_id", "field"], keep="last")
    typed, issues = coerce_types(prov.reindex(columns=COLS), FIELD_PROVENANCE)
    return typed, issues


def main() -> None:
    prov, issues = build()
    prov.to_csv(OUT / "field_provenance.csv", index=False)
    print(f"field_provenance {len(prov):,}행 / 형식 문제 {len(issues)} / 등급 "
          f"{prov['source_tier'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
