"""2일차 1단계: 노드 표 조립 → data/out/{parts,chipsets,ram_specs,cpu_chipset_support,field_provenance}.csv

- parts: 위키 정규화본 + 세대 사실(소켓·메모리 세대·최대 메모리, family_facts.csv) + 캐시 관찰값(위키 빈칸만)
  + 수동 출시일(release_dates_todo.csv에 사람이 채운 행) → 정제 파이프라인(clean.py)으로 타입·결측·범위·단종일 처리
- chipsets: 위키 칩셋 + 계열 사실(소켓·메모리 세대 옵션)
- ram_specs: 설정의 DDR 세대 × 총용량(정의값, 수집 아님)
- cpu_chipset_support: 위키 칩셋 표의 마이크로아키텍처별 지원 칸(Yes/No/Varies/Partial) → CPU×칩셋으로 전개.
  표에 칸이 없는 소켓(LGA1700·LGA1851)은 NOT_LISTED. 사람이 cpu_chipset_support.csv에 적은 행이 우선한다.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import yaml

from pccompat.clean import CleaningPipeline, assign_eol
from pccompat.config import PROJECT_ROOT, load_config
from pccompat.schema import CHIPSETS, PARTS, LIST_SEPARATOR
from pccompat.wikitable import parse_tables, reference_urls

INTERIM = PROJECT_ROOT / "data" / "interim"
MANUAL = PROJECT_ROOT / "data" / "manual"
OUT = PROJECT_ROOT / "data" / "out"
RAW_WIKI = PROJECT_ROOT / "data" / "raw" / "wikipedia"

_SUPPORT = {"yes": "SUPPORTED", "no": "UNSUPPORTED", "varies": "VARIES", "partial": "VARIES"}


def _support_value(text: str) -> str | None:
    t = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return _SUPPORT.get(t.split(" ")[0]) if t else None


def wiki_support_matrix() -> pd.DataFrame:
    """(uarch, chipset name) → 지원 값. 출처는 위키 리비전 URL."""
    rows = []
    am4 = json.loads((RAW_WIKI / "amd-am4-chipsets.json").read_text())
    t = parse_tables(am4["html"])[0]
    for i, h in enumerate(t.headers):
        m = re.match(r"CPU support / (Zen 3)$", h)
        if m:
            for r in t.rows:
                rows.append({"uarch": m.group(1), "chipset": r[0].text, "support_text": r[i].text, "url": am4["url"]})
    am5 = json.loads((RAW_WIKI / "amd-am5-chipsets.json").read_text())
    t = parse_tables(am5["html"])[0]
    for r in t.rows:
        if len(r) > 2 and r[1].text == "CPU support" and r[2].text in ("Zen 4", "Zen 5"):
            for i, h in enumerate(t.headers):
                if i >= 3 and h:
                    for name in [n.strip() for n in h.split("/") if n.strip()]:
                        rows.append({"uarch": r[2].text, "chipset": name, "support_text": r[i].text, "url": am5["url"]})
    lga = json.loads((RAW_WIKI / "intel-lga1200-chipsets.json").read_text())
    t = parse_tables(lga["html"])[0]
    i = next(i for i, h in enumerate(t.headers) if "Rocket Lake support" in h)
    for r in t.rows:
        rows.append({"uarch": "Rocket Lake", "chipset": r[0].text, "support_text": r[i].text, "url": lga["url"]})
    df = pd.DataFrame(rows)
    df["support"] = df["support_text"].map(_support_value)
    return df


def build_parts(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    parts = pd.read_csv(INTERIM / "parts_wiki.csv", dtype=str, keep_default_na=False)
    facts = pd.read_csv(MANUAL / "family_facts.csv", dtype=str, keep_default_na=False)
    cpu_facts = facts[facts["kind"] == "cpu"].set_index("family")
    for col in ("socket", "memory_gens", "max_memory_gb"):
        parts[col] = parts.apply(lambda r: cpu_facts.at[r["family"], col] if r["category"] == "CPU"
                                 and r["family"] in cpu_facts.index and not r.get(col) else r.get(col, ""), axis=1)
    cache = pd.read_csv(INTERIM / "cache_linuxhw.csv")
    cache = cache[cache["adopted"]].set_index("part_id")
    for col in ("l1_cache_kb", "l2_cache_mb"):
        parts[col] = parts.apply(lambda r: str(cache.at[r["part_id"], col])
                                 if r["part_id"] in cache.index and not r[col] else r[col], axis=1)
    todo = pd.read_csv(MANUAL / "release_dates_todo.csv", dtype=str, keep_default_na=False)
    filled = todo[todo["release_date"].str.strip() != ""].set_index("part_id")
    for pid, r in filled.iterrows():
        mask = parts["part_id"] == pid
        parts.loc[mask, "release_date"] = r["release_date"]
        parts.loc[mask, "release_date_precision"] = r["release_date_precision"] or "DAY"
        parts.loc[mask, "release_date_type"] = "LAUNCH"
    # 세대 출시일 대체(위키 문서 단위, 사용자 결정): 표에 날짜가 없고 사람 입력도 없는 행만
    wcfg = yaml.safe_load((PROJECT_ROOT / "config" / "wikipedia.yml").read_text(encoding="utf-8"))
    fallback_rows = []
    for fb in wcfg.get("family_release_fallback", []):
        mask = (parts["family"] == fb["family"]) & (parts["release_date"] == "")
        for pid in parts.loc[mask, "part_id"]:
            fallback_rows.append({"entity_id": pid, "field": "release_date", "value": str(fb["date"]),
                                  "source_url": fb["cited_url"], "source_tier": "WIKIPEDIA_FAMILY",
                                  "source_revision": str(fb["revid"]), "license_scope": "FACT_CITED",
                                  "verified": "false", "note": fb["note"]})
        parts.loc[mask, "release_date"] = str(fb["date"])
        parts.loc[mask, "release_date_precision"] = "FAMILY"
        parts.loc[mask, "release_date_type"] = "LAUNCH"
    pd.DataFrame(fallback_rows).to_csv(INTERIM / "provenance_family_dates.csv", index=False)
    window_start = str(cfg["window"]["start"])
    before = parts[(parts["release_date"] != "") & (parts["release_date"] < window_start)]
    parts = parts.drop(before.index)
    result = CleaningPipeline(cfg).run(parts.drop(columns=["release_date_precision"]), PARTS)
    frame = result.frame
    frame["release_date_precision"] = parts["release_date_precision"].replace("", pd.NA).values
    frame["proxy_perf"] = frame.apply(
        lambda r: (r["cores"] * r["boost_clock_mhz"] / 1000) if r["category"] == "CPU"
        and pd.notna(r["cores"]) and pd.notna(r["boost_clock_mhz"]) else r["fp32_tflops"], axis=1)
    return frame, result.reports["issues"], {"dropped_before_window": len(before)}


def excluded_chipsets(chips: pd.DataFrame, rules: dict[str, str] | list[dict[str, str]] | None) -> pd.Series:
    """범위 밖 칩셋 표시. 규칙은 {vendor, pattern} 하나 또는 그 목록(config.exclude_chipsets)."""
    drop = pd.Series(False, index=chips.index)
    for rule in ([rules] if isinstance(rules, dict) else rules or []):
        drop |= (chips["vendor"] == rule["vendor"]) & chips["name"].str.match(rule["pattern"])
    return drop


def build_chipsets(cfg: dict[str, Any]) -> pd.DataFrame:
    chips = pd.read_csv(INTERIM / "chipsets_wiki.csv", dtype=str, keep_default_na=False)
    rules = yaml.safe_load((PROJECT_ROOT / "config" / "wikipedia.yml").read_text(encoding="utf-8")).get("exclude_chipsets")
    chips = chips[~excluded_chipsets(chips, rules)].reset_index(drop=True)
    facts = pd.read_csv(MANUAL / "family_facts.csv", dtype=str, keep_default_na=False)
    fam = facts[facts["kind"] == "chipset"].set_index("family")
    chips["socket"] = chips["family"].map(fam["socket"])
    chips["memory_gen_options"] = chips["family"].map(fam["memory_gens"])
    result = CleaningPipeline(cfg).run(chips.drop(columns=["family", "release_date_precision"]), CHIPSETS)
    frame = result.frame
    frame["family"] = chips["family"].values
    # 칩셋 단종일: 다음 소켓 세대 칩셋의 첫 출시일(사용자 결정 (나)), config.lineage의 *_chipset 사슬
    return assign_eol(frame, cfg["lineage"])


def _gens(v: Any) -> tuple[str, ...]:
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return tuple(x for x in str(v or "").split(LIST_SEPARATOR) if x)


def split_board_memory(chips: pd.DataFrame, support: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DDR4·DDR5를 모두 지원하는 칩셋을 보드 메모리 종류별 노드로 나눈다(실제 보드는 한 종류만 단다).

    한 종류만 지원하는 칩셋은 그대로 두고 base_* 열만 자기 자신으로 채운다. CPU–칩셋 지원은 보드 메모리와 무관하므로
    두 변형에 복제한다. 바뀌는 판정은 칩셋–RAM뿐이다(변형의 memory_gen_options가 한 종류라 규칙이 PASS/FAIL을 낸다).
    """
    rows, variant_of = [], {}
    for r in chips.to_dict("records"):
        gens = _gens(r["memory_gen_options"])
        base = {**r, "base_chipset_id": r["chipset_id"], "base_name": r["name"], "board_memory": pd.NA}
        if len(gens) < 2:
            rows.append(base)
            continue
        for g in gens:
            vid = f"{r['chipset_id']}-{g.lower()}"
            rows.append({**base, "chipset_id": vid, "name": f"{r['name']} ({g})", "memory_gen_options": (g,),
                         "board_memory": g})
            variant_of.setdefault(r["chipset_id"], []).append(vid)
    sup = support.assign(chipset_id=support["chipset_id"].map(lambda c: variant_of.get(c, [c]))).explode("chipset_id")
    return pd.DataFrame(rows).reset_index(drop=True), sup.reset_index(drop=True)


def build_ram_specs(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = [{"ram_id": f"ram:{gen.lower()}:{cap}gb", "ddr_gen": gen, "total_capacity_gb": cap, "release_date": pd.NA}
            for gen, caps in cfg["ram"]["capacities_gb"].items() for cap in caps]
    return pd.DataFrame(rows)


def build_support(parts: pd.DataFrame, chips: pd.DataFrame) -> pd.DataFrame:
    facts = pd.read_csv(MANUAL / "family_facts.csv", dtype=str, keep_default_na=False)
    uarch = facts[facts["kind"] == "cpu"].set_index("family")["uarch"]
    wiki = wiki_support_matrix()
    lookup = {(r.uarch, r.chipset): r for r in wiki.itertuples()}
    fam_path = MANUAL / "chipset_support_by_uarch.csv"
    fam_manual = pd.read_csv(fam_path, dtype=str, keep_default_na=False) if fam_path.exists() else pd.DataFrame()
    fam_lookup = {(r.uarch, r.chipset): r for r in fam_manual.itertuples() if r.support.strip()}
    cpus = parts[parts["category"] == "CPU"]
    rows = []
    for c in cpus.itertuples():
        for ch in chips.itertuples():
            if c.socket != ch.socket:  # 소켓이 다르면 지원 여부를 따질 필요 없음(HC-001이 실패 처리)
                continue
            u = uarch.get(c.family)
            f = fam_lookup.get((u, ch.name))
            if f is not None:  # 사람이 마이크로아키텍처 단위로 채운 값이 위키보다 우선
                rows.append({"cpu_id": c.part_id, "chipset_id": ch.chipset_id, "support": f.support.strip(),
                             "source_url": f.source_url, "verified": f.verified.strip().lower() == "true",
                             "note": f.note, "origin": "MANUAL_UARCH"})
                continue
            w = lookup.get((u, ch.name))
            rows.append({
                "cpu_id": c.part_id, "chipset_id": ch.chipset_id,
                "support": (w.support or "NOT_LISTED") if w is not None else "NOT_LISTED",
                "source_url": w.url if w is not None else "", "verified": False,
                "note": f"위키 칩셋 표({uarch.get(c.family)} 칸): {w.support_text}" if w is not None else "위키 표에 칸 없음",
                "origin": "WIKIPEDIA" if w is not None else "NONE",
            })
    support = pd.DataFrame(rows)
    manual = pd.read_csv(MANUAL / "cpu_chipset_support.csv", dtype=str, keep_default_na=False)
    if not manual.empty:
        # 시트의 "true"/"false" 문자열을 bool로(문자열 "false"는 bool로 바꾸면 참이 된다)
        manual = manual.assign(origin="MANUAL", verified=manual["verified"].str.strip().str.lower() == "true")
        key = ["cpu_id", "chipset_id"]
        support = pd.concat([manual, support]).drop_duplicates(subset=key, keep="first")
    return support


def write_uarch_todo(support: pd.DataFrame, parts: pd.DataFrame, chips: pd.DataFrame) -> int:
    """NOT_LISTED인 (마이크로아키텍처, 칩셋) 조합을 사람 입력 시트로(CPU마다가 아니라 묶어서 → 입력량 축소).
    사람이 이미 채운 행(origin=MANUAL_UARCH)도 남겨야 다음 재실행에서 입력이 사라지지 않는다."""
    facts = pd.read_csv(MANUAL / "family_facts.csv", dtype=str, keep_default_na=False)
    uarch = facts[facts["kind"] == "cpu"].set_index("family")["uarch"]
    fam = parts.set_index("part_id")["family"]
    name = chips.set_index("chipset_id")["name"]
    open_or_filled = (support["support"] == "NOT_LISTED") | (support["origin"] == "MANUAL_UARCH")
    todo = support[open_or_filled].assign(
        uarch=lambda d: d["cpu_id"].map(fam).map(uarch), chipset=lambda d: d["chipset_id"].map(name))
    todo = todo.groupby(["uarch", "chipset"]).size().reset_index(name="n_cpus")
    path = MANUAL / "chipset_support_by_uarch.csv"
    if path.exists():
        done = pd.read_csv(path, dtype=str, keep_default_na=False)
        done = done[done["chipset"].isin(set(todo["chipset"]))]
        todo = todo.merge(done.drop(columns=["n_cpus"], errors="ignore"), on=["uarch", "chipset"], how="left")
    for c in ["support", "source_url", "verified", "note"]:
        todo[c] = todo[c].fillna("") if c in todo else ""
    todo.to_csv(path, index=False)
    return len(todo)


def main() -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    parts, issues, info = build_parts(cfg)
    chips = build_chipsets(cfg)
    rams = build_ram_specs(cfg)
    support = build_support(parts, chips)
    n_todo = write_uarch_todo(support, parts, chips)  # 사람 입력 시트는 칩셋 이름(분할 전) 기준
    chips, support = split_board_memory(chips, support)

    def ser(df: pd.DataFrame) -> pd.DataFrame:
        return df.map(lambda v: LIST_SEPARATOR.join(v) if isinstance(v, tuple) else v)

    ser(parts).to_csv(OUT / "parts.csv", index=False)
    ser(chips).to_csv(OUT / "chipsets.csv", index=False)
    rams.to_csv(OUT / "ram_specs.csv", index=False)
    support.to_csv(OUT / "cpu_chipset_support.csv", index=False)
    issues.to_csv(INTERIM / "assemble_issues.csv", index=False)
    print(f"parts {len(parts)} (범위 전 출시 제외 {info['dropped_before_window']}), PARTIAL "
          f"{int((parts['record_status'] == 'PARTIAL').sum())}, 타입 문제 {len(issues)}")
    print(f"chipsets {len(chips)}(분할 전 {chips['base_chipset_id'].nunique()}), ram {len(rams)}, support {len(support)} {support['support'].value_counts().to_dict()}")
    print(f"사람 입력 필요(마이크로아키텍처×칩셋): {n_todo}행 → data/manual/chipset_support_by_uarch.csv")


if __name__ == "__main__":
    main()
