"""위키 원본(data/raw/wikipedia) → 정규화 중간 산출물(data/interim).

산출
- parts_wiki.csv: parts 스키마의 문자열 행(정제 파이프라인 입력). 표 문구는 없고 사실값만.
- chipsets_wiki.csv: 칩셋 이름·출시일(소켓·메모리 세대는 family_facts에서 붙인다)
- provenance_wiki.csv: (entity_id, field) → 셀 각주 원출처 URL(없으면 위키 리비전 URL, WIKIPEDIA_UNCITED)
- excluded_wiki.csv: 범위 규칙으로 뺀 행(OEM 전용·PRO·임베디드 등)과 이유

계산값: fp32_tflops = shader_count × 2 × boost_clock_mhz / 1e6 (공식 공개, 복사값 아님)
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pccompat import parse_values as pv
from pccompat.config import PROJECT_ROOT
from pccompat.wikitable import Cell, Table, parse_tables, reference_urls

RAW = PROJECT_ROOT / "data" / "raw" / "wikipedia"
OUT = PROJECT_ROOT / "data" / "interim"
CONFIG = PROJECT_ROOT / "config" / "wikipedia.yml"
MANUAL = PROJECT_ROOT / "data" / "manual"

PART_FIELDS = [
    "part_id", "category", "vendor", "family", "model_name", "socket", "memory_gens", "max_memory_gb",
    "cores", "threads", "boost_clock_mhz", "tdp_w", "l1_cache_kb", "l2_cache_mb", "l3_cache_mb",
    "shader_count", "fp32_tflops", "pcie_gen", "pcie_lanes",
    "release_date", "release_date_type", "release_date_precision", "msrp_usd",
]
PROV_FIELDS = ["entity_id", "field", "value", "source_url", "source_tier", "source_revision", "license_scope", "verified"]


@dataclass
class Collected:
    parts: list[dict[str, Any]] = field(default_factory=list)
    chipsets: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)


def _find(headers: list[str], pattern: str, skip: set[int] = frozenset()) -> int | None:
    rx = re.compile(pattern, re.I)
    for i, h in enumerate(headers):
        if i not in skip and rx.search(h.strip()):
            return i
    return None


def _find_all(headers: list[str], pattern: str) -> list[int]:
    rx = re.compile(pattern, re.I)
    return [i for i, h in enumerate(headers) if rx.search(h.strip())]


class _Row:
    """한 표 행. 출처는 값 셀 → 행(모델명 셀) → 열(머리글) 순으로 찾고, 찾은 수준을 등급으로 남긴다."""

    def __init__(self, cells: list[Cell], refs: dict[str, list[str]], rev_url: str,
                 header_cites: list[tuple[str, ...]] | None = None, name_idxs: tuple[int | None, ...] = ()):
        self.cells, self.refs, self.rev_url = cells, refs, rev_url
        self.header_cites = header_cites or []
        self.name_idxs = name_idxs

    def text(self, idx: int | None) -> str | None:
        return None if idx is None or idx >= len(self.cells) else self.cells[idx].text

    def _urls(self, cite_ids: Any) -> list[str]:
        urls: list[str] = []
        for cid in cite_ids:
            urls += self.refs.get(cid, [])
        return list(dict.fromkeys(urls))

    def sources(self, *idxs: int | None) -> list[str]:
        return self._urls(c for i in idxs if i is not None and i < len(self.cells) for c in self.cells[i].cite_ids)

    def tiered(self, *idxs: int | None) -> tuple[list[str], str]:
        cell = self.sources(*idxs)
        if cell:
            return cell, "WIKIPEDIA_CITED_CELL"
        row = self.sources(*self.name_idxs)
        if row:
            return row, "WIKIPEDIA_CITED_ROW"
        col = self._urls(c for i in idxs if i is not None and i < len(self.header_cites) for c in self.header_cites[i])
        if col:
            return col, "WIKIPEDIA_CITED_COLUMN"
        return [], "WIKIPEDIA_UNCITED"


def _cpu_name(brand: str, model: str) -> str:
    brand, model = brand.strip(), model.strip()
    if model.lower().startswith(brand.lower()):
        return model
    return f"{brand}-{model}" if re.fullmatch(r"Core i\d", brand) else f"{brand} {model}"


def _gpu_name(brand: str | None, model: str) -> str:
    model = re.sub(r"\s*\([^)]*\)", "", model).strip()
    if brand and re.match(r"Arc \d", brand):  # 'Arc 3' + 'A310' → 'Arc A310'
        return f"Arc {model}"
    return model


def _prov(c: Collected, entity: str, fname: str, value: Any, src: tuple[list[str], str] | list[str], rev: str,
          rev_url: str, row: dict[str, Any] | None = None) -> None:
    if value is None:
        return
    urls, tier = src if isinstance(src, tuple) else (src, "WIKIPEDIA_CITED_CELL" if src else "WIKIPEDIA_UNCITED")
    c.provenance.append({
        "_row": id(row) if row is not None else None,
        "entity_id": entity, "field": fname, "value": value,
        "source_url": urls[0] if urls else rev_url,
        "source_tier": tier,
        "source_revision": rev, "license_scope": "FACT_CITED", "verified": "false",
    })


def _excluded(text: str, patterns: list[str], unless: list[str] | tuple[str, ...] = ()) -> str | None:
    """범위 밖이면 걸린 패턴을 돌려준다. unless 패턴이 보이면(예: OEM 선출시 뒤 리테일 판매) 제외하지 않는다."""
    if any(re.search(u, text, re.I) for u in unless):
        return None
    for p in patterns:
        if re.search(p, text):
            return p
    return None


def _normalize_cpu(src: dict, table: Table, refs: dict, rec: dict, cols: dict, c: Collected, excl: list[str],
                    unless: tuple[str, ...] = ()) -> None:
    h = table.headers
    bi = _find(h, cols["brand"])
    mi = _find(h, cols["model"], skip={bi} if bi is not None else set())
    ct, pc, ec = _find(h, cols["cores_threads"]), _find(h, cols["p_cores"]), _find(h, cols["e_cores"])
    boost_cols = _find_all(h, cols["boost_ghz"])
    tdp, rel, price = _find(h, cols["tdp"]), _find(h, cols["release"]), _find(h, cols["price"])
    l3i, l2i, l1i = _find(h, cols["l3"]), _find(h, cols["l2"]), _find(h, cols["l1"])
    for cells in table.rows:
        r = _Row(cells, refs, rec["url"], table.header_cites, (bi, mi))
        brand, model = r.text(bi) or "", r.text(mi) or ""
        if not model:
            continue
        name = _cpu_name(brand, model)
        why = _excluded(" ".join(x.text for x in cells), excl, unless)
        if why:
            c.excluded.append({"source": src["id"], "model_name": name, "reason": why})
            continue
        if pc is not None:
            p, e = pv.cores_threads(r.text(pc)), pv.cores_threads(r.text(ec))
            cores = (p[0] or 0) + (e[0] or 0) if p[0] is not None else None
            threads = (p[1] or 0) + (e[1] or 0) if p[1] is not None else None
            core_src = r.tiered(pc, ec)
        else:
            cores, threads = pv.cores_threads(r.text(ct))
            core_src = r.tiered(ct)
        boosts = [b for b in (pv.ghz_to_mhz(r.text(i)) for i in boost_cols) if b is not None]
        rdate, precision = pv.release(r.text(rel))
        row = {
            "part_id": f"cpu:{src['vendor'].lower()}:{pv.slug(name)}", "category": "CPU", "vendor": src["vendor"],
            "family": src["family"], "model_name": name, "cores": cores, "threads": threads,
            "boost_clock_mhz": max(boosts) if boosts else None, "tdp_w": pv.watts(r.text(tdp)),
            "l3_cache_mb": pv.size_mb(r.text(l3i)), "l2_cache_mb": pv.size_mb(r.text(l2i)),
            "l1_cache_kb": (lambda v: round(v * 1024, 1) if v is not None else None)(pv.size_mb(r.text(l1i))),
            "release_date": rdate.isoformat() if rdate else None, "release_date_precision": precision,
            "release_date_type": "LAUNCH" if rdate else None, "msrp_usd": pv.usd(r.text(price)),
        }
        c.parts.append(row)
        e, rev = row["part_id"], str(rec["revid"])
        for fname, val, urls in [
            ("cores", cores, core_src), ("threads", threads, core_src),
            ("boost_clock_mhz", row["boost_clock_mhz"], r.tiered(*boost_cols)),
            ("tdp_w", row["tdp_w"], r.tiered(tdp)), ("release_date", row["release_date"], r.tiered(rel)),
            ("msrp_usd", row["msrp_usd"], r.tiered(price)), ("l3_cache_mb", row["l3_cache_mb"], r.tiered(l3i)),
            ("l2_cache_mb", row["l2_cache_mb"], r.tiered(l2i)), ("l1_cache_kb", row["l1_cache_kb"], r.tiered(l1i)),
        ]:
            _prov(c, e, fname, val, urls, rev, rec["url"], row)


def _normalize_gpu(src: dict, table: Table, refs: dict, rec: dict, cols: dict, c: Collected, excl: list[str],
                    unless: tuple[str, ...] = ()) -> None:
    h = table.headers
    bi = _find(h, cols["brand"]) if src["vendor"] == "INTEL" else None
    mi = _find(h, cols["model"], skip={bi} if bi is not None else set())
    rel, price = _find(h, cols["release"]), _find(h, cols["price"])
    cfg_i, clk, tdp = _find(h, cols["core_config"]), _find(h, cols["clock"]), _find(h, cols["tdp"])
    bus = _find(h, cols["bus"]) if "bus" in cols else None
    for cells in table.rows:
        r = _Row(cells, refs, rec["url"], table.header_cites, (bi, mi))
        model = r.text(mi) or ""
        if not model:
            continue
        name = _gpu_name(r.text(bi), model)
        why = _excluded(" ".join(x.text for x in cells), excl, unless)
        if why:
            c.excluded.append({"source": src["id"], "model_name": name, "reason": why})
            continue
        shaders, boost = pv.shader_count(r.text(cfg_i)), pv.boost_mhz(r.text(clk))
        fp32 = round(shaders * 2 * boost / 1e6, 2) if shaders and boost else None
        rdate, precision = pv.release(r.text(rel))
        pgen, plane = pv.pcie(r.text(bus))
        row = {
            "pcie_gen": pgen, "pcie_lanes": plane,
            "part_id": f"gpu:{src['vendor'].lower()}:{pv.slug(name)}", "category": "GPU", "vendor": src["vendor"],
            "family": src["family"], "model_name": name, "shader_count": shaders, "boost_clock_mhz": boost,
            "fp32_tflops": fp32, "tdp_w": pv.watts(r.text(tdp)),
            "release_date": rdate.isoformat() if rdate else None, "release_date_precision": precision,
            "release_date_type": "LAUNCH" if rdate else None, "msrp_usd": pv.usd(r.text(price)),
        }
        c.parts.append(row)
        e, rev = row["part_id"], str(rec["revid"])
        for fname, val, urls in [
            ("shader_count", shaders, r.tiered(cfg_i)), ("boost_clock_mhz", boost, r.tiered(clk)),
            ("tdp_w", row["tdp_w"], r.tiered(tdp)), ("release_date", row["release_date"], r.tiered(rel)),
            ("msrp_usd", row["msrp_usd"], r.tiered(price)),
            ("pcie_gen", pgen, r.tiered(bus)), ("pcie_lanes", plane, r.tiered(bus)),
        ]:
            _prov(c, e, fname, val, urls, rev, rec["url"], row)


def _normalize_chipsets(src: dict, table: Table, refs: dict, rec: dict, cols: dict, c: Collected) -> None:
    h = table.headers
    ni, rel = _find(h, cols["name"]), _find(h, cols["release"])
    if ni is not None:  # 행 = 칩셋
        entries = [(cells[ni].text, cells[rel].text if rel is not None else None,
                    _Row(cells, refs, rec["url"]).sources(ni, rel)) for cells in table.rows]
    else:  # 전치 표(AM5): 머리글 = 칩셋, '출시' 행에서 날짜
        date_row = next((row for row in table.rows if any(re.search("release", x.text, re.I) for x in row[:3])), None)
        entries = []
        for i, name in enumerate(h):
            if name and not re.search(r"^\s*$", name):
                d = date_row[i].text if date_row is not None and i < len(date_row) else None
                entries.append((name, d, list(dict.fromkeys(u for x in ([date_row[i]] if date_row else []) for cid in x.cite_ids for u in refs.get(cid, [])))))
    for name, dtext, urls in entries:
        for single in [n.strip() for n in name.split("/") if n.strip()]:  # 'A620 / A620A'
            rdate, precision = pv.release(dtext)
            cid = f"chipset:{src['vendor'].lower()}:{pv.slug(single)}"
            c.chipsets.append({
                "chipset_id": cid, "vendor": src["vendor"], "name": single, "family": src["family"],
                "release_date": rdate.isoformat() if rdate else None, "release_date_precision": precision,
            })
            _prov(c, cid, "release_date", rdate.isoformat() if rdate else None, urls, str(rec["revid"]), rec["url"])


def _dedupe_variants(c: Collected) -> None:
    """같은 모델명의 변형 행(다른 다이·메모리 리비전)은 가장 이른 출시 행 하나만 남기고 나머지는 제외 보고."""
    best: dict[str, dict[str, Any]] = {}
    for row in c.parts:
        cur = best.get(row["part_id"])
        if cur is None or (row["release_date"] or "9999") < (cur["release_date"] or "9999"):
            best[row["part_id"]] = row
    kept_ids = {id(r) for r in best.values()}
    for row in c.parts:
        if id(row) not in kept_ids:
            c.excluded.append({"source": row["family"], "model_name": row["model_name"], "reason": "VARIANT_DUPLICATE"})
    c.parts = [r for r in c.parts if id(r) in kept_ids]
    # 출처 행도 남긴 행의 것만(변형 행의 각주가 섞이지 않게). 칩셋 출처(_row 없음)는 그대로.
    c.provenance = [p for p in c.provenance if p["_row"] is None or p["_row"] in kept_ids]


def collect(cfg: dict[str, Any]) -> Collected:
    c = Collected()
    excl = cfg.get("exclude_row_patterns", [])
    unless = tuple(cfg.get("exclude_row_unless", []))
    for src in cfg["sources"]:
        path = RAW / f"{src['id']}.json"
        if not path.exists():
            raise FileNotFoundError(f"원본 없음: {path} (collect_wiki 먼저 실행)")
        rec = json.loads(path.read_text(encoding="utf-8"))
        tables, refs = parse_tables(rec["html"]), reference_urls(rec["html"])
        for table in tables:
            kind = src["kind"]
            if kind == "cpu":
                _normalize_cpu(src, table, refs, rec, cfg["columns"]["cpu"], c, excl, unless)
            elif kind == "gpu":
                _normalize_gpu(src, table, refs, rec, cfg["columns"]["gpu"], c, excl, unless)
            else:
                _normalize_chipsets(src, table, refs, rec, cfg["columns"]["chipset"], c)
    name_excl = cfg.get("exclude_model_patterns", []) + cfg.get("exclude_model_patterns_extra", [])
    for row in list(c.parts):
        why = _excluded(row["model_name"], name_excl)
        if why:
            c.excluded.append({"source": row["family"], "model_name": row["model_name"], "reason": why})
            c.parts.remove(row)
    _dedupe_variants(c)
    return c


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    c = collect(cfg)
    _write(OUT / "parts_wiki.csv", c.parts, PART_FIELDS)
    _write(OUT / "chipsets_wiki.csv", c.chipsets,
           ["chipset_id", "vendor", "name", "family", "release_date", "release_date_precision"])
    _write(OUT / "provenance_wiki.csv", c.provenance, PROV_FIELDS)
    _write(OUT / "excluded_wiki.csv", c.excluded, ["source", "model_name", "reason"])
    from collections import Counter
    tiers = Counter(p["source_tier"] for p in c.provenance)
    print(f"parts {len(c.parts)}, chipsets {len(c.chipsets)}, provenance {len(c.provenance)}, excluded {len(c.excluded)}")
    print("출처 등급:", dict(tiers))


if __name__ == "__main__":
    main()
