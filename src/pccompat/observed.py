"""linuxhw DMI(CC BY 4.0) → 실제 PC에서 관찰된 (보드, 칩셋, CPU, 메모리 세대, BIOS) 조합 집계.

- 보고서마다: Base Board 제조사·모델, BIOS 버전·날짜, Processor Version, 장착된 Memory Device의 Type(DDR4/DDR5).
- 칩셋: 보드 모델명에 들어 있는 칩셋 이름(B650, Z790 …, chipsets.csv의 이름 목록과 대조). 못 찾으면 비운다.
- CPU: names.py 확정 매칭만.
- 공개: 집계 행(observed_combos)만. 보고서 경로·HWID·일련번호 같은 식별자는 저장하지 않는다(개인 PC 덤프).
- 의미: "이 조합으로 리눅스가 부팅된 PC가 n대 있었다"(관찰 증거). 공식 지원 여부가 아니며,
  관찰된 BIOS 버전은 필요한 최소 버전의 상한일 뿐이다.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from pccompat.config import PROJECT_ROOT, load_config
from pccompat.names import Catalog, MatchStatus, NameMatcher

DMI = PROJECT_ROOT / "data" / "raw" / "linuxhw" / "DMI"
OUT = PROJECT_ROOT / "data" / "out"


def _field(block: str, name: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(name)}:\s*(.+?)\s*$", block, re.M)
    return m.group(1) if m else None


def parse_report(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"board_vendor": None, "board_model": None, "bios_version": None, "bios_date": None,
                           "cpu_version": None, "ram_gens": set()}
    for block in text.split("\n\n"):
        head = block.lstrip().split("\n", 1)[0]
        if "Base Board Information" in block and out["board_model"] is None:
            out["board_vendor"], out["board_model"] = _field(block, "Manufacturer"), _field(block, "Product Name")
        elif head.startswith("BIOS Information") or "\nBIOS Information" in block[:200]:
            out["bios_version"], out["bios_date"] = _field(block, "Version"), _field(block, "Release Date")
        elif "Processor Information" in block and out["cpu_version"] is None:
            out["cpu_version"] = _field(block, "Version")
        elif re.search(r"^Memory Device\s*$", block, re.M):
            size, typ = _field(block, "Size"), _field(block, "Type")
            if size and "No Module" not in size and typ in ("DDR4", "DDR5"):
                out["ram_gens"].add(typ)
    return out


def chipset_from_board(board_model: str | None, chipset_names: list[str]) -> str | None:
    """보드 모델명에서 칩셋 이름을 찾는다. 'B650E'처럼 긴 이름을 먼저 본다. 두 개 이상 걸리면 None."""
    if not board_model:
        return None
    hits = []
    for name in sorted(chipset_names, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9]){re.escape(name)}(?![0-9])", board_model.upper()):
            if not any(name in h for h in hits):
                hits.append(name)
    return hits[0] if len(hits) == 1 else None


def _reports(root: Path) -> Iterator[str]:
    for dirpath, _, files in os.walk(root / "Desktop"):
        for f in files:
            try:
                yield (Path(dirpath) / f).read_text(errors="ignore")
            except OSError:
                continue


def _date(s: str | None) -> str | None:
    try:
        return datetime.strptime(s, "%m/%d/%Y").date().isoformat() if s else None
    except ValueError:
        return None


def aggregate(parts: pd.DataFrame, chips: pd.DataFrame, cfg: dict[str, Any], root: Path = DMI) -> tuple[pd.DataFrame, Counter]:
    cpus = parts[parts["category"] == "CPU"]
    matcher = NameMatcher(Catalog.from_rows(((r.part_id, r.model_name, []) for r in cpus.itertuples()), cfg["matching"]),
                          cfg["matching"])
    # 보드 이름에는 칩셋 이름(B760)만 있으므로 분할 전 이름·ID로 찾는다. 변형은 pair_observations에서 램 세대로 정한다.
    base = chips[["base_name", "base_chipset_id"]].drop_duplicates() if "base_name" in chips else \
        chips[["name", "chipset_id"]].set_axis(["base_name", "base_chipset_id"], axis=1)
    name_to_id = dict(zip(base["base_name"].str.upper(), base["base_chipset_id"]))
    seen: dict[str, str | None] = {}
    counts: Counter = Counter()
    first_bios: dict[tuple, str] = {}
    stats: Counter = Counter()
    for text in _reports(root):
        r = parse_report(text)
        if not r["cpu_version"]:
            stats["no_cpu"] += 1
            continue
        if r["cpu_version"] not in seen:
            m = matcher.match(r["cpu_version"])
            seen[r["cpu_version"]] = m.model_id if m.status is MatchStatus.CONFIRMED else None
        cpu = seen[r["cpu_version"]]
        if cpu is None:
            stats["cpu_out_of_scope"] += 1
            continue
        chip_name = chipset_from_board(r["board_model"], list(name_to_id))
        ram = ";".join(sorted(r["ram_gens"])) or None
        key = (r["board_vendor"], r["board_model"], name_to_id.get(chip_name) if chip_name else None, cpu, ram,
               r["bios_version"])
        counts[key] += 1
        d = _date(r["bios_date"])
        if d and (key not in first_bios or d < first_bios[key]):
            first_bios[key] = d
        stats["used"] += 1
        stats["chipset_found" if chip_name else "chipset_unknown"] += 1
    rows = [{"board_vendor": k[0], "board_model": k[1], "chipset_id": k[2], "cpu_id": k[3], "ram_gen": k[4],
             "bios_version": k[5], "bios_date": first_bios.get(k), "n_reports": n} for k, n in counts.items()]
    return pd.DataFrame(rows), stats


def to_board_variants(obs: pd.DataFrame, chips: pd.DataFrame | None) -> pd.DataFrame:
    """관찰의 칩셋(보드 이름에서 찾은 분할 전 ID)을 램 세대로 보드 메모리 변형 노드에 붙인다.

    분할된 칩셋인데 램 세대가 하나로 정해지지 않은 보고는 어느 보드인지 알 수 없어 쌍 관찰에서 뺀다.
    """
    if chips is None or "board_memory" not in chips:
        return obs
    split = chips[chips["board_memory"].notna()]
    variant = {(b, g): c for c, b, g in zip(split["chipset_id"], split["base_chipset_id"], split["board_memory"])}
    is_split = obs["chipset_id"].isin(set(split["base_chipset_id"]))
    mapped = [variant.get((c, g)) if s else c for c, g, s in zip(obs["chipset_id"], obs["ram_gen"], is_split)]
    out = obs.assign(chipset_id=mapped)
    return out[~(is_split & out["chipset_id"].isna())]


def pair_observations(obs: pd.DataFrame, chips: pd.DataFrame | None = None) -> pd.DataFrame:
    """쌍 단위 관찰 수: CPU–칩셋, CPU–RAM 세대, 칩셋–RAM 세대(src, dst_kind, dst, n)."""
    o = to_board_variants(obs.dropna(subset=["cpu_id"]), chips)
    out = []
    cc = o.dropna(subset=["chipset_id"]).groupby(["cpu_id", "chipset_id"])["n_reports"].sum()
    out += [{"edge_type": "CPU_CHIPSET", "src": a, "dst": b, "n_observed": int(n)} for (a, b), n in cc.items()]
    single = o[o["ram_gen"].isin(["DDR4", "DDR5"])]
    cr = single.groupby(["cpu_id", "ram_gen"])["n_reports"].sum()
    out += [{"edge_type": "CPU_RAMGEN", "src": a, "dst": g, "n_observed": int(n)} for (a, g), n in cr.items()]
    hr = single.dropna(subset=["chipset_id"]).groupby(["chipset_id", "ram_gen"])["n_reports"].sum()
    out += [{"edge_type": "CHIPSET_RAMGEN", "src": a, "dst": g, "n_observed": int(n)} for (a, g), n in hr.items()]
    return pd.DataFrame(out)


def apply_to_pairs(pairs: pd.DataFrame, rams: pd.DataFrame, pobs: pd.DataFrame) -> pd.DataFrame:
    """관찰된 쌍의 evidence_type을 BOTH로, n_observed를 채운다. RAM 쌍은 세대 단위 관찰을 용량별 행에 같이 붙인다."""
    out = pairs.copy()
    gen = dict(zip(rams["ram_id"], rams["ddr_gen"]))
    obs = {(r.edge_type, r.src, r.dst): r.n_observed for r in pobs.itertuples()}

    def lookup(r: pd.Series) -> Any:
        if r["edge_type"] == "CPU_CHIPSET":
            return obs.get(("CPU_CHIPSET", r["src"], r["dst"]))
        if r["edge_type"] == "CPU_RAM":
            return obs.get(("CPU_RAMGEN", r["src"], gen.get(r["dst"])))
        return obs.get(("CHIPSET_RAMGEN", r["src"], gen.get(r["dst"])))

    n = out.apply(lookup, axis=1)
    out["n_observed"] = n
    out["evidence_type"] = n.notna().map({True: "BOTH", False: "DOCUMENTED"})
    return out


def refresh_pairs() -> pd.DataFrame:
    """원본 3.4GB를 다시 읽지 않고 observed_combos.csv에서 쌍 관찰만 다시 센다(칩셋 노드가 바뀌었을 때)."""
    pobs = pair_observations(pd.read_csv(OUT / "observed_combos.csv"), pd.read_csv(OUT / "chipsets.csv"))
    pobs.to_csv(OUT / "observed_pairs.csv", index=False)
    return pobs


def source_commit(root: Path = DMI) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    cfg = load_config()
    parts = pd.read_csv(OUT / "parts.csv")
    chips = pd.read_csv(OUT / "chipsets.csv")
    obs, stats = aggregate(parts, chips, cfg)
    obs["source_commit"] = source_commit()
    obs.to_csv(OUT / "observed_combos.csv", index=False)
    pobs = pair_observations(obs, chips)
    pobs.to_csv(OUT / "observed_pairs.csv", index=False)
    print("보고서:", dict(stats))
    print(f"관찰 조합 {len(obs):,}행 / CPU {obs['cpu_id'].nunique()} / 보드 {obs['board_model'].nunique()} / "
          f"쌍 관찰 {pobs.groupby('edge_type').size().to_dict()}")


if __name__ == "__main__":
    main()
