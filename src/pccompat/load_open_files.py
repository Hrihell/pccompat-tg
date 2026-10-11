"""파일 원천 로더: GetDeploying(CC BY 4.0), Kishonti GFXBench·CompuBench(CC BY 4.0), pci.ids(BSD-3).

- GetDeploying: 범위 안 소비자용 NVIDIA 행을 확정 매칭으로 모델에 연결 → 교차 확인용 표 + 행별 공식 데이터시트 URL.
  (CC BY 4.0은 "published dataset files only"에 적용. 사이트·API는 별도 약관이라 쓰지 않는다.)
- Kishonti: 데스크톱·외장 GPU 행만. 값은 최근 100회 결과의 집계(중앙값)라 IQR 절단을 하지 않는다.
  장치명이 'AMD 744C:C8' 같은 PCI ID면 pci.ids로 풀되, 한 ID에 여러 제품이 묶여 있으면(예: 7900 XT/XTX/GRE) 확정하지 않는다.
- 원 문구는 저장하지 않고 사실값·출처만 남긴다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from pccompat.config import PROJECT_ROOT, load_config
from pccompat.names import Catalog, MatchStatus, NameMatcher

RAW = PROJECT_ROOT / "data" / "raw"
INTERIM = PROJECT_ROOT / "data" / "interim"

GETDEPLOYING_FIELDS = {"shader_count": "shader_cores", "tdp_w": "tdp_w", "release_date": "release_date",
                       "fp32_tflops": "fp32_dense_tflops"}


def _matcher(parts: pd.DataFrame, category: str, cfg: dict[str, Any]) -> NameMatcher:
    sub = parts[parts["category"] == category]
    return NameMatcher(Catalog.from_rows(((r.part_id, r.model_name, []) for r in sub.itertuples()), cfg["matching"]),
                       cfg["matching"])


def load_getdeploying(parts: pd.DataFrame, cfg: dict[str, Any], window_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW / "getdeploying" / "gpu-models.csv")
    raw = raw[(raw["market_class"] == "CONSUMER") & (raw["release_date"] >= window_start)]
    m = _matcher(parts, "GPU", cfg)
    rows, unmatched = [], []
    for r in raw.itertuples():
        res = m.match(r.name)
        if res.status is not MatchStatus.CONFIRMED:
            unmatched.append({"name": r.name, "status": res.status.value, "reason": res.reason})
            continue
        row = {"part_id": res.model_id, "name": r.name, "datasheet_url": r.datasheet_url}
        row.update({ours: getattr(r, theirs) for ours, theirs in GETDEPLOYING_FIELDS.items()})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(unmatched)


def parse_pci_ids(path: Path) -> dict[tuple[str, str], str]:
    """(vendor_id, device_id) 소문자 16진 → 장치명. 서브시스템 줄(탭 2개)은 무시."""
    out: dict[tuple[str, str], str] = {}
    vendor = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("C "):  # 클래스 목록부터는 장치가 아니다
            break
        if not line.startswith("\t"):
            vendor = line.split(None, 1)[0].lower()
        elif not line.startswith("\t\t") and vendor:
            dev, name = line.strip().split(None, 1)
            out[(vendor, dev.lower())] = name
    return out


_PCI_NAME = re.compile(r"^(AMD|NVIDIA|Intel)\s+([0-9A-Fa-f]{4}):[0-9A-Fa-f]{2}$")
_VENDOR_ID = {"AMD": "1002", "NVIDIA": "10de", "INTEL": "8086"}


def resolve_pci_name(device_name: str, pci: dict[tuple[str, str], str]) -> str | None:
    """'AMD 744C:C8' → pci.ids 장치명의 대괄호 안 제품명. 제품이 여럿 묶여 있으면(/로 구분) None."""
    m = _PCI_NAME.match(device_name.strip())
    if not m:
        return device_name
    name = pci.get((_VENDOR_ID[m.group(1).upper()], m.group(2).lower()))
    if not name:
        return None
    inner = re.search(r"\[([^\]]+)\]", name)
    label = inner.group(1) if inner else name
    return None if "/" in label else label


def load_kishonti(parts: pd.DataFrame, cfg: dict[str, Any], pci: dict[tuple[str, str], str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = _matcher(parts, "GPU", cfg)
    frames, names_log = [], []
    for fname, source in (("GFXBench-5.0-results.csv", "GFXBENCH"), ("CompuBench-2.0-Desktop-results.csv", "COMPUBENCH")):
        df = pd.read_csv(RAW / "kishonti" / fname,
                         usecols=["test name", "device name", "median result (raw)", "test unit", "os", "api",
                                  "hardware type", "form factor"])
        df = df[(df["form factor"] == "desktop") & (df["hardware type"].isin(["dGPU", "GPU"]))]
        cache: dict[str, tuple[str | None, str]] = {}
        for name in df["device name"].drop_duplicates():
            resolved = resolve_pci_name(name, pci)
            if resolved is None:
                cache[name] = (None, "PCI_ID_AMBIGUOUS_OR_UNKNOWN")
                continue
            r = m.match(resolved)
            cache[name] = (r.model_id if r.status is MatchStatus.CONFIRMED else None, r.status.value)
            names_log.append({"source": source, "device_name": name, "resolved": resolved, "status": r.status.value,
                              "model_id": r.model_id})
        df = df.assign(part_id=df["device name"].map(lambda n: cache[n][0])).dropna(subset=["part_id"])
        out = (df.groupby(["part_id", "test name", "api", "os", "test unit"], as_index=False)
               .agg(median_result=("median result (raw)", "median"), n_device_rows=("device name", "size"),
                    matched_device_names=("device name", lambda x: ";".join(sorted(set(x))))))
        out["source"] = source
        frames.append(out)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(names_log)


def main() -> None:
    cfg = load_config()
    parts = pd.read_csv(INTERIM / "parts_wiki.csv")
    gd, gd_unmatched = load_getdeploying(parts, cfg, str(cfg["window"]["start"]))
    gd.to_csv(INTERIM / "getdeploying_specs.csv", index=False)
    print(f"GetDeploying: 매칭 {len(gd)} / 미매칭 {len(gd_unmatched)} {gd_unmatched.to_dict('records')}")
    pci = parse_pci_ids(RAW / "ids" / "pci.ids")
    kish, log = load_kishonti(parts, cfg, pci)
    kish.to_csv(INTERIM / "perf_kishonti.csv", index=False)
    log.to_csv(INTERIM / "kishonti_name_matches.csv", index=False)
    print(f"Kishonti: 측정 행 {len(kish):,} / 매칭 모델 {kish['part_id'].nunique()} / 장치명 상태 "
          f"{log['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
