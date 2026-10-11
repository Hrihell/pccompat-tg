"""G2: GPU 연결 경고(PCIe 대역폭). 3값 라벨과 별개의 표시다.

- 입력: parts.csv의 GPU pcie_gen·pcie_lanes(위키 Bus interface), manual/platform_facts.csv(CPU·칩셋의 GPU용 PCIe).
  사람이 공식 자료로 확인한 행(verified=true, source_url)만 쓴다.
- 출력: (CPU, 칩셋, GPU) 조합별 경고 비트와 미확인 비트. build_labels가 조합 행에 붙인다. 비트 0 = PCIE_BANDWIDTH.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pccompat import rules
from pccompat.config import PROJECT_ROOT

MANUAL = PROJECT_ROOT / "data" / "manual"
WARNINGS = ["PCIE_BANDWIDTH"]


def _verified(df: pd.DataFrame) -> pd.DataFrame:
    ok = df["verified"].astype(str).str.lower().eq("true") & df["source_url"].astype(str).str.strip().ne("")
    return df[ok]


def load_facts(manual: Path = MANUAL) -> pd.DataFrame:
    return _verified(pd.read_csv(manual / "platform_facts.csv", dtype=str, keep_default_na=False))


def _value(v: Any) -> Any:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "" else v


def _cpu_spec(cpu: pd.Series, facts: pd.DataFrame) -> dict[str, Any]:
    """모델 행(kind=cpu_model)이 있으면 그 값, 없으면 세대 행(kind=cpu_family)."""
    out: dict[str, Any] = {}
    for kind, key in (("cpu_family", cpu["family"]), ("cpu_model", cpu["part_id"])):
        rows = facts[(facts["kind"] == kind) & (facts["key"] == key)]
        for col in ("gpu_pcie_gen", "gpu_pcie_lanes"):
            v = _value(rows[col].iloc[0]) if len(rows) else None
            if v is not None:
                out[col] = v
    return out


def warning_tables(cpus: pd.DataFrame, chips: pd.DataFrame, gpus: pd.DataFrame,
                   facts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(CPU, 칩셋, GPU) 축의 경고 비트와 미확인 비트(int8 배열 두 개)."""
    nC, nH, nG = len(cpus), len(chips), len(gpus)
    warn = np.zeros((nC, nH, nG), np.int8)
    unchecked = np.zeros_like(warn)
    cs = [_cpu_spec(c, facts) for _, c in cpus.iterrows()]
    chip_rows = facts[facts["kind"] == "chipset"].set_index("key")
    base = chips["base_name"] if "base_name" in chips else chips["name"]
    hs = [chip_rows.loc[b].to_dict() if b in chip_rows.index else {} for b in base]
    gl = list(gpus.itertuples())
    for i, c in enumerate(cs):
        for j, h in enumerate(hs):
            for k, g in enumerate(gl):
                status = rules.gpu_link(_value(g.pcie_gen), _value(g.pcie_lanes), c.get("gpu_pcie_gen"),
                                        c.get("gpu_pcie_lanes"), _value(h.get("gpu_pcie_gen")))
                if status == rules.WARN:
                    warn[i, j, k] |= 1
                elif status == rules.UNCHECKED:
                    unchecked[i, j, k] |= 1
    return warn, unchecked
