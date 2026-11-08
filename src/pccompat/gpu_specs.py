"""D1: 사람이 공식 페이지에서 옮긴 GPU 사양(manual/gpu_spec_check.csv)으로 위키 값을 바로잡는다.

- 공식 값이 시트의 우리 값과 다를 때만 바꾼다: 셰이더는 정확히 같아야 하고, 부스트는 차이 1% 이내면 같다고 본다
  (공식 페이지가 GHz를 소수 둘째 자리까지만 적는 반올림). 일치한 행은 검증 표본처럼 등급을 그대로 둔다.
- 바꾼 값은 OFFICIAL·verified=true 출처가 되고, fp32_tflops는 normalize_wiki와 같은 식으로 다시 계산한다.
- 인텔 셰이더는 Xe-core × 128로 환산해 시트에 적는다(시트 note).
"""

from __future__ import annotations

import pandas as pd

from pccompat.config import PROJECT_ROOT

SHEET = PROJECT_ROOT / "data" / "manual" / "gpu_spec_check.csv"
BOOST_TOLERANCE = 0.01
COLS = ["part_id", "field", "value", "source_url"]


def load() -> pd.DataFrame:
    return pd.read_csv(SHEET, dtype=str, keep_default_na=False) if SHEET.exists() else pd.DataFrame()


def corrections(sheet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in sheet.itertuples():
        if not (r.official_shader.strip() and r.official_boost_mhz.strip()):
            continue
        shader, boost = int(float(r.official_shader)), int(float(r.official_boost_mhz))
        if shader != int(float(r.our_shader)):
            rows.append((r.part_id, "shader_count", str(shader), r.source_url))
        if abs(boost - float(r.our_boost_mhz)) / boost > BOOST_TOLERANCE:
            rows.append((r.part_id, "boost_clock_mhz", str(boost), r.source_url))
    return pd.DataFrame(rows, columns=COLS)


def apply(parts: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
    out = parts.copy()
    for f in fixes.itertuples():
        out.loc[out["part_id"] == f.part_id, f.field] = f.value
    touched = out["part_id"].isin(set(fixes["part_id"]))
    fp32 = out.loc[touched].apply(
        lambda r: str(round(float(r["shader_count"]) * 2 * float(r["boost_clock_mhz"]) / 1e6, 2)), axis=1)
    out.loc[touched, "fp32_tflops"] = fp32
    return out


def provenance_rows(fixes: pd.DataFrame) -> list[dict]:
    return [{"entity_id": f.part_id, "field": f.field, "value": f.value, "source_url": f.source_url,
             "source_tier": "OFFICIAL", "license_scope": "FACT_CITED", "verified": True, "source_revision": "manual"}
            for f in fixes.itertuples()]
