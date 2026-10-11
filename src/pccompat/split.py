"""시간 기준 family-group 분할(설계서 §2-1).

- 노드: 계열(family)의 첫 출시일이 속한 구간으로 계열 전체를 보낸다(형제 모델 누수 방지).
  계열 키는 공백·대소문자를 무시해 비교한다(출처마다 표기가 달라 같은 계열이 갈라지는 누수 방지).
  자기 출시일이 없어도 계열 첫 출시일이 있으면 계열을 따른다. 계열이 없으면 자기 출시일을 쓴다.
- 간선·조합: 참여 노드 중 가장 늦은 분할을 물려받는다(test 조합에는 test 노드가 반드시 하나 이상).
- 기준 날짜가 전혀 없으면 분할을 추정하지 않고 비워 두며, split_ratios가 unassigned로 보고한다.
- 무작위성이 없으므로 시드가 필요 없다. 균형 부분집합 추출은 config.seed를 쓴다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

ORDER = {"train": 0, "val": 1, "test": 2}


def _bucket(date: Any, train_end: pd.Timestamp, val_end: pd.Timestamp) -> Any:
    if pd.isna(date):
        return pd.NA
    return "train" if date <= train_end else "val" if date <= val_end else "test"


def family_key(values: pd.Series) -> pd.Series:
    return values.map(lambda v: " ".join(str(v).split()).casefold() if pd.notna(v) and str(v).strip() else pd.NA)


def assign_time_group_split(nodes: pd.DataFrame, split_cfg: Mapping[str, Any]) -> pd.DataFrame:
    date_col, group_col = split_cfg["date_column"], split_cfg["group_column"]
    train_end = pd.Timestamp(split_cfg["train_end"])
    val_end = pd.Timestamp(split_cfg["val_end"])
    out = nodes.copy()
    key = family_key(out[group_col])
    first = out[date_col].groupby(key).transform("min")
    basis = first.where(key.notna(), out[date_col])
    out["split"] = basis.map(lambda d: _bucket(d, train_end, val_end)).astype("string")
    return out


def inherit_latest_split(frame: pd.DataFrame, id_columns: Sequence[str], split_of: Mapping[str, str]) -> pd.DataFrame:
    out = frame.copy()

    def latest(row: pd.Series) -> Any:
        splits = [split_of.get(row[c]) for c in id_columns]
        if any(s is None or pd.isna(s) for s in splits):
            return pd.NA
        return max(splits, key=ORDER.__getitem__)

    out["split"] = out.apply(latest, axis=1).astype("string")
    return out


def split_ratios(frame: pd.DataFrame) -> dict[str, float]:
    """분할 비율(배정된 행 기준)과 미배정 행 수(unassigned, 개수)."""
    counts = frame["split"].value_counts(dropna=True)
    total = counts.sum()
    ratios: dict[str, float] = {k: float(counts.get(k, 0) / total) if total else 0.0 for k in ORDER}
    ratios["unassigned"] = int(frame["split"].isna().sum())
    return ratios
