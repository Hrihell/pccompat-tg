"""정제 파이프라인(EDA → 타입 → 결측 → 이상치 → 비교).

결정 근거(데이터 시트에도 그대로 옮긴다)
- 결측: 스펙 값은 사실이라 추정으로 채우지 않는다. 비우고 record_status=PARTIAL로 표시하면
  규칙 엔진이 UNKNOWN/MISSING으로 판정한다. 채우면 "모르는 것을 아는 척"하는 라벨이 생긴다.
  (v2 모델 베이스라인용 특징 행렬에서만 train 중앙값 대치를 쓴다 → features.py)
- 이상치: 범위 밖 값도 지우지 않는다. 출처 오기인지 실제 값인지는 원출처를 봐야 알 수 있으므로
  review_flags에 남겨 사람이 확인한다. 예외는 실측 원표본(사용자 업로드)으로, 모델별 IQR 밖 표본은
  측정 오류·설정 차이일 가능성이 높아 중앙값 계산 전에 잘라내고 잘라낸 수를 기록한다.
- 타입 변환을 결측·이상치보다 먼저 한다: 문자열 숫자는 범위 검사를 할 수 없고, 파싱 실패가 새 결측을
  만들기 때문이다. 파싱 실패는 조용히 버리지 않고 issues로 보고한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pccompat.schema import LIST_SEPARATOR, REQUIRED_BY_CATEGORY, Column, Table
from pccompat.validate import compare_profiles, profile


@dataclass(frozen=True)
class Issue:
    row: int
    column: str
    value: Any
    problem: str


_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (tuple, list, dict)):
        return False
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        return False
    return isinstance(v, str) and not v.strip()


def _parse_list(v: Any, col: Column) -> tuple[tuple[str, ...] | None, str | None]:
    if _is_blank(v):
        return None, None
    items = v if isinstance(v, (list, tuple)) else [s.strip() for s in str(v).split(LIST_SEPARATOR)]
    items = tuple(s for s in items if s)
    if not col.allowed:
        return items, None
    canon = {a.casefold(): a for a in col.allowed}
    bad = [s for s in items if s.casefold() not in canon]
    return (None, f"허용값 밖: {bad}") if bad else (tuple(canon[s.casefold()] for s in items), None)


def _coerce_column(series: pd.Series, col: Column) -> tuple[pd.Series, list[Issue]]:
    issues: list[Issue] = []
    if col.kind in ("int", "float"):
        num = pd.to_numeric(series, errors="coerce")
        failed = series.notna() & num.isna() & ~series.map(_is_blank)
        if col.kind == "int":
            # 정수가 아니거나 Int64에 안 들어가는 값(1e30·inf)은 변환 실패로 보고(전체 실행이 죽지 않게)
            nonint = num.notna() & (~np.isfinite(num) | (num.abs() >= 2**63) | (num % 1 != 0))
            failed |= nonint
            num = num.where(~nonint)
            num = num.astype("Int64")
        issues += [Issue(int(i), col.name, series[i], f"{col.kind} 변환 실패") for i in series.index[failed]]
        return num, issues
    if col.kind == "date":
        dt = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
        failed = series.notna() & dt.isna() & ~series.map(_is_blank)
        issues += [Issue(int(i), col.name, series[i], "날짜 형식(YYYY-MM-DD) 아님") for i in series.index[failed]]
        return dt, issues
    if col.kind == "bool":
        def to_bool(v: Any) -> Any:
            if _is_blank(v) or isinstance(v, bool):
                return pd.NA if _is_blank(v) else v
            s = str(v).strip().lower()
            if s in _TRUE or s in _FALSE:
                return s in _TRUE
            try:  # CSV가 1.0/0.0으로 저장한 bool
                f = float(s)
            except ValueError:
                return "BAD"
            return bool(f) if f in (0.0, 1.0) else "BAD"
        mapped = series.map(to_bool)
        failed = mapped.eq("BAD")
        issues += [Issue(int(i), col.name, series[i], "bool 변환 실패") for i in series.index[failed]]
        return mapped.mask(failed, pd.NA).astype("boolean"), issues
    if col.kind == "enum":
        canon = {a.casefold(): a for a in col.allowed}  # 대소문자만 다른 값은 표준 표기로
        stripped = series.map(lambda v: pd.NA if _is_blank(v) else str(v).strip())
        canonical = stripped.map(lambda v: canon.get(v.casefold(), pd.NA) if pd.notna(v) else pd.NA)
        failed = stripped.notna() & canonical.isna()
        issues += [Issue(int(i), col.name, series[i], f"허용값 밖 {col.allowed}") for i in series.index[failed]]
        return canonical.astype("object").where(canonical.notna(), pd.NA), issues
    if col.kind == "list":
        out = []
        for i, v in series.items():
            parsed, err = _parse_list(v, col)
            if err:
                issues.append(Issue(int(i), col.name, v, err))
            out.append(parsed)
        return pd.Series(out, index=series.index, dtype="object"), issues
    if col.kind == "json":
        def to_json(v: Any) -> Any:
            if _is_blank(v) or isinstance(v, dict):
                return None if _is_blank(v) else v
            try:
                return json.loads(v)
            except (TypeError, ValueError):
                return "BAD"
        mapped = series.map(to_json)
        failed = mapped.map(lambda v: isinstance(v, str) and v == "BAD")
        issues += [Issue(int(i), col.name, series[i], "JSON 아님") for i in series.index[failed]]
        return mapped.mask(failed, None), issues
    cleaned = series.map(lambda v: pd.NA if _is_blank(v) else str(v).strip())
    return cleaned.astype("string"), issues


def coerce_types(df: pd.DataFrame, table: Table) -> tuple[pd.DataFrame, list[Issue]]:
    """스키마 타입으로 변환한 새 프레임과 변환 실패 목록을 돌려준다(입력은 바꾸지 않는다).

    출력 인덱스는 0부터의 위치 번호다. Issue.row도 같은 위치 번호(원본 인덱스가 중복이어도 안전).
    """
    out = df.reset_index(drop=True)
    issues: list[Issue] = []
    for col in table.columns:
        if col.name not in out.columns:
            out[col.name] = pd.Series([None] * len(out), index=out.index, dtype="object")
        converted, col_issues = _coerce_column(out[col.name], col)
        out[col.name] = converted
        issues += col_issues
    unknown = [c for c in df.columns if c not in table.column_names]
    issues += [Issue(-1, c, None, "스키마에 없는 컬럼") for c in unknown]
    for col in table.columns:
        if not col.nullable:
            for i in out.index[out[col.name].map(_is_blank)]:
                issues.append(Issue(int(i), col.name, None, "필수 키 결측"))
    if table.key and all(k in out.columns for k in table.key):
        dup = out.duplicated(subset=list(table.key), keep=False)
        issues += [Issue(int(i), "+".join(table.key), None, "키 중복") for i in out.index[dup]]
    return out, issues


def _required_for(row: pd.Series, table: Table) -> list[str]:
    base = [c.name for c in table.columns if c.required]
    if "category" in row.index and isinstance(row["category"], str):
        base += list(REQUIRED_BY_CATEGORY.get(row["category"], ()))
    return base


def flag_missing(df: pd.DataFrame, table: Table) -> tuple[pd.DataFrame, pd.DataFrame]:
    """결측은 채우지 않는다. 필수 필드 결측 행은 PARTIAL + missing_required 목록으로 표시."""
    out = df.copy()
    missing_lists = [
        tuple(c for c in _required_for(row, table) if c in out.columns and _is_blank(row[c]))
        for _, row in out.iterrows()
    ]
    out["missing_required"] = pd.Series(missing_lists, index=out.index, dtype="object")
    if "record_status" in table.column_names:
        out["record_status"] = ["PARTIAL" if m else "COMPLETE" for m in missing_lists]
    report = pd.DataFrame(
        {
            "missing": [int(out[c].map(_is_blank).sum()) for c in table.column_names],
            "missing_rate": [float(out[c].map(_is_blank).mean()) if len(out) else 0.0 for c in table.column_names],
        },
        index=table.column_names,
    )
    return out, report


def flag_out_of_range(df: pd.DataFrame, ranges: dict[str, list[float]]) -> pd.DataFrame:
    """설정의 범위 밖 값을 review_flags에 남긴다. 값과 행은 그대로 둔다."""
    out = df.copy()
    flags: list[tuple[str, ...]] = []
    for _, row in out.iterrows():
        bad = []
        for column, (low, high) in ranges.items():
            if column in out.columns and not _is_blank(row[column]):
                value = float(row[column])
                if value < low or value > high:
                    bad.append(column)
        flags.append(tuple(bad))
    out["review_flags"] = pd.Series(flags, index=out.index, dtype="object")
    return out


def trim_samples_iqr(
    samples: pd.DataFrame, group_col: str, value_col: str, k: float, min_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """그룹(모델)별 IQR 밖 표본 제거와 그룹별 보고서(group, n, n_nan, n_removed, n_no_key, trimmed).

    - 값이 NaN인 표본은 이상치가 아니라 결측이다: 중앙값 계산에서 빼되 n_nan으로 따로 센다.
    - 표본이 min_n(low_n_threshold) 미만이거나 IQR이 0이면 자르지 않는다(작은 그룹의 실제 값 손실 방지).
    - 그룹 키가 없는 행도 "<NA>" 그룹으로 보고한다(조용히 사라지지 않게).
    """
    kept_parts, rows = [], []
    for name, group in samples.groupby(group_col, sort=True, dropna=False):
        label = "<NA>" if pd.isna(name) else str(name)
        values = pd.to_numeric(group[value_col], errors="coerce")
        valid = group[values.notna()]
        vals = values[values.notna()]
        q1, q3 = (vals.quantile([0.25, 0.75]) if len(vals) else (np.nan, np.nan))
        iqr = q3 - q1
        trimmed = len(vals) >= min_n and bool(iqr > 0)
        keep = vals.between(q1 - k * iqr, q3 + k * iqr) if trimmed else pd.Series(True, index=vals.index)
        if label != "<NA>":
            kept_parts.append(valid[keep])
        no_key = label == "<NA>"
        rows.append({
            "group": label, "n": len(group), "n_nan": int(values.isna().sum()),
            "n_removed": int((~keep).sum()), "n_no_key": int(len(valid)) if no_key else 0, "trimmed": trimmed,
        })
    kept = pd.concat(kept_parts) if kept_parts else samples.iloc[0:0]
    # 회계: len(kept) + n_removed + n_nan + n_no_key == len(samples)
    return kept, pd.DataFrame(rows, columns=["group", "n", "n_nan", "n_removed", "n_no_key", "trimmed"])


def assign_eol(df: pd.DataFrame, lineage: dict[str, list[str]]) -> pd.DataFrame:
    """단종일이 비어 있는 행만 "자기 출시일 이후 처음 나온 같은 계열 후속 세대의 최소 출시일"(가정)로 채운다.

    다음 세대가 나온 뒤에 출시된 모델(예: 5700X3D, RTX 3050 6GB)에 바로 다음 세대 출시일을 쓰면
    단종일이 출시일보다 앞서 판매 구간이 사라진다. 그래서 그 모델 출시 뒤에 처음 나온 세대를 찾는다.

    출처에 있는 단종일은 덮어쓰지 않는다. eol_is_assumed는 이번에 채운 행만 True,
    출처 값이 있는 행은 출처의 표시(없으면 False)를 유지한다. 다음 세대 데이터가 없으면 비워 둔다.
    """
    out = df.copy()
    if "eol_date" not in out.columns:
        out["eol_date"] = pd.NaT
    if "eol_is_assumed" not in out.columns:
        out["eol_is_assumed"] = pd.NA
    first_release = pd.to_datetime(out.groupby("family")["release_date"].min())
    later = {fam: chain[i + 1:] for chain in lineage.values() for i, fam in enumerate(chain)}

    def eol_for(family: Any, released: Any) -> Any:
        starts = [first_release.get(f, pd.NaT) for f in later.get(family, [])]
        starts = [s for s in starts if pd.notna(s) and (pd.isna(released) or s > released)]
        return starts[0] if starts else pd.NaT
    sourced = pd.to_datetime(out["eol_date"])
    released = pd.to_datetime(out["release_date"])
    derived = pd.to_datetime(pd.Series([eol_for(f, r) for f, r in zip(out["family"], released)], index=out.index))
    fill = sourced.isna() & derived.notna()
    out["eol_date"] = sourced.where(~fill, derived)
    prior = out["eol_is_assumed"].astype("boolean")
    out["eol_is_assumed"] = prior.mask(fill, True).mask(~fill & sourced.notna() & prior.isna(), False)
    return out


@dataclass
class CleanResult:
    frame: pd.DataFrame
    reports: dict[str, pd.DataFrame] = field(default_factory=dict)


class CleaningPipeline:
    """표 하나를 정제하고 단계별 보고서를 모은다. 행을 지우지 않는다."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def run(self, raw: pd.DataFrame, table: Table) -> CleanResult:
        before = profile(raw)
        typed, issues = coerce_types(raw, table)
        if "family" in typed.columns and "release_date" in typed.columns and "eol_date" in table.column_names:
            typed = assign_eol(typed, self.cfg["lineage"])
        flagged, missing = flag_missing(typed, table)
        checked = flag_out_of_range(flagged, self.cfg["ranges"])
        after = profile(checked[table.column_names])
        return CleanResult(
            frame=checked,
            reports={
                "before": before,
                "after": after,
                "comparison": compare_profiles(before, after),
                "issues": pd.DataFrame([i.__dict__ for i in issues], columns=["row", "column", "value", "problem"]),
                "missing": missing,
            },
        )
