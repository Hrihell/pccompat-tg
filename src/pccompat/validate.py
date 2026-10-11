"""정제 전후 통계·분포 점검.

- profile: 컬럼별 건수·결측률·고유값 수·수치 요약(EDA 겸 전후 비교 입력)
- compare_profiles: 전후 차이표(결측 증가 = 파싱 실패, 평균 이동 = 단위 오류 의심)
- distribution_shift: 수치 컬럼 KS 검정, 범주 컬럼 빈도 차이(분할 간 분포 비교에도 사용)
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        hashable = s.map(lambda v: v if not isinstance(v, (list, dict)) else str(v))
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if is_num:
            num = _numeric(s)
        elif pd.api.types.is_datetime64_any_dtype(s):
            num = pd.Series(dtype=float)
        else:  # 원본 CSV는 문자열이다: 값 대부분(90%↑)이 숫자로 읽히면 정제 전 통계에도 수치 요약을 낸다
            parsed = _numeric(hashable.where(hashable.map(lambda v: isinstance(v, (str, int, float))), None))
            nonnull = s.notna().sum()
            num = parsed if nonnull and parsed.notna().sum() / nonnull >= 0.9 else pd.Series(dtype=float)
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "count": int(s.notna().sum()),
                "missing_rate": float(s.isna().mean()) if len(s) else 0.0,
                "n_unique": int(hashable.nunique(dropna=True)),
                "mean": float(num.mean()) if num.notna().any() else None,
                "median": float(num.median()) if num.notna().any() else None,
                "std": float(num.std()) if num.notna().sum() > 1 else None,
                "min": float(num.min()) if num.notna().any() else None,
                "max": float(num.max()) if num.notna().any() else None,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def compare_profiles(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    cols = before.index.intersection(after.index)
    out = pd.DataFrame(index=cols)
    out["count_before"] = before.loc[cols, "count"]
    out["count_after"] = after.loc[cols, "count"]
    out["missing_delta"] = after.loc[cols, "missing_rate"] - before.loc[cols, "missing_rate"]
    out["mean_before"] = before.loc[cols, "mean"]
    out["mean_after"] = after.loc[cols, "mean"]
    out["dtype_after"] = after.loc[cols, "dtype"]
    return out


def distribution_shift(a: pd.DataFrame, b: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """두 표본(전/후 또는 train/test)의 분포 차이. 수치는 KS, 범주는 총변동거리(TVD)."""
    rows = []
    for c in numeric:
        x, y = _numeric(a[c]).dropna(), _numeric(b[c]).dropna()
        if len(x) > 1 and len(y) > 1:
            res = stats.ks_2samp(x, y)
            rows.append({"column": c, "test": "KS", "statistic": float(res.statistic), "p_value": float(res.pvalue)})
    for c in categorical:
        p = a[c].value_counts(normalize=True, dropna=False)
        q = b[c].value_counts(normalize=True, dropna=False)
        tvd = 0.5 * p.subtract(q, fill_value=0).abs().sum()
        rows.append({"column": c, "test": "TVD", "statistic": float(tvd), "p_value": None})
    return pd.DataFrame(rows, columns=["column", "test", "statistic", "p_value"])


def to_markdown(df: pd.DataFrame, title: str) -> str:
    return f"### {title}\n\n" + df.to_string() + "\n"
