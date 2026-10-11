"""v2 모델 베이스라인 전용 특징 변환(인코딩·스케일링).

공개 CSV에는 적용하지 않는다. 반드시 train에만 fit하고 val·test에는 transform만 한다.
- 범주: 원-핫(처음 보는 범주는 0 벡터, handle_unknown="ignore")
- 수치: train 중앙값 대치 → RobustScaler, 그리고 모든 수치 열의 결측 표시 열(features="all")
  (스펙 분포는 플래그십 때문에 꼬리가 길어 평균·표준편차보다 중앙값·IQR 기반이 덜 흔들린다.
   결측 표시를 모든 열에 두는 이유: train에 결측이 없던 열이 test에서 비어도 "값이 없었다"는 신호가 남아야 한다)
- 정제 결과의 pd.NA(Int64·string dtype)는 sklearn이 다루지 못하므로 먼저 np.nan 기반 dtype으로 바꾼다.
- eol_date처럼 미래 정보(다음 세대 출시일)에서 나온 열은 특징에 넣지 않는다(config.features 목록으로 통제).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


def _to_float(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda s: pd.to_numeric(s, errors="coerce")).astype("float64")


def _to_object(frame: pd.DataFrame) -> pd.DataFrame:
    obj = frame.astype(object)
    return obj.where(frame.notna(), np.nan)


def build_feature_transformer(feature_cfg: Mapping[str, Any]) -> ColumnTransformer:
    values = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler()),
        ]
    )
    numeric = Pipeline(
        [
            ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
            ("union", FeatureUnion([("values", values), ("missing", MissingIndicator(features="all"))])),
        ]
    )
    categorical = Pipeline(
        [
            ("to_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(feature_cfg["numeric"])),
            ("cat", categorical, list(feature_cfg["categorical"])),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def numeric_imputer(fitted: ColumnTransformer) -> SimpleImputer:
    """학습된 변환기에서 수치 대치기를 꺼낸다(누수 검사용)."""
    union = fitted.named_transformers_["num"].named_steps["union"]
    return dict(union.transformer_list)["values"].named_steps["impute"]
