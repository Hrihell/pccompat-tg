"""파이프라인 설정 로더. 값·컬럼명은 config/pipeline.yml 한 곳에서만 읽는다."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"설정 파일 형식 오류: {path}")
    return data


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """설정을 읽어 사본을 돌려준다(호출자가 바꿔도 캐시가 오염되지 않게)."""
    import copy

    return copy.deepcopy(_load(str(path)))
