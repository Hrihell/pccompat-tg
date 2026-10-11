"""WP5 분석 d: 스펙 대리 지표가 실측과 얼마나 맞나(Spearman ρ + 부트스트랩 95% CI).

선택 규칙(리뷰 반영, plan-remaining.md §5-4)
- CPU: Blender device_type=CPU. (버전, 장면) 중 매칭 모델이 가장 많은 조합을 주 결과로, 나머지는 부록.
- GPU: 백엔드가 다르면 한 축에 섞지 않는다 → 제조사별 백엔드(NVIDIA: OPTIX, 없으면 CUDA / AMD: HIP / 인텔: ONEAPI).
  각 제조사 안에서 (버전, 장면) 중 모델이 가장 많은 조합을 주 결과로.
- Kishonti: 같은 시험·API는 제조사 간 비교 가능 → (시험, API)별로 전체 ρ, 모델 수 상위를 주 결과로.
- 표본이 min_models 미만이면 ρ를 내지 않는다(설계값 8).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pccompat.config import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "data" / "out"
INTERIM = PROJECT_ROOT / "data" / "interim"
GPU_BACKENDS = {"NVIDIA": ["OPTIX", "CUDA"], "AMD": ["HIP"], "INTEL": ["ONEAPI"]}
MIN_MODELS = 8


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            boots.append(stats.spearmanr(x[idx], y[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(rho), float(p), float(lo), float(hi)


def _rows(df: pd.DataFrame, group_cols: list[str], x: str, y: str, label: dict[str, Any], seed: int,
          n_boot: int) -> list[dict]:
    out = []
    for key, g in df.groupby(group_cols):
        g = g.dropna(subset=[x, y])
        if len(g) < MIN_MODELS:
            continue
        rho, p, lo, hi = spearman_ci(g[x].to_numpy(float), g[y].to_numpy(float), n_boot, seed)
        out.append({**label, **dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                    "n_models": len(g), "rho": round(rho, 3), "p_value": p, "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    return out


def run(parts: pd.DataFrame, blender: pd.DataFrame, kishonti: pd.DataFrame, seed: int, n_boot: int = 1000) -> pd.DataFrame:
    p = parts[["part_id", "category", "vendor", "proxy_perf"]]
    rows: list[dict] = []
    cpu = blender[blender["device_type"] == "CPU"].merge(p, on="part_id")
    cpu = cpu[~cpu["low_n_flag"].astype(bool)]
    rows += _rows(cpu, ["benchmark_version", "scene", "device_type"], "proxy_perf", "median_samples_per_min",
                  {"target": "CPU", "source": "BLENDER", "vendor": "ALL"}, seed, n_boot)
    gpu = blender[blender["device_type"] != "CPU"].merge(p, on="part_id")
    gpu = gpu[~gpu["low_n_flag"].astype(bool)]
    for vendor, backends in GPU_BACKENDS.items():
        sub = gpu[(gpu["vendor"] == vendor) & gpu["device_type"].isin(backends)]
        if vendor == "NVIDIA" and (sub["device_type"] == "OPTIX").any():
            sub = sub[sub["device_type"] == "OPTIX"]
        rows += _rows(sub, ["benchmark_version", "scene", "device_type"], "proxy_perf", "median_samples_per_min",
                      {"target": "GPU", "source": "BLENDER", "vendor": vendor}, seed, n_boot)
    k = kishonti.merge(p, on="part_id")
    rows += _rows(k, ["source_bench", "test name", "api"], "proxy_perf", "median_result",
                  {"target": "GPU", "source": "KISHONTI", "vendor": "ALL"}, seed, n_boot)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    # 주 결과: (target, source, vendor)별로 모델 수가 가장 많은 조합 1개
    res["primary"] = False
    for _, g in res.groupby(["target", "source", "vendor"]):
        res.loc[g["n_models"].idxmax(), "primary"] = True
    return res.sort_values(["primary", "target", "source", "vendor", "n_models"], ascending=[False, True, True, True, False])


def main() -> None:
    cfg = load_config()
    parts = pd.read_csv(OUT / "parts.csv")
    blender = pd.read_csv(INTERIM / "perf_blender.csv")
    kishonti = pd.read_csv(INTERIM / "perf_kishonti.csv").rename(columns={"source": "source_bench"})
    res = run(parts, blender, kishonti, seed=int(cfg["seed"]))
    out = OUT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "d_validity.csv", index=False)
    cols = ["target", "source", "vendor", "benchmark_version", "scene", "device_type", "test name", "api",
            "n_models", "rho", "ci_low", "ci_high"]
    print(res[res["primary"]].reindex(columns=cols).to_string(index=False))
    print(f"부록 포함 전체 {len(res)}개 조합")


if __name__ == "__main__":
    main()
