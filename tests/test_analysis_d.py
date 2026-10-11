import numpy as np
import pandas as pd

from pccompat.analysis_d import MIN_MODELS, run, spearman_ci


def test_perfect_monotone_gives_rho_one_and_reproducible_ci():
    x = np.arange(20, dtype=float)
    a = spearman_ci(x, x ** 2, 200, seed=1)
    b = spearman_ci(x, x ** 2, 200, seed=1)
    assert a[0] == 1.0 and a == b


def test_backends_not_mixed_and_small_groups_skipped():
    n = MIN_MODELS + 2
    parts = pd.DataFrame({"part_id": [f"g{i}" for i in range(n)] + [f"a{i}" for i in range(3)],
                          "category": "GPU", "vendor": ["NVIDIA"] * n + ["AMD"] * 3,
                          "proxy_perf": list(range(n)) + [1, 2, 3]})
    blender = pd.DataFrame({"part_id": parts["part_id"], "device_type": ["OPTIX"] * n + ["HIP"] * 3,
                            "benchmark_version": "4.2.0", "scene": "monster",
                            "median_samples_per_min": list(range(n)) + [3, 2, 1], "low_n_flag": False})
    kish = pd.DataFrame(columns=["part_id", "source_bench", "test name", "api", "median_result"])
    res = run(parts, blender, kish, seed=1, n_boot=50)
    assert set(res["vendor"]) == {"NVIDIA"}          # AMD는 모델 3개라 제외
    assert res.iloc[0]["rho"] == 1.0
