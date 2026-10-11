import pandas as pd

from pccompat.config import load_config
from pccompat.quality import (accuracy_by_tier, cross_source_agreement, label_balance, leakage, missingness, outliers,
                              verification_sample, wilson_interval)


def _parts():
    return pd.DataFrame({
        "part_id": [f"cpu:x:{i}" for i in range(6)] + ["gpu:nvidia:geforce-rtx-4070", "gpu:nvidia:geforce-rtx-4060"],
        "category": ["CPU"] * 6 + ["GPU"] * 2, "vendor": ["AMD"] * 6 + ["NVIDIA"] * 2,
        "family": ["A"] * 3 + ["B"] * 3 + ["G"] * 2,
        "model_name": [f"Model {i}" for i in range(6)] + ["GeForce RTX 4070", "GeForce RTX 4060"],
        "cores": [8, 8, 8, 6, 6, 6, None, None], "threads": [16] * 6 + [None] * 2,
        "boost_clock_mhz": [5000] * 6 + [2475, 2460], "tdp_w": [65] * 6 + [200, 115],
        "l3_cache_mb": [32] * 6 + [None] * 2, "shader_count": [None] * 6 + [5888, 3072],
        "fp32_tflops": [None] * 6 + [29.15, 15.11],
        "release_date": ["2022-01-01"] * 3 + [None] * 3 + ["2023-04-13", "2023-06-29"],
        "msrp_usd": [300] * 6 + [599, 299],
    })


def test_missing_by_family_is_detected_as_not_mcar():
    rates, tests = missingness(_parts())
    assert rates.loc["B", "release_date"] == 1.0
    assert tests.loc["cores", "missing_rate"] == 0          # GPU 행은 '해당 없음'이라 결측으로 세지 않음


def test_outliers_flag_only():
    p = _parts()
    p.loc[0, "boost_clock_mhz"] = 49000
    flagged = outliers(p, load_config()["ranges"])
    assert "cpu:x:0" in set(flagged["part_id"])


def test_cross_source_agreement_counts_mismatch():
    other = pd.DataFrame({"name": ["NVIDIA GeForce RTX 4070", "NVIDIA GeForce RTX 4060"],
                          "tdp_w": [200, 120], "shader_cores": [5888, 3072],
                          "release_date": ["2023-04-13", "2023-06-29"], "fp32_dense_tflops": [29.1, 15.1]})
    summary, detail = cross_source_agreement(
        _parts(), other, {"tdp_w": "tdp_w", "shader_count": "shader_cores"}, load_config()["matching"])
    assert summary.loc["tdp_w", "agreement"] == 0.5
    assert summary.loc["shader_count", "agreement"] == 1.0
    assert list(detail["part_id"]) == ["gpu:nvidia:geforce-rtx-4060"]


def test_leakage_reports_unassigned_rows():
    rep = leakage(_parts(), load_config())
    assert rep.loc["분할 미배정 행", "detail"] == "3"
    assert bool(rep.loc["미래 정보 열이 특징에 없음", "ok"])


def test_verification_sample_is_stratified_and_reproducible():
    p = _parts()
    prov = pd.DataFrame([{"entity_id": r.part_id, "field": f, "value": 1, "source_url": "u", "source_tier": "WIKIPEDIA_UNCITED"}
                         for r in p.itertuples() for f in ["cores", "tdp_w", "msrp_usd"]])
    prov.loc[:3, "source_tier"] = "CELL"
    a = verification_sample(prov, p, n=8, seed=1, min_per_tier=3)
    b = verification_sample(prov, p, n=8, seed=1, min_per_tier=3)
    assert len(a) == 8 and a.equals(b)
    assert (a["source_tier"].value_counts() >= 3).all()          # 등급마다 최소 건수
    assert a["field"].nunique() == 3
    assert {"official_value", "official_url", "match"} <= set(a.columns)


def test_wilson_interval_and_accuracy_by_tier():
    lo, hi = wilson_interval(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0
    checked = pd.DataFrame({"source_tier": ["A"] * 4 + ["B"] * 2, "match": ["true", "true", "false", "true", "true", ""]})
    rep = accuracy_by_tier(checked).set_index("source_tier")
    assert rep.loc["A", "n"] == 4 and rep.loc["A", "accuracy"] == 0.75
    assert rep.loc["B", "n"] == 1                     # 빈 칸은 아직 검증 안 한 행


def test_label_balance_flags_rare_classes_per_unit():
    units = {"쌍(CPU–칩셋)": pd.Series(["FAIL"] * 90 + ["PASS"] * 8 + ["UNKNOWN"] * 2),
             "조합": pd.Series(["PASS"] * 40 + ["UNKNOWN"] * 30 + ["FAIL"] * 30)}
    t = label_balance(units).set_index("unit")
    assert t.loc["쌍(CPU–칩셋)", "UNKNOWN"] == 2 and t.loc["쌍(CPU–칩셋)", "minority_share"] == 0.02
    assert t.loc["쌍(CPU–칩셋)", "suggestion"] != ""          # 소수 클래스 2% → 평가 시 주의
    assert t.loc["조합", "suggestion"] == ""


def test_gpu_bus_fields_are_not_missing_for_cpus():
    from pccompat.quality import _applicable
    parts = pd.DataFrame({"category": ["CPU", "GPU"]})
    assert _applicable(parts, "pcie_gen").tolist() == [False, True]
