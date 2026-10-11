import numpy as np
import pandas as pd

from pccompat.analysis import months, official_vs_observed, pareto_mask, upgrade_window


def test_pareto_mask_hand_example():
    price = np.array([100, 120, 100, 200])
    cpu = np.array([10, 12, 9, 12])
    gpu = np.array([5, 5, 5, 5])
    # 0: 비지배, 1: 비지배(더 비싸지만 CPU 더 높음), 2: 0에 지배(같은 가격·더 낮은 CPU), 3: 1에 지배
    assert pareto_mask(price, cpu, gpu).tolist() == [True, True, False, False]


def test_pareto_ties_are_kept():
    assert pareto_mask(np.array([1, 1]), np.array([1, 1]), np.array([1, 1])).tolist() == [True, True]


def test_months_inclusive():
    m = months("2020-10-01", "2020-12-31")
    assert [d.month for d in m] == [10, 11, 12]


def test_upgrade_window_counts_and_span():
    parts = pd.DataFrame({"part_id": ["c1", "c2", "c3"], "release_date": ["2022-01-01", "2024-01-01", "2023-01-01"],
                          "family": ["F1", "F2", "F2"]})
    chips = pd.DataFrame({"chipset_id": ["h"], "name": ["X"], "family": ["S"], "release_date": ["2021-01-01"], "eol_date": [None]})
    pairs = pd.DataFrame([
        {"edge_type": "CPU_CHIPSET", "src": "c1", "dst": "h", "label": "PASS", "unknown_cause": None},
        {"edge_type": "CPU_CHIPSET", "src": "c2", "dst": "h", "label": "PASS", "unknown_cause": None},
        {"edge_type": "CPU_CHIPSET", "src": "c3", "dst": "h", "label": "UNKNOWN", "unknown_cause": "BIOS_REQUIRED"},
    ])
    u = upgrade_window(pairs, parts, chips).iloc[0]
    assert (u["n_cpu_pass"], u["n_cpu_unknown_board_or_bios"], u["n_cpu_families_pass"]) == (2, 1, 2)
    assert 23.5 < u["span_months"] < 24.5


def test_official_vs_observed_totals():
    pairs = pd.DataFrame({"edge_type": ["CPU_CHIPSET"] * 3, "label": ["PASS", "FAIL", "UNKNOWN"],
                          "evidence_type": ["BOTH", "BOTH", "DOCUMENTED"], "n_observed": [5, 1, None],
                          "src": list("abc"), "dst": list("xyz"), "evidence": ["{}"] * 3, "unknown_cause": [None, None, "MISSING"]})
    table, fail_obs, resolve = official_vs_observed(pairs)
    assert table.loc["All", "All"] == 3 and len(fail_obs) == 1 and resolve.empty


def test_upgrade_window_counts_a_split_chipset_once():
    # 보드 메모리 변형(B760 DDR4/DDR5)은 CPU 지원이 같으므로 업그레이드 집계에서는 칩셋 하나로 센다
    parts = pd.DataFrame({"part_id": ["c1"], "release_date": ["2023-01-01"], "family": ["F1"]})
    chips = pd.DataFrame({"chipset_id": ["h-ddr4", "h-ddr5"], "name": ["B760 (DDR4)", "B760 (DDR5)"],
                          "base_chipset_id": ["h", "h"], "base_name": ["B760", "B760"], "family": ["S", "S"],
                          "release_date": ["2023-01-01"] * 2, "eol_date": [None] * 2})
    pairs = pd.DataFrame([{"edge_type": "CPU_CHIPSET", "src": "c1", "dst": d, "label": "PASS", "unknown_cause": None}
                          for d in ["h-ddr4", "h-ddr5"]])
    u = upgrade_window(pairs, parts, chips)
    assert len(u) == 1 and u.iloc[0]["name"] == "B760"


def test_official_vs_observed_counts_split_chipset_pairs_once():
    # 변형(h-ddr4/h-ddr5)의 CPU–칩셋 쌍은 라벨이 같으므로 칩셋 하나로 합치고 관찰 수는 더한다
    pairs = pd.DataFrame({"edge_type": ["CPU_CHIPSET"] * 3, "label": ["PASS"] * 3,
                          "evidence_type": ["BOTH", "DOCUMENTED", "DOCUMENTED"], "n_observed": [4, None, None],
                          "src": ["c", "c", "c"], "dst": ["h-ddr4", "h-ddr5", "k"], "evidence": ["{}"] * 3,
                          "unknown_cause": [None] * 3})
    chips = pd.DataFrame({"chipset_id": ["h-ddr4", "h-ddr5", "k"], "base_chipset_id": ["h", "h", "k"]})
    table, _, _ = official_vs_observed(pairs, chips)
    assert table.loc["PASS", "All"] == 2 and table.loc["PASS", "BOTH"] == 1
