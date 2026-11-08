import numpy as np
import pandas as pd

from pccompat import paper_stats


def test_tier_tests_compare_cited_family_and_release_only():
    rows = ([("WIKIPEDIA_CITED_ROW", "cores", "true", "A")] * 6 + [("WIKIPEDIA_CITED_CELL", "release_date", "true", "A")] * 4
            + [("WIKIPEDIA_UNCITED", "cores", "true", "A")] * 5 + [("WIKIPEDIA_FAMILY", "release_date", "false", "Core 12th")] * 5
            + [("WIKIPEDIA_FAMILY", "release_date", "true", "Core 12th")] * 5 + [("OFFICIAL", "cores", "na", "A")])
    sample = pd.DataFrame(rows, columns=["source_tier", "field", "match", "family"])
    n = paper_stats.tier_tests(sample)
    assert (n["s_cited"], n["s_family"], n["s_rel_other"], n["s_rel_family"]) == ("10/10", "5/10", "4/4", "5/10")
    assert n["s_family_err_families"] == 1 and n["s_family_fields"] == 1
    assert float(n["s_p_cited_family"]) < 0.05 and float(n["s_p_cited_uncited"]) == 1.0


def test_same_socket_drops_socket_mismatch_and_dedups_gpu():
    t = pd.Timestamp("2022-01-01")
    b = pd.DataFrame({"cpu_id": ["c1", "c1", "c2", "c3"], "chipset_id": ["h1"] * 4, "ram_id": ["r1"] * 4,
                      "label": ["PASS", "PASS", "FAIL", "UNKNOWN"], "failed_rule_mask": pd.Series([0, 0, 1, 0], dtype="int8"),
                      "valid_from": [t, t, t, None]})
    n = paper_stats.same_socket(b)
    assert n["x_socket_fail"] == 1 and n["x_socket_fail_share"] == "100.0%"
    assert (n["x_same_triples"], n["x_same_triples_pass"], n["x_same_triples_unknown"]) == (2, 1, 1)
    assert n["x_same_coexist"] == 2 and n["x_same_coexist_pass_pct"] == "100.0%"


def test_holm_counts_only_adjusted_significant():
    assert paper_stats.holm_significant(np.array([0.001, 0.03, 0.04])) == 1


def test_observed_tests_formats_tiny_p_as_mantissa_exponent():
    ct = pd.DataFrame({"BOTH": [59, 345, 2], "All": [253, 1340, 6595]}, index=["UNKNOWN", "PASS", "FAIL"])
    n = paper_stats.observed_tests(ct)
    assert n["x_obs_p_unknown_pass"] == "0.43" and int(n["x_obs_p_unknown_fail_exp"]) < -80
    assert (n["x_obs_p_unknown_pass_mant"], n["x_obs_p_unknown_pass_exp"]) == ("4.3", "-1")


def _split_fixture():
    t = pd.Timestamp("2022-01-01")
    # h2를 보드 메모리로 나눈 변형 h2-ddr4/h2-ddr5, 램은 DDR4(r4). c1은 칩셋 지원 PASS, c2는 칩셋 지원 UNKNOWN.
    b = pd.DataFrame({"cpu_id": ["c1", "c1", "c1", "c2", "c2"], "chipset_id": ["h1", "h2-ddr4", "h2-ddr5", "h2-ddr4", "h2-ddr5"],
                      "ram_id": ["r4"] * 5, "gpu_id": ["g"] * 5,
                      "label": ["PASS", "PASS", "FAIL", "UNKNOWN", "FAIL"], "failed_rule_mask": pd.Series([0] * 5, dtype="int8"),
                      "valid_from": [t, t, t, None, None]})
    chips = pd.DataFrame({"chipset_id": ["h1", "h2-ddr4", "h2-ddr5"], "base_chipset_id": ["h1", "h2", "h2"],
                          "board_memory": [None, "DDR4", "DDR5"]})
    rams = pd.DataFrame({"ram_id": ["r4"], "ddr_gen": ["DDR4"]})
    return b, chips, rams


def test_chipset_level_collapses_variants_back_to_board_dependent_unknown():
    b, chips, rams = _split_fixture()
    n = paper_stats.board_memory_stats(b, chips, rams)
    # 칩셋 단위로 되돌리면 (c1,h1)=PASS, (c1,h2)=PASS·FAIL 섞임→UNKNOWN, (c2,h2)=UNKNOWN·FAIL→UNKNOWN
    assert n["x_chiplevel_same_unknown_pct"] == "66.7%"
    assert n["x_chiplevel_coexist_unknown_pct"] == "50.0%"      # 공존: (c1,h1) PASS, (c1,h2) UNKNOWN


def test_memory_matched_unit_drops_board_memory_mismatch():
    b, chips, rams = _split_fixture()
    n = paper_stats.board_memory_stats(b, chips, rams)
    # 보드 메모리와 램 세대가 다른 변형(h2-ddr5 + DDR4)은 뻔한 FAIL이라 뺀다 → 공존 2개: PASS 2
    assert n["x_memmatch_coexist"] == 2 and n["x_memmatch_coexist_pass_pct"] == "100.0%"
    assert n["x_memmatch_triples"] == 3 and n["x_memmatch_triples_unknown"] == 1
    assert n["x_memmismatch_builds"] == 2

