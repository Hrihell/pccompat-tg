import numpy as np
import pandas as pd
import pytest

from pccompat import paper_stats, validation_stats as vs


def _pairs(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """(label, cause, n, n_observed_pairs) → 분할 전 칩셋 기준 CPU–칩셋 쌍 표."""
    out = []
    for label, cause, n, k in rows:
        for i in range(n):
            out.append({"src": f"cpu-{label}-{cause}-{i}", "dst": "c", "label": label, "unknown_cause": cause or np.nan,
                        "evidence_type": "BOTH" if i < k else "DOCUMENTED", "basis_verified": True})
    return pd.DataFrame(out)


def test_holm_adjusted_matches_step_down_and_stays_monotone():
    adj = paper_stats.holm_adjusted(np.array([0.01, 0.04, 0.03]))
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert paper_stats.holm_significant(np.array([0.01, 0.04, 0.03])) == 1


def test_unknown_cause_tests_reports_rates_and_holm_adjusted_p():
    cc = _pairs([("PASS", "", 100, 30), ("FAIL", "", 200, 0), ("UNKNOWN", "BIOS_REQUIRED", 20, 7),
                 ("UNKNOWN", "BOARD_DEPENDENT", 20, 2), ("UNKNOWN", "MISSING", 10, 1)])

    n = vs.unknown_cause_tests(cc)

    assert (n["v_obs_bios_required_k"], n["v_obs_bios_required_n"], n["v_obs_bios_required_rate"]) == (7, 20, "35.0%")
    assert n["v_obs_pass_rate"] == "30.0%"
    assert (n["v_obs_pass_n"], n["v_obs_fail_n"]) == (100, 200)
    assert n["v_obs_fail_rate"] == "0.0%"
    assert vs._rate(2, 6142) == "0.03%"
    # BIOS 필요 vs 나머지 UNKNOWN(보드마다 다름+값 없음 3/30)도 같은 Holm 검정군(7개)에 넣는다
    assert (n["v_obs_unk_rest_k"], n["v_obs_unk_rest_n"], n["v_obs_unk_rest_rate"]) == (3, 30, "10.0%")
    assert float(n["v_obs_bios_vs_rest_p"]) <= 1.0
    assert n["v_obs_n_tests"] == 7
    for c in ("bios_required", "board_dependent", "missing"):
        assert float(n[f"v_obs_{c}_p_pass"]) <= 1.0
        assert {f"v_obs_{c}_p_fail_mant", f"v_obs_{c}_p_fail_exp"} <= n.keys()


def test_unknown_cause_tests_keeps_keys_when_a_cause_is_absent():
    cc = _pairs([("PASS", "", 10, 3), ("FAIL", "", 10, 0)])
    n = vs.unknown_cause_tests(cc)
    assert n["v_obs_missing_n"] == 0
    assert n["v_obs_missing_rate"] == "—"
    assert n["v_obs_missing_p_pass"] == "—"


def test_pass_basis_counts_pass_pairs_without_human_checked_source():
    cc = _pairs([("PASS", "", 4, 0)]).assign(basis_verified=[True, True, False, False])
    n = vs.pass_basis(cc)
    assert (n["v_pass_unverified_n"], n["v_pass_total"], n["v_pass_unverified_pct"]) == (2, 4, "50.0%")


def test_bios_date_check_compares_bios_build_date_with_cpu_release():
    cc = pd.DataFrame([
        {"src": "cpu-a", "dst": "chip-x", "label": "UNKNOWN", "unknown_cause": "BIOS_REQUIRED"},
        {"src": "cpu-b", "dst": "chip-x", "label": "PASS", "unknown_cause": np.nan}])
    parts = pd.DataFrame([{"part_id": "cpu-a", "release_date": "2022-01-01"}, {"part_id": "cpu-b", "release_date": "2022-01-01"}])
    chips = pd.DataFrame([{"chipset_id": "chip-x-ddr4", "base_chipset_id": "chip-x"}])
    combos = pd.DataFrame([
        {"chipset_id": "chip-x-ddr4", "cpu_id": "cpu-a", "bios_date": "2022-03-01"},
        {"chipset_id": "chip-x-ddr4", "cpu_id": "cpu-a", "bios_date": "2021-06-01"},
        {"chipset_id": "chip-x-ddr4", "cpu_id": "cpu-b", "bios_date": "2023-01-01"},
        {"chipset_id": "", "cpu_id": "cpu-b", "bios_date": "2023-01-01"}])  # 칩셋 모름 → 제외

    n = vs.bios_date_check(combos, cc, parts, chips)

    assert (n["v_bios_after_bios_k"], n["v_bios_after_bios_n"], n["v_bios_after_bios_rate"]) == (1, 2, "50.0%")
    assert n["v_bios_after_bios_pairs"] == 1  # 보고 2건은 같은 (CPU, 칩셋) 쌍
    assert (n["v_bios_after_pass_k"], n["v_bios_after_pass_n"]) == (1, 1)
    assert "v_bios_after_p" in n


def test_eol_stats_reports_official_coverage_and_error_months():
    chips = pd.DataFrame({
        "base_chipset_id": ["a", "a", "b", "c", "d"],
        "eol_date": ["2023-01-01", "2023-01-01", "2024-01-01", "2025-01-01", None],
        "eol_assumed_date": ["2022-07-01", "2022-07-01", "2024-01-01", "2025-01-01", None],  # d: 후속 세대 없음
        "eol_is_assumed": [False, False, True, True, None]})

    n = vs.eol_stats(chips)

    assert (n["v_eol_official_n"], n["v_eol_total"]) == (1, 3)
    assert n["v_eol_err_median_mo"] == "6.0"
