import pandas as pd

from pccompat import labels, rules
from pccompat.config import load_config
from pccompat.labels import CAUSES, build_labels, decode_mask


def test_build_label_is_worst_pair_and_masks_decode():
    cfg = load_config()
    parts = pd.DataFrame({
        "part_id": ["cpu:a", "gpu:g"], "category": ["CPU", "GPU"], "family": ["F", "G"],
        "release_date": ["2023-01-01", "2023-06-01"], "eol_date": [None, None], "msrp_usd": [300, 500],
    })
    chips = pd.DataFrame({"chipset_id": ["chipset:x", "chipset:y"], "family": ["S", "S"],
                          "release_date": ["2022-01-01", "2022-01-01"], "eol_date": [None, None]})
    rams = pd.DataFrame({"ram_id": ["ram:ddr5:32gb"], "ddr_gen": ["DDR5"], "total_capacity_gb": [32]})
    pairs = pd.DataFrame([
        {"edge_type": "CPU_CHIPSET", "src": "cpu:a", "dst": "chipset:x", "label": "PASS", "unknown_cause": None, "rule_ids": "HC-001;HC-003"},
        {"edge_type": "CPU_CHIPSET", "src": "cpu:a", "dst": "chipset:y", "label": "FAIL", "unknown_cause": None, "rule_ids": "HC-001"},
        {"edge_type": "CPU_RAM", "src": "cpu:a", "dst": "ram:ddr5:32gb", "label": "PASS", "unknown_cause": None, "rule_ids": "HC-004;HC-005"},
        {"edge_type": "CHIPSET_RAM", "src": "chipset:x", "dst": "ram:ddr5:32gb", "label": "UNKNOWN", "unknown_cause": "BOARD_DEPENDENT", "rule_ids": "HC-004"},
        {"edge_type": "CHIPSET_RAM", "src": "chipset:y", "dst": "ram:ddr5:32gb", "label": "PASS", "unknown_cause": None, "rule_ids": "HC-004"},
    ])
    # GPU 연결 경고 배열: 두 칩셋 모두 경고(1)로 줘도 FAIL 조합(chipset:y, 소켓 불일치)에는 붙지 않아야 한다
    import numpy as np
    ones = np.ones((1, 2, 1), np.int8)
    b = build_labels(parts, chips, rams, pairs, {"cpu:a": "train", "gpu:g": "train", "chipset:x": "train", "chipset:y": "train"},
                     cfg, (ones, ones))
    b = b.set_index(b["chipset_id"].astype(str))
    assert b.loc["chipset:x", "warn_mask"] == 1 and b.loc["chipset:y", "warn_mask"] == 0
    assert b.loc["chipset:y", "warn_unchecked_mask"] == 0
    assert b.loc["chipset:x", "label"] == "UNKNOWN"
    assert decode_mask(int(b.loc["chipset:x", "unknown_cause_mask"]), CAUSES) == ["BOARD_DEPENDENT"]
    assert b.loc["chipset:y", "label"] == "FAIL" and b.loc["chipset:y", "difficulty"] == "EASY"
    assert b.loc["chipset:x", "valid_from"] == pd.Timestamp("2023-06-01")       # 세 노드 출시일 중 가장 늦은 날
    assert b.loc["chipset:x", "total_msrp_usd"] == 800


def test_pair_row_never_keeps_an_inverted_interval():
    # 단종일이 출시일보다 앞선 노드의 구간을 그대로 복사하면 valid_from > valid_to 쌍이 생긴다 → 구간 없음으로 둔다
    v = rules.Verdict("PASS", None, ("HC-004",), {})
    row = labels._row("CPU_RAM", "cpu:a", "ram:b", v, pd.Timestamp("2024-01-31"), pd.Timestamp("2022-09-27"), False)
    assert row["valid_from"] is None and row["valid_to"] is None
    ok = labels._row("CPU_RAM", "cpu:a", "ram:b", v, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-09-27"), False)
    assert (ok["valid_from"], ok["valid_to"]) == ("2022-01-01", "2022-09-27")


def test_cpu_chipset_basis_is_verified_only_for_socket_fail_or_checked_support(monkeypatch):
    """같은 소켓의 FAIL(미지원)은 지원 근거가 검증됐을 때만 basis_verified. 소켓 불일치 FAIL은 지원 근거가 필요 없다."""
    from pccompat import labels
    monkeypatch.setattr(labels, "_verified_families", lambda: {"cpuF", "chipF"})
    parts = pd.DataFrame([{"part_id": "cpu-1", "category": "CPU", "socket": "LGA1200", "family": "cpuF",
                           "release_date": "2021-03-01", "eol_date": None, "memory_gens": "DDR4", "max_memory_gb": 128}])
    chips = pd.DataFrame([
        {"chipset_id": "b460", "socket": "LGA1200", "family": "chipF", "release_date": "2020-05-01", "eol_date": None},
        {"chipset_id": "b560", "socket": "LGA1200", "family": "chipF", "release_date": "2021-03-01", "eol_date": None},
        {"chipset_id": "x570", "socket": "AM4", "family": "chipF", "release_date": "2019-07-01", "eol_date": None}])
    support = pd.DataFrame([{"cpu_id": "cpu-1", "chipset_id": "b460", "support": "UNSUPPORTED", "verified": False},
                            {"cpu_id": "cpu-1", "chipset_id": "b560", "support": "UNSUPPORTED", "verified": True}])
    rams = pd.DataFrame(columns=["ram_id", "ddr_gen", "total_capacity_gb"])

    out = labels.pair_labels(parts, chips, rams, support, {"window": {"end": "2026-10-01"}})

    got = out[out["edge_type"] == "CPU_CHIPSET"].set_index("dst")[["label", "basis_verified"]]
    assert tuple(got.loc["b460"]) == ("FAIL", False)  # 위키 근거 미지원
    assert tuple(got.loc["b560"]) == ("FAIL", True)
    assert tuple(got.loc["x570"]) == ("FAIL", True)   # 소켓 불일치
