import pandas as pd

from pccompat.observed import apply_to_pairs, chipset_from_board, pair_observations, parse_report

REPORT = """BIOS Information
\tVendor: AMI
\tVersion: 7.02
\tRelease Date: 06/08/2023

Base Board Information
\tManufacturer: ASRock
\tProduct Name: Z790 PG Lightning
\tSerial Number: --

Processor Information
\tVersion: 13th Gen Intel(R) Core(TM) i7-13700K

Memory Device
\tSize: No Module Installed
\tType: Unknown

Memory Device
\tSize: 32 GB
\tType: DDR5
"""


def test_parse_report_skips_empty_slots_and_identifiers():
    r = parse_report(REPORT)
    assert (r["board_model"], r["bios_version"], r["ram_gens"]) == ("Z790 PG Lightning", "7.02", {"DDR5"})
    assert "Serial" not in str(r)


def test_chipset_from_board_prefers_longest_and_rejects_ambiguous():
    names = ["B650", "B650E", "X670", "Z790"]
    assert chipset_from_board("MAG B650E TOMAHAWK", names) == "B650E"
    assert chipset_from_board("PRO Z790-P WIFI", names) == "Z790"
    assert chipset_from_board("TUF GAMING", names) is None
    assert chipset_from_board("B650 X670 combo", names) is None


def test_pairs_get_both_evidence_when_observed():
    obs = pd.DataFrame([{"cpu_id": "cpu:a", "chipset_id": "chipset:z790", "ram_gen": "DDR5", "n_reports": 3}])
    pobs = pair_observations(obs)
    pairs = pd.DataFrame([
        {"edge_type": "CPU_CHIPSET", "src": "cpu:a", "dst": "chipset:z790", "evidence_type": "DOCUMENTED"},
        {"edge_type": "CPU_RAM", "src": "cpu:a", "dst": "ram:ddr5:32gb", "evidence_type": "DOCUMENTED"},
        {"edge_type": "CPU_RAM", "src": "cpu:a", "dst": "ram:ddr4:32gb", "evidence_type": "DOCUMENTED"},
    ])
    rams = pd.DataFrame({"ram_id": ["ram:ddr5:32gb", "ram:ddr4:32gb"], "ddr_gen": ["DDR5", "DDR4"]})
    out = apply_to_pairs(pairs, rams, pobs)
    assert out["evidence_type"].tolist() == ["BOTH", "BOTH", "DOCUMENTED"]
    assert out["n_observed"].tolist()[:2] == [3, 3]


def test_pair_observations_resolve_board_memory_variant_from_ram_generation():
    from pccompat.observed import pair_observations
    chips = pd.DataFrame({"chipset_id": ["chipset:intel:b760-ddr4", "chipset:intel:b760-ddr5", "chipset:amd:b650"],
                          "base_chipset_id": ["chipset:intel:b760", "chipset:intel:b760", "chipset:amd:b650"],
                          "board_memory": ["DDR4", "DDR5", None]})
    obs = pd.DataFrame({"cpu_id": ["cpu:a", "cpu:a", "cpu:a", "cpu:b"],
                        "chipset_id": ["chipset:intel:b760", "chipset:intel:b760", "chipset:intel:b760", "chipset:amd:b650"],
                        "ram_gen": ["DDR4", "DDR5", None, "DDR5"], "n_reports": [3, 5, 7, 2]})
    p = pair_observations(obs, chips).set_index(["edge_type", "src", "dst"])["n_observed"].to_dict()
    # 램 세대로 어느 보드인지 알 수 있을 때만 변형에 붙인다(램 정보 없는 7건은 어느 보드인지 몰라 버린다)
    assert p[("CPU_CHIPSET", "cpu:a", "chipset:intel:b760-ddr4")] == 3
    assert p[("CPU_CHIPSET", "cpu:a", "chipset:intel:b760-ddr5")] == 5
    assert p[("CHIPSET_RAMGEN", "chipset:intel:b760-ddr5", "DDR5")] == 5
    assert p[("CPU_CHIPSET", "cpu:b", "chipset:amd:b650")] == 2
    assert not any(k[2] == "chipset:intel:b760" for k in p)
