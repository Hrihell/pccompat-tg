import pandas as pd

from pccompat import gpu_platform as platform


def _facts(rows):
    cols = ["kind", "key", "gpu_pcie_gen", "gpu_pcie_lanes", "source_url", "verified", "note"]
    return platform._verified(pd.DataFrame(rows, columns=cols))


def test_warning_tables_use_only_verified_facts_and_model_overrides():
    cpus = pd.DataFrame({"part_id": ["cpu:x", "cpu:apu"], "family": ["R5", "R5"]})
    chips = pd.DataFrame({"chipset_id": ["h", "k"], "name": ["A520", "B550"], "base_name": ["A520", "B550"]})
    gpus = pd.DataFrame({"part_id": ["gpu:narrow", "gpu:wide"], "family": ["RX 6000", "RX 6000"],
                         "pcie_gen": [4.0, 4.0], "pcie_lanes": [4.0, 16.0]})
    facts = _facts([
        ["cpu_family", "R5", "4", "16", "https://amd.example/r5", "true", ""],
        ["cpu_model", "cpu:apu", "3", "", "https://amd.example/apu", "true", ""],   # 모델 행이 세대 행보다 우선
        ["chipset", "A520", "3", "", "https://amd.example/a520", "true", ""],
        ["chipset", "B550", "4", "", "", "true", ""],                               # URL 없음 → 쓰지 않음
    ])
    warn, unchecked = platform.warning_tables(cpus, chips, gpus, facts)
    # x4 GPU를 PCIe 3.0 슬롯(A520)에: 대역폭 경고, x16 GPU는 경고 없음
    assert warn[0, 0, 0] == 1 and warn[1, 0, 0] == 1 and warn[0, 0, 1] == 0
    # 출처 없는 B550 행은 쓰지 않으므로 판단 보류
    assert unchecked[0, 1, 0] == 1 and warn[0, 1, 0] == 0


def test_unverified_sheet_leaves_everything_unchecked():
    cpus = pd.DataFrame({"part_id": ["cpu:x"], "family": ["R5"]})
    chips = pd.DataFrame({"chipset_id": ["h"], "name": ["B550"], "base_name": ["B550"]})
    gpus = pd.DataFrame({"part_id": ["g"], "family": ["Arc A"], "pcie_gen": [4.0], "pcie_lanes": [8.0]})
    facts = _facts([["cpu_family", "R5", "4", "16", "https://x", "false", ""]])
    warn, unchecked = platform.warning_tables(cpus, chips, gpus, facts)
    assert warn.sum() == 0 and unchecked[0, 0, 0] == 1
