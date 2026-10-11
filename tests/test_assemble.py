import pandas as pd

from pccompat import assemble


def test_uarch_todo_keeps_rows_the_user_already_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "MANUAL", tmp_path)
    pd.DataFrame([{"family": "Core 12th", "kind": "cpu", "uarch": "Alder Lake"}]).to_csv(
        tmp_path / "family_facts.csv", index=False)
    pd.DataFrame([{"uarch": "Alder Lake", "chipset": "B660", "n_cpus": 1, "support": "SUPPORTED",
                   "source_url": "", "verified": "false", "note": "사용자 확인"}]).to_csv(
        tmp_path / "chipset_support_by_uarch.csv", index=False)
    parts = pd.DataFrame([{"part_id": "cpu-1", "family": "Core 12th"}])
    chips = pd.DataFrame([{"chipset_id": "c-b660", "name": "B660"}, {"chipset_id": "c-b760", "name": "B760"}])
    # B660은 시트 값이 반영된 상태(MANUAL_UARCH), B760은 아직 빈칸(NOT_LISTED)
    support = pd.DataFrame([
        {"cpu_id": "cpu-1", "chipset_id": "c-b660", "support": "SUPPORTED", "origin": "MANUAL_UARCH"},
        {"cpu_id": "cpu-1", "chipset_id": "c-b760", "support": "NOT_LISTED", "origin": "NONE"},
    ])

    n = assemble.write_uarch_todo(support, parts, chips)

    sheet = pd.read_csv(tmp_path / "chipset_support_by_uarch.csv", dtype=str, keep_default_na=False)
    assert n == 2
    b660 = sheet[sheet["chipset"] == "B660"].iloc[0]
    assert (b660["support"], b660["note"]) == ("SUPPORTED", "사용자 확인")
    assert sheet.loc[sheet["chipset"] == "B760", "support"].iloc[0] == ""


def test_per_cpu_sheet_false_string_becomes_false(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "MANUAL", tmp_path)
    monkeypatch.setattr(assemble, "wiki_support_matrix", lambda: pd.DataFrame(columns=["uarch", "chipset", "support", "url", "support_text"]))
    pd.DataFrame([{"family": "Core Ultra 200S", "kind": "cpu", "uarch": "Arrow Lake"}]).to_csv(tmp_path / "family_facts.csv", index=False)
    pd.DataFrame([{"cpu_id": "cpu-1", "chipset_id": "c-z890", "support": "NOT_LISTED", "source_url": "u",
                   "verified": "false", "note": ""}]).to_csv(tmp_path / "cpu_chipset_support.csv", index=False)
    parts = pd.DataFrame([{"part_id": "cpu-1", "category": "CPU", "family": "Core Ultra 200S", "socket": "LGA 1851"}])
    chips = pd.DataFrame([{"chipset_id": "c-z890", "name": "Z890", "socket": "LGA 1851"}])

    sup = assemble.build_support(parts, chips)

    assert sup.loc[sup["cpu_id"] == "cpu-1", "verified"].tolist() == [False]


def test_excluded_chipsets_accepts_one_rule_or_a_list_of_rules():
    chips = pd.DataFrame({"vendor": ["INTEL", "INTEL", "AMD", "AMD", "AMD"],
                          "name": ["H610E", "H610", "A300", "X300", "B550"]})
    one = assemble.excluded_chipsets(chips, {"vendor": "INTEL", "pattern": r"^[A-Z]\d{3}E$"})
    assert one.tolist() == [True, False, False, False, False]
    # APU 전용 미니 PC 플랫폼(외장 GPU 슬롯 없음)은 AMD에만 적용, B550 같은 일반 칩셋은 남는다
    many = assemble.excluded_chipsets(chips, [{"vendor": "INTEL", "pattern": r"^[A-Z]\d{3}E$"},
                                              {"vendor": "AMD", "pattern": r"^[AX]300$"}])
    assert many.tolist() == [True, False, True, True, False]
    assert not assemble.excluded_chipsets(chips, None).any()


def test_split_board_memory_makes_one_node_per_memory_type_only_for_dual_chipsets():
    # 칩셋 단위에서 DDR4·DDR5를 모두 지원하는 칩셋(B760)만 보드 메모리별 노드 두 개로 나눈다. B650(DDR5 전용)은 그대로.
    chips = pd.DataFrame({"chipset_id": ["chipset:intel:b760", "chipset:amd:b650"], "name": ["B760", "B650"],
                          "memory_gen_options": [("DDR4", "DDR5"), ("DDR5",)], "socket": ["LGA1700", "AM5"]})
    support = pd.DataFrame({"cpu_id": ["cpu:a", "cpu:b"], "chipset_id": ["chipset:intel:b760", "chipset:amd:b650"],
                            "support": ["SUPPORTED", "SUPPORTED"]})
    c2, s2 = assemble.split_board_memory(chips, support)
    assert c2["chipset_id"].tolist() == ["chipset:intel:b760-ddr4", "chipset:intel:b760-ddr5", "chipset:amd:b650"]
    assert c2["name"].tolist() == ["B760 (DDR4)", "B760 (DDR5)", "B650"]
    assert c2["memory_gen_options"].tolist() == [("DDR4",), ("DDR5",), ("DDR5",)]
    assert c2["base_chipset_id"].tolist() == ["chipset:intel:b760", "chipset:intel:b760", "chipset:amd:b650"]
    assert c2["base_name"].tolist() == ["B760", "B760", "B650"]
    assert c2["board_memory"].isna().tolist() == [False, False, True]
    # CPU–칩셋 지원은 보드 메모리와 무관하므로 두 변형에 그대로 복제
    assert sorted(s2["chipset_id"]) == ["chipset:amd:b650", "chipset:intel:b760-ddr4", "chipset:intel:b760-ddr5"]
    assert set(s2.loc[s2["cpu_id"] == "cpu:a", "support"]) == {"SUPPORTED"}
