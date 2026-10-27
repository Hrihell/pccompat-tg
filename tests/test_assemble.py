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


def test_uarch_todo_lists_wikipedia_cells_for_official_check(tmp_path, monkeypatch):
    """위키 칩셋 표에서 온 칸도 사람이 공식 페이지로 확인할 대상이다. 위키 값은 참고 열에만 두고 support는 비운다."""
    monkeypatch.setattr(assemble, "MANUAL", tmp_path)
    pd.DataFrame([{"family": "Ryzen 7000", "kind": "cpu", "uarch": "Zen 4"}]).to_csv(
        tmp_path / "family_facts.csv", index=False)
    parts = pd.DataFrame([{"part_id": "cpu-1", "family": "Ryzen 7000"}, {"part_id": "cpu-2", "family": "Ryzen 7000"}])
    chips = pd.DataFrame([{"chipset_id": "c-b650", "name": "B650"}])
    support = pd.DataFrame([
        {"cpu_id": c, "chipset_id": "c-b650", "support": "SUPPORTED", "origin": "WIKIPEDIA"} for c in ["cpu-1", "cpu-2"]])

    n = assemble.write_uarch_todo(support, parts, chips)
    n_again = assemble.write_uarch_todo(support, parts, chips)  # 다시 돌려도 참고 열이 겹치지 않아야 한다

    sheet = pd.read_csv(tmp_path / "chipset_support_by_uarch.csv", dtype=str, keep_default_na=False)
    assert n == n_again == 1
    row = sheet.iloc[0]
    assert (row["uarch"], row["chipset"], row["n_cpus"]) == ("Zen 4", "B650", "2")
    assert row["support"] == ""
    assert row["wiki_support"] == "SUPPORTED"
    assert not any(c.endswith(("_x", "_y")) for c in sheet.columns)


def test_official_eol_overrides_assumed_date_only_when_verified():
    chips = pd.DataFrame({"name": ["B550", "X570", "A520"],
                          "eol_date": pd.to_datetime(["2022-09-27"] * 3), "eol_is_assumed": [True, True, True]})
    sheet = pd.DataFrame([
        {"chipset": "B550", "eol_date": "2024-06-30", "source_url": "https://x", "verified": "true", "note": ""},
        {"chipset": "X570", "eol_date": "2023-01-01", "source_url": "https://y", "verified": "false", "note": ""}])

    out = assemble.apply_official_eol(chips, sheet)

    b550, x570, a520 = (out[out["name"] == n].iloc[0] for n in ("B550", "X570", "A520"))
    assert b550["eol_date"] == pd.Timestamp("2024-06-30") and not b550["eol_is_assumed"]
    assert b550["eol_assumed_date"] == pd.Timestamp("2022-09-27")  # 가정값은 오차 계산용으로 남긴다
    assert x570["eol_date"] == pd.Timestamp("2022-09-27") and x570["eol_is_assumed"]  # 미검증 행은 무시
    assert a520["eol_assumed_date"] == pd.Timestamp("2022-09-27")


def test_official_eol_sheet_with_only_unverified_rows_changes_nothing():
    chips = pd.DataFrame({"name": ["B460"], "eol_date": pd.to_datetime(["2021-01-01"]).astype("datetime64[s]"),
                          "eol_is_assumed": [True]})
    sheet = pd.DataFrame([{"chipset": "B460", "eol_date": "2021-12-24", "source_url": "https://x", "verified": "false",
                           "note": ""}])

    out = assemble.apply_official_eol(chips, sheet)

    assert out["eol_date"].iloc[0] == pd.Timestamp("2021-01-01") and bool(out["eol_is_assumed"].iloc[0])


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
