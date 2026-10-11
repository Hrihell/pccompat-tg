import pandas as pd

from pccompat import labels, provenance


def _facts(tmp_path):
    pd.DataFrame([{"family": "Ryzen 5000", "kind": "cpu", "socket": "AM4", "memory_gens": "DDR4", "uarch": "Zen 3",
                   "max_memory_gb": "128", "source_url": "https://www.amd.com/x", "verified": "true",
                   "verification": "human", "note": "",
                   "max_memory_source_url": "https://thirdparty.example/x"}]).to_csv(tmp_path / "family_facts.csv", index=False)


def test_max_memory_from_other_site_is_not_official(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "MANUAL", tmp_path)
    _facts(tmp_path)
    parts = pd.DataFrame([{"part_id": "cpu-1", "family": "Ryzen 5000"}])
    rows = {r["field"]: r for r in provenance._facts_rows(parts, pd.DataFrame(columns=["family", "chipset_id"]))}
    assert rows["socket"]["source_tier"] == "OFFICIAL" and rows["socket"]["verified"] is True
    assert rows["max_memory_gb"]["source_tier"] == "OTHER" and rows["max_memory_gb"]["verified"] is False
    assert rows["max_memory_gb"]["source_url"] == "https://thirdparty.example/x"


def test_family_with_unofficial_max_memory_is_not_fully_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "MANUAL", tmp_path)
    _facts(tmp_path)
    assert labels._verified_families() == set()


def test_manual_release_row_keeps_given_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "MANUAL", tmp_path)
    pd.DataFrame([
        {"part_id": "cpu-a", "model_name": "A", "family": "F", "release_date": "2021-03-01", "release_date_precision": "MONTH",
         "source_url": "https://en.wikipedia.org/w/index.php?oldid=1", "note": "", "source_tier": "WIKIPEDIA_UNCITED"},
        {"part_id": "cpu-b", "model_name": "B", "family": "F", "release_date": "2022-01-01", "release_date_precision": "QUARTER",
         "source_url": "https://www.intel.com/x", "note": "", "source_tier": ""},
    ]).to_csv(tmp_path / "release_dates_todo.csv", index=False)
    rows = {r["entity_id"]: r for r in provenance._release_rows()}
    assert (rows["cpu-a"]["source_tier"], rows["cpu-a"]["verified"]) == ("WIKIPEDIA_UNCITED", False)
    assert (rows["cpu-b"]["source_tier"], rows["cpu-b"]["verified"]) == ("OFFICIAL", True)


def test_variant_chipsets_inherit_base_provenance_with_their_own_memory_value():
    chips = pd.DataFrame({"chipset_id": ["chipset:intel:b760-ddr4", "chipset:intel:b760-ddr5", "chipset:amd:b650"],
                          "base_chipset_id": ["chipset:intel:b760", "chipset:intel:b760", "chipset:amd:b650"],
                          "board_memory": ["DDR4", "DDR5", None]})
    prov = pd.DataFrame({"entity_id": ["chipset:intel:b760", "chipset:intel:b760-ddr4", "chipset:intel:b760-ddr5", "chipset:amd:b650"],
                         "field": ["release_date", "memory_gen_options", "memory_gen_options", "release_date"],
                         "value": ["2023-01-03", "DDR4;DDR5", "DDR4;DDR5", "2022-10-01"], "source_url": ["w", "f", "f", "w"]})
    out = provenance.expand_to_variants(prov, chips)
    rel = out[out["field"] == "release_date"].set_index("entity_id")["value"].to_dict()
    assert rel == {"chipset:intel:b760-ddr4": "2023-01-03", "chipset:intel:b760-ddr5": "2023-01-03", "chipset:amd:b650": "2022-10-01"}
    mem = out[out["field"] == "memory_gen_options"].set_index("entity_id")["value"].to_dict()
    assert mem == {"chipset:intel:b760-ddr4": "DDR4", "chipset:intel:b760-ddr5": "DDR5"}
