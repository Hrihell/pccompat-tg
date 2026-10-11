from pccompat.normalize_wiki import Collected, _cpu_name, _dedupe_variants, _excluded, _gpu_name


def test_cpu_names():
    assert _cpu_name("Core i9", "12900KS") == "Core i9-12900KS"
    assert _cpu_name("Core Ultra 9", "285K") == "Core Ultra 9 285K"
    assert _cpu_name("Ryzen 7", "7800X3D") == "Ryzen 7 7800X3D"


def test_gpu_names():
    assert _gpu_name(None, "Radeon RX 7600 (Navi 33)") == "Radeon RX 7600"
    assert _gpu_name("Arc 3", "A310") == "Arc A310"


def test_variant_dedupe_keeps_earliest_row_and_its_provenance():
    c = Collected()
    a = {"part_id": "gpu:x:rtx-4060", "family": "F", "model_name": "RTX 4060", "release_date": "2023-06-29"}
    b = {"part_id": "gpu:x:rtx-4060", "family": "F", "model_name": "RTX 4060", "release_date": "2024-04-01"}
    c.parts = [b, a]
    c.provenance = [
        {"_row": id(a), "entity_id": "gpu:x:rtx-4060", "field": "release_date", "value": "2023-06-29"},
        {"_row": id(b), "entity_id": "gpu:x:rtx-4060", "field": "release_date", "value": "2024-04-01"},
    ]
    _dedupe_variants(c)
    assert c.parts == [a]
    assert [p["value"] for p in c.provenance] == ["2023-06-29"]
    assert c.excluded[0]["reason"] == "VARIANT_DUPLICATE"


def test_oem_row_is_kept_when_it_was_also_sold_retail():
    pats, unless = [r"\bOEM\b"], [r"\(retail\)"]
    assert _excluded("5300G 4 (8) Apr 13, 2021 (OEM) OEM", pats, unless) == r"\bOEM\b"
    assert _excluded("5700G 8 (16) Apr 13, 2021 (OEM), Aug 5, 2021 (retail)", pats, unless) is None
    assert _excluded("5600GT 6 (12) Jan 8, 2024", pats, unless) is None
