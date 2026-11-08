import pandas as pd

from pccompat import gpu_specs


def _sheet():
    base = {"family": "Arc A", "our_tier_shader": "WIKIPEDIA_UNCITED", "our_tier_boost": "WIKIPEDIA_UNCITED",
            "source_url": "https://www.intel.com/x", "verified": "", "note": ""}
    return pd.DataFrame([
        {**base, "part_id": "gpu:a", "our_shader": "4096", "our_boost_mhz": "2400",
         "official_shader": "4096", "official_boost_mhz": "2100"},
        {**base, "part_id": "gpu:b", "our_shader": "4864", "our_boost_mhz": "1665",
         "official_shader": "4864", "official_boost_mhz": "1670"},
        {**base, "part_id": "gpu:c", "our_shader": "768", "our_boost_mhz": "2000",
         "official_shader": "", "official_boost_mhz": ""},
    ])


def test_only_values_differing_from_official_are_corrected():
    fixes = gpu_specs.corrections(_sheet())
    # gpu:b 차이 0.3%는 페이지 반올림, gpu:c는 아직 확인하지 않음
    assert list(zip(fixes["part_id"], fixes["field"], fixes["value"])) == [("gpu:a", "boost_clock_mhz", "2100")]


def test_shader_mismatch_is_corrected_too():
    sheet = _sheet().assign(official_shader=["3584", "4864", ""])
    fixes = gpu_specs.corrections(sheet)
    assert set(zip(fixes["part_id"], fixes["field"])) == {("gpu:a", "shader_count"), ("gpu:a", "boost_clock_mhz")}


def test_apply_replaces_value_and_recomputes_fp32():
    parts = pd.DataFrame([{"part_id": "gpu:a", "shader_count": "4096", "boost_clock_mhz": "2400", "fp32_tflops": "19.66"},
                          {"part_id": "gpu:b", "shader_count": "4864", "boost_clock_mhz": "1665", "fp32_tflops": "16.2"}])
    out = gpu_specs.apply(parts, gpu_specs.corrections(_sheet()))
    assert out.loc[0, "boost_clock_mhz"] == "2100" and out.loc[0, "fp32_tflops"] == "17.2"
    assert out.loc[1].to_dict() == parts.loc[1].to_dict()
    assert parts.loc[0, "boost_clock_mhz"] == "2400"  # 입력은 그대로


def test_provenance_rows_are_official_and_verified():
    rows = gpu_specs.provenance_rows(gpu_specs.corrections(_sheet()))
    assert rows == [{"entity_id": "gpu:a", "field": "boost_clock_mhz", "value": "2100",
                     "source_url": "https://www.intel.com/x", "source_tier": "OFFICIAL",
                     "license_scope": "FACT_CITED", "verified": True, "source_revision": "manual"}]
