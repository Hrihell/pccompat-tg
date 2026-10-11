"""§7-1 장치명 → 모델 ID 매칭 규칙 테스트(설계서 pccompat-dataset-design.md §7-1).

모델명은 가상 이름이 아니라 실제 제품명 문자열을 쓰지만, 스펙 값은 다루지 않는다(이름 규칙만 검사).
"""

import pytest

from pccompat.config import load_config
from pccompat.names import Catalog, MatchStatus, NameMatcher, normalize, tokens


@pytest.fixture(scope="module")
def cfg():
    return load_config()["matching"]


@pytest.fixture(scope="module")
def matcher(cfg):
    catalog = Catalog.from_rows(
        [
            ("gpu:nvidia:geforce-rtx-4060", "GeForce RTX 4060", []),
            ("gpu:nvidia:geforce-rtx-4070", "GeForce RTX 4070", []),
            ("gpu:nvidia:geforce-rtx-4070-ti", "GeForce RTX 4070 Ti", []),
            ("gpu:nvidia:geforce-rtx-4070-super", "GeForce RTX 4070 SUPER", []),
            ("gpu:amd:radeon-rx-7900-xtx", "Radeon RX 7900 XTX", []),
            ("cpu:amd:ryzen-7-7800x3d", "Ryzen 7 7800X3D", []),
            ("cpu:amd:ryzen-9-7950x", "Ryzen 9 7950X", []),
            ("cpu:amd:ryzen-9-7950x3d", "Ryzen 9 7950X3D", []),
            ("cpu:intel:core-i7-12700k", "Core i7-12700K", []),
            ("cpu:intel:core-i7-12700kf", "Core i7-12700KF", []),
            ("cpu:intel:core-ultra-9-285k", "Core Ultra 9 285K", []),
        ],
        cfg,
    )
    return NameMatcher(catalog, cfg)


class TestNormalize:
    def test_strips_cpuid_brand_noise(self, cfg):
        assert normalize("AMD Ryzen 9 7950X 16-Core Processor", cfg) == normalize("Ryzen 9 7950X", cfg)
        assert normalize("12th Gen Intel(R) Core(TM) i7-12700K", cfg) == normalize("Core i7-12700K", cfg)
        assert normalize("11th Gen Intel Core i9-11900K @ 3.50GHz", cfg) == normalize("Core i9-11900K", cfg)

    def test_splits_glued_suffix(self, cfg):
        assert tokens("RTX 4070Ti", cfg) == tokens("RTX 4070 Ti", cfg)
        assert tokens("i7-14700KF", cfg)[-2:] == ["14700", "kf"]
        assert tokens("Ryzen 7 7800X3D", cfg)[-2:] == ["7800", "x3d"]

    def test_unit_suffix_is_not_split(self, cfg):
        assert "16gb" in tokens("RTX 4060 Ti 16GB", cfg)


class TestConfirmedMatch:
    def test_exact_name_is_confirmed(self, matcher):
        r = matcher.match("NVIDIA GeForce RTX 4070")
        assert r.status is MatchStatus.CONFIRMED
        assert r.model_id == "gpu:nvidia:geforce-rtx-4070"
        assert r.score == 1.0

    def test_cpuid_brand_string_is_confirmed(self, matcher):
        r = matcher.match("AMD Ryzen 7 7800X3D 8-Core Processor")
        assert r.status is MatchStatus.CONFIRMED
        assert r.model_id == "cpu:amd:ryzen-7-7800x3d"


class TestDistinguishingMarks:
    @pytest.mark.parametrize(
        "name, wrong",
        [
            ("GeForce RTX 4070 Ti", "gpu:nvidia:geforce-rtx-4070"),
            ("Ryzen 9 7950X3D", "cpu:amd:ryzen-9-7950x"),
            ("Core i7-12700KF", "cpu:intel:core-i7-12700k"),
        ],
    )
    def test_different_mark_never_matches_sibling(self, matcher, name, wrong):
        r = matcher.match(name)
        assert r.model_id != wrong
        assert all(c.model_id != wrong for c in r.candidates)


class TestNumbers:
    def test_both_sides_different_numbers_is_no_match(self, matcher):
        r = matcher.match("GeForce RTX 4080")  # 4080 없음: 4060·4070과 숫자가 서로 다름
        assert r.status is MatchStatus.NO_MATCH
        assert r.candidates == []

    def test_one_sided_extra_number_is_candidate_not_confirmed(self, matcher, cfg):
        r = matcher.match("GeForce RTX 4060 8GB")
        assert r.status is MatchStatus.CANDIDATE
        assert r.model_id is None
        assert r.candidates[0].model_id == "gpu:nvidia:geforce-rtx-4060"
        assert r.candidates[0].score <= cfg["candidate_score_cap"] < 1.0


class TestMultiset:
    def test_repeated_tokens_cannot_reach_confirmation(self, matcher):
        r = matcher.match("GeForce RTX 4070 RTX 4070")
        assert r.status is not MatchStatus.CONFIRMED

    def test_two_model_keys_in_one_name_is_excluded(self, matcher):
        r = matcher.match("GeForce RTX 4070 + GeForce RTX 4070 Ti")
        assert r.status is MatchStatus.EXCLUDED
        assert r.reason == "MULTI_MODEL"

    def test_nested_key_is_not_multi_model(self, matcher):
        r = matcher.match("GeForce RTX 4070 Ti")
        assert r.status is MatchStatus.CONFIRMED
        assert r.model_id == "gpu:nvidia:geforce-rtx-4070-ti"


class TestMobile:
    @pytest.mark.parametrize(
        "name",
        [
            "NVIDIA GeForce RTX 4070 Laptop GPU",
            "GeForce RTX 4070 Max-Q",
            "AMD Ryzen 9 7945HX",
            "Intel Core Ultra 9 285H",
        ],
    )
    def test_mobile_parts_are_excluded(self, matcher, name):
        r = matcher.match(name)
        assert r.status is MatchStatus.EXCLUDED
        assert r.reason == "MOBILE"


class TestReviewRegressions:
    @pytest.mark.parametrize("name", ["GeForce RTX 4070/4080", "Ryzen 9 7950X / 7800X3D", "RTX 4060 & 4070"])
    def test_shorthand_second_model_is_multi_model(self, matcher, name):
        r = matcher.match(name)
        assert r.status is MatchStatus.EXCLUDED and r.reason == "MULTI_MODEL"

    @pytest.mark.parametrize("name", ["Ryzen 7 7800X3D", "Core Ultra 9 285K", "GeForce RTX 4060 16GB"])
    def test_tier_digit_and_unit_are_not_second_model(self, matcher, name):
        assert matcher.match(name).reason != "MULTI_MODEL"

    @pytest.mark.parametrize("name", ["Core i9-13900HK", "Ryzen 9 7945HX3D", "Core i7-1270P"])
    def test_more_mobile_suffixes(self, matcher, name):
        r = matcher.match(name)
        assert r.status is MatchStatus.EXCLUDED and r.reason == "MOBILE"
