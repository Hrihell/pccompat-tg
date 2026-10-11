"""정제 단계 테스트: 결측은 채우지 않고, 이상치는 지우지 않고, 타입 오류는 드러낸다."""

import numpy as np
import pandas as pd
import pytest

from pccompat.clean import (
    CleaningPipeline,
    assign_eol,
    coerce_types,
    flag_missing,
    flag_out_of_range,
    trim_samples_iqr,
)
from pccompat.config import load_config
from pccompat.schema import PARTS


@pytest.fixture
def cfg():
    return load_config()


def _parts(**overrides):
    base = {
        "part_id": ["cpu:x:a", "cpu:x:b", "gpu:x:c"],
        "category": ["CPU", "CPU", "GPU"],
        "vendor": ["AMD", "INTEL", "NVIDIA"],
        "family": ["Ryzen 7000", "Core 12th Gen", "GeForce RTX 40"],
        "model_name": ["A", "B", "C"],
        "socket": ["AM5", None, None],
        "memory_gens": ["DDR5", "DDR4;DDR5", None],
        "max_memory_gb": ["128", "128", None],
        "cores": ["8", "12", None],
        "boost_clock_mhz": ["5000", "4900", "2475"],
        "shader_count": [None, None, "5888"],
        "release_date": ["2023-04-06", "2021-11-04", "2023-04-13"],
        "msrp_usd": ["449", None, "599"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestCoerceTypes:
    def test_parses_kinds_and_reports_failures(self):
        df = _parts(cores=["8", "twelve", None], release_date=["2023-04-06", "2021-13-40", "2023-04-13"])
        out, issues = coerce_types(df, PARTS)
        assert str(out["cores"].dtype) == "Int64"
        assert out.loc[1, "cores"] is pd.NA                      # 파싱 실패는 결측이 되고
        assert pd.isna(out.loc[1, "release_date"])
        assert {(i.row, i.column) for i in issues} >= {(1, "cores"), (1, "release_date")}  # 반드시 보고된다
        assert out.loc[1, "memory_gens"] == ("DDR4", "DDR5")

    def test_rejects_value_outside_enum(self):
        df = _parts(vendor=["AMD", "VIA", "NVIDIA"])
        _, issues = coerce_types(df, PARTS)
        assert any(i.column == "vendor" and i.row == 1 for i in issues)

    def test_input_frame_is_not_mutated(self):
        df = _parts()
        before = df.copy()
        coerce_types(df, PARTS)
        pd.testing.assert_frame_equal(df, before)


class TestMissing:
    def test_never_imputes_and_marks_partial(self):
        typed, _ = coerce_types(_parts(), PARTS)
        out, report = flag_missing(typed, PARTS)
        assert out.loc[1, "record_status"] == "PARTIAL"           # 소켓 없음
        assert pd.isna(out.loc[1, "socket"])                      # 값을 채우지 않음
        assert out.loc[0, "record_status"] == "COMPLETE"
        assert "socket" in out.loc[1, "missing_required"]
        assert report.loc["socket", "missing"] == 2               # GPU 행의 소켓 결측도 집계는 됨


class TestOutliers:
    def test_flags_but_keeps_rows(self, cfg):
        typed, _ = coerce_types(_parts(boost_clock_mhz=["5000", "49000", "2475"]), PARTS)
        out = flag_out_of_range(typed, cfg["ranges"])
        assert len(out) == 3
        assert out.loc[1, "boost_clock_mhz"] == 49000             # 값은 그대로
        assert "boost_clock_mhz" in out.loc[1, "review_flags"]
        assert out.loc[0, "review_flags"] == ()

    def test_iqr_trims_upload_samples_per_group(self, cfg):
        rng = np.random.default_rng(cfg["seed"])
        a = list(rng.normal(100, 2, 50)) + [400.0]                # 업로드 이상치 1건
        b = list(rng.normal(50, 1, 30))
        samples = pd.DataFrame({"device": ["a"] * 51 + ["b"] * 30, "score": a + b})
        kept, report = trim_samples_iqr(samples, "device", "score", k=cfg["measurements"]["iqr_k"],
                                        min_n=cfg["measurements"]["low_n_threshold"])
        rep = report.set_index("group")
        assert 400.0 not in kept["score"].values
        assert rep.loc["a", "n_removed"] >= 1 and rep.loc["a", "trimmed"]
        assert len(kept) + rep["n_removed"].sum() == len(samples)


class TestEol:
    def test_eol_is_next_generation_first_release(self, cfg):
        typed, _ = coerce_types(
            _parts(
                family=["Ryzen 7000", "Ryzen 9000", "GeForce RTX 40"],
                release_date=["2022-09-27", "2024-08-08", "2022-10-12"],
            ),
            PARTS,
        )
        out = assign_eol(typed, cfg["lineage"])
        assert out.loc[0, "eol_date"] == pd.Timestamp("2024-08-08")
        assert bool(out.loc[0, "eol_is_assumed"]) is True
        assert pd.isna(out.loc[1, "eol_date"])                    # 다음 세대 없음 → 열린 구간
        assert pd.isna(out.loc[2, "eol_date"])                    # 다음 세대(RTX 50) 행이 없으면 계산 불가


class TestPipeline:
    def test_run_returns_reports_and_does_not_drop_rows(self, cfg):
        result = CleaningPipeline(cfg).run(_parts(cores=["8", "x", None]), PARTS)
        assert len(result.frame) == 3
        assert {"before", "after", "comparison", "issues", "missing"} <= set(result.reports)
        assert not result.reports["issues"].empty


class TestValidation:
    def test_before_profile_reads_numeric_strings_and_comparison_shows_parse_loss(self, cfg):
        result = CleaningPipeline(cfg).run(_parts(cores=["8", "x", "12"]), PARTS)
        cmp = result.reports["comparison"]
        assert result.reports["before"].loc["boost_clock_mhz", "mean"] is not None   # 원본 문자열도 수치 요약
        assert cmp.loc["cores", "count_after"] == cmp.loc["cores", "count_before"] - 1  # 파싱 실패 1건이 드러남


class TestReviewRegressions:
    """코드 리뷰(2026-10-08) 지적 재현 테스트."""

    def test_eol_keeps_sourced_values_and_fills_only_gaps(self, cfg):
        df = _parts(family=["Ryzen 5000", "Ryzen 7000", "Ryzen 9000"],
                    release_date=["2020-11-05", "2022-09-27", "2024-08-08"])
        df["eol_date"] = ["2023-01-01", None, "2026-01-01"]
        df["eol_is_assumed"] = ["false", None, "false"]
        typed, _ = coerce_types(df, PARTS)
        out = assign_eol(typed, cfg["lineage"])
        assert out.loc[0, "eol_date"] == pd.Timestamp("2023-01-01")      # 출처 값 유지
        assert bool(out.loc[0, "eol_is_assumed"]) is False
        assert out.loc[1, "eol_date"] == pd.Timestamp("2024-08-08")      # 빈 칸만 가정으로 채움
        assert bool(out.loc[1, "eol_is_assumed"]) is True
        assert out.loc[2, "eol_date"] == pd.Timestamp("2026-01-01")

    def test_huge_integer_is_reported_not_crash(self):
        out, issues = coerce_types(_parts(cores=["8", "1e30", None]), PARTS)
        assert out.loc[1, "cores"] is pd.NA
        assert any(i.column == "cores" and i.row == 1 for i in issues)

    def test_non_unique_index_reports_positional_rows(self):
        df = _parts(cores=["8", "x", None])
        df.index = [0, 0, 0]
        out, issues = coerce_types(df, PARTS)
        bad = [i for i in issues if i.column == "cores"]
        assert len(bad) == 1 and bad[0].row == 1 and bad[0].value == "x"
        assert list(out.index) == [0, 1, 2]

    def test_float_bool_and_enum_case_are_normalized(self):
        df = _parts(category=["cpu", "CPU", "Gpu"], memory_gens=["ddr5", "DDR4;ddr5", None])
        df["eol_is_assumed"] = ["1.0", "0.0", None]
        out, issues = coerce_types(df, PARTS)
        assert out["category"].tolist() == ["CPU", "CPU", "GPU"]
        assert out.loc[1, "memory_gens"] == ("DDR4", "DDR5")
        assert out.loc[0, "eol_is_assumed"] == True  # noqa: E712
        assert not [i for i in issues if i.column in ("category", "memory_gens", "eol_is_assumed")]

    def test_iqr_counts_nan_separately_and_skips_small_groups(self, cfg):
        samples = pd.DataFrame(
            {
                "device": ["a"] * 4 + ["b"] * 3 + [None, None],
                "score": [10, 10, 10, 11, 5, np.nan, 6, 7, 8],
            }
        )
        kept, report = trim_samples_iqr(samples, "device", "score", k=1.5, min_n=cfg["measurements"]["low_n_threshold"])
        rep = report.set_index("group")
        assert rep.loc["a", "n_removed"] == 0 and not rep.loc["a", "trimmed"]   # n<10 → 자르지 않음
        assert rep.loc["b", "n_nan"] == 1 and rep.loc["b", "n_removed"] == 0
        assert rep.loc["<NA>", "n"] == 2                                       # 키 없는 행도 보고
        assert rep.loc["<NA>", "n_no_key"] == 2
        assert len(kept) + rep[["n_removed", "n_nan", "n_no_key"]].to_numpy().sum() == len(samples)


def test_assign_eol_uses_first_successor_released_after_the_model_itself():
    # 5700X3D처럼 다음 세대(7000)가 나온 뒤에 출시된 모델은 그 뒤에 처음 나온 세대(9000) 출시일을 단종일로 쓴다.
    lineage = {"amd": ["R5", "R7", "R9"]}
    df = pd.DataFrame({"family": ["R5", "R5", "R7", "R9", "R5"],
                       "release_date": pd.to_datetime(["2020-11-05", "2024-01-31", "2022-09-27", "2024-08-08", "2026-09-10"]),
                       "eol_date": pd.NaT, "eol_is_assumed": pd.NA})
    out = assign_eol(df, lineage)
    assert out.loc[0, "eol_date"] == pd.Timestamp("2022-09-27")   # 보통 모델: 다음 세대 첫 출시일
    assert out.loc[1, "eol_date"] == pd.Timestamp("2024-08-08")   # 늦게 나온 모델: 자기 출시 뒤 첫 후속 세대
    assert pd.isna(out.loc[4, "eol_date"])                         # 그런 세대가 없으면 열린 구간(창 끝)
    assert (out["eol_date"].dropna() >= out.loc[out["eol_date"].notna(), "release_date"]).all()
