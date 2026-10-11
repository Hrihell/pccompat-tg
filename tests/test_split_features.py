"""분할(시간 기준 family-group)과 v2 특징 변환(누수 방지) 테스트."""

import numpy as np
import pandas as pd
import pytest

from pccompat.config import load_config
from pccompat.features import build_feature_transformer, numeric_imputer
from pccompat.split import assign_time_group_split, inherit_latest_split, split_ratios


@pytest.fixture
def cfg():
    return load_config()


def _nodes():
    return pd.DataFrame(
        {
            "part_id": ["a1", "a2", "b1", "b2", "c1"],
            "family": ["F-old", "F-old", "F-mid", "F-mid", "F-new"],
            "release_date": pd.to_datetime(["2021-01-10", "2023-09-01", "2023-08-01", "2024-01-01", "2025-02-01"]),
        }
    )


class TestTimeGroupSplit:
    def test_family_goes_where_its_first_member_falls(self, cfg):
        out = assign_time_group_split(_nodes(), cfg["split"])
        # F-old 첫 출시 2021 → train(2023-09 출시 형제도 train에 묶임)
        assert out.set_index("part_id").loc[["a1", "a2"], "split"].tolist() == ["train", "train"]
        assert out.set_index("part_id").loc[["b1", "b2"], "split"].tolist() == ["val", "val"]
        assert out.set_index("part_id").loc["c1", "split"] == "test"

    def test_missing_release_date_is_reported_not_guessed(self, cfg):
        nodes = _nodes()
        nodes.loc[4, "release_date"] = pd.NaT
        out = assign_time_group_split(nodes, cfg["split"])
        assert pd.isna(out.loc[4, "split"])

    def test_edges_inherit_latest_node_split(self):
        split_of = {"a1": "train", "b1": "val", "c1": "test"}
        edges = pd.DataFrame({"src": ["a1", "a1", "b1"], "dst": ["b1", "c1", "a1"]})
        out = inherit_latest_split(edges, ["src", "dst"], split_of)
        assert out["split"].tolist() == ["val", "test", "val"]

    def test_ratios_report(self, cfg):
        out = assign_time_group_split(_nodes(), cfg["split"])
        ratios = split_ratios(out)
        assert pytest.approx(sum(v for k, v in ratios.items() if k != "unassigned")) == 1.0


class TestFeatureLeakage:
    def test_fit_on_train_only(self, cfg):
        rng = np.random.default_rng(cfg["seed"])
        n = 40
        df = pd.DataFrame(
            {
                "vendor": rng.choice(["AMD", "INTEL", "NVIDIA"], n),
                "category": rng.choice(["CPU", "GPU"], n),
                "socket": rng.choice(["AM5", "LGA1700", None], n),
                "cores": rng.integers(4, 24, n).astype(float),
                "threads": rng.integers(4, 32, n).astype(float),
                "boost_clock_mhz": rng.normal(4500, 300, n),
                "tdp_w": rng.normal(120, 30, n),
                "fp32_tflops": rng.normal(20, 5, n),
                "shader_count": rng.normal(6000, 1000, n),
                "msrp_usd": rng.normal(500, 100, n),
            }
        )
        df.loc[::5, "cores"] = np.nan
        train, test = df.iloc[:30], df.iloc[30:].copy()
        test["msrp_usd"] = test["msrp_usd"] * 100                # test 분포가 크게 달라도
        t = build_feature_transformer(cfg["features"])
        t.fit(train)
        imputer = numeric_imputer(t)
        assert imputer.statistics_[0] == pytest.approx(train["cores"].median())  # train 통계만 사용
        x_test = t.transform(test)
        assert x_test.shape[0] == len(test)
        unseen = pd.DataFrame([{**train.iloc[0].to_dict(), "socket": "LGA9999"}])
        t.transform(unseen)                                       # 처음 보는 범주도 오류 없이


class TestReviewRegressions:
    def test_row_without_date_follows_its_family(self, cfg):
        nodes = pd.DataFrame({"part_id": ["a", "b", "c"], "family": ["F", "F", None],
                              "release_date": pd.to_datetime(["2021-01-01", None, None])})
        out = assign_time_group_split(nodes, cfg["split"]).set_index("part_id")
        assert out.loc["b", "split"] == "train"
        assert pd.isna(out.loc["c", "split"])
        assert split_ratios(out.reset_index())["unassigned"] == 1

    def test_family_case_variants_share_a_split(self, cfg):
        nodes = pd.DataFrame({"part_id": ["a", "b"], "family": ["Ryzen 7000", "ryzen 7000 "],
                              "release_date": pd.to_datetime(["2021-01-01", "2025-01-01"])})
        out = assign_time_group_split(nodes, cfg["split"])
        assert out["split"].nunique() == 1

    def test_indicator_for_columns_missing_only_in_test(self, cfg):
        n = 30
        train = pd.DataFrame({c: np.linspace(1, 2, n) for c in cfg["features"]["numeric"]})
        for c in cfg["features"]["categorical"]:
            train[c] = "A"
        test = train.iloc[:5].copy()
        test["cores"] = np.nan
        t = build_feature_transformer(cfg["features"]).fit(train)
        names = list(t.get_feature_names_out())
        assert any("missingindicator_cores" in n for n in names)
        col = names.index(next(n for n in names if "missingindicator_cores" in n))
        assert (t.transform(test)[:, col] == 1).all()

    def test_fits_on_cleaned_frame_with_pd_na(self, cfg):
        from pccompat.clean import CleaningPipeline
        from pccompat.schema import PARTS
        raw = pd.DataFrame({
            "part_id": [f"p{i}" for i in range(6)], "category": ["CPU", "GPU"] * 3,
            "vendor": ["AMD", "NVIDIA"] * 3, "family": ["Ryzen 7000", "GeForce RTX 40"] * 3,
            "model_name": list("abcdef"), "socket": ["AM5", None] * 3,
            "cores": ["8", None] * 3, "release_date": ["2023-01-01"] * 6,
        })
        cleaned = CleaningPipeline(cfg).run(raw, PARTS).frame
        build_feature_transformer(cfg["features"]).fit(cleaned)
