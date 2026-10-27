import pandas as pd
import pytest

from pccompat import release, report


def test_render_fills_keys_and_formats_ints():
    out = report.render("PASS {{builds_pass}} / ρ {{d_rho}}", {"builds_pass": 222525, "d_rho": "0.855"})
    assert out == "PASS 222,525 / ρ 0.855"


def test_sheet_status_counts_wiki_backed_cells_separately(tmp_path, monkeypatch):
    """위키 값이 있는 칸은 라벨이 이미 정해져 있어 '미입력'이 아니다(숫자에 잠정 표시를 붙이지 않음). 공식 확인 대기는 따로 센다."""
    monkeypatch.setattr(report, "MANUAL", tmp_path)
    pd.DataFrame([{"support": "SUPPORTED", "wiki_support": ""}, {"support": "", "wiki_support": ""},
                  {"support": "", "wiki_support": "SUPPORTED"}]).to_csv(tmp_path / "chipset_support_by_uarch.csv", index=False)
    pd.DataFrame([{"match": "true"}]).to_csv(tmp_path / "verification_sample.csv", index=False)
    pd.DataFrame([{"release_date": "2021-01-01"}]).to_csv(tmp_path / "release_dates_todo.csv", index=False)
    pd.DataFrame([{"l3_cache_mb": "32"}]).to_csv(tmp_path / "cache_todo.csv", index=False)
    pd.DataFrame([{"verified": "true"}]).to_csv(tmp_path / "family_facts.csv", index=False)

    s = report.sheet_status()

    assert (s["sheet_support_done"], s["sheet_support_total"]) == (1, 2)
    assert s["sheet_support_wiki_open"] == 1


def test_render_marks_sheet_dependent_numbers_when_pending():
    nums = {"builds_pass": 10, "n_parts": 247, "pending_sheets": True}
    assert report.render("{{builds_pass}} {{n_parts}}", nums) == "10† 247"


def test_render_fails_on_unknown_key():
    with pytest.raises(KeyError):
        report.render("{{no_such_key}}", {})


def test_render_all_writes_target_named_in_first_line(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "a.tmpl.md").write_text("<!-- out: docs/a.md -->\nn={{n}}\n", encoding="utf-8")
    written = report.render_all({"n": 3}, templates=tmp_path / "t", root=tmp_path)
    assert written == [tmp_path / "docs" / "a.md"]
    assert (tmp_path / "docs" / "a.md").read_text(encoding="utf-8") == "n=3\n"


def test_aggregate_observed_drops_bios_and_small_groups():
    combos = pd.DataFrame([
        {"board_vendor": "A", "board_model": "B550 X", "chipset_id": "c1", "cpu_id": "p1", "ram_gen": "DDR4",
         "bios_version": "1", "bios_date": "2024", "n_reports": 2, "source_commit": "abc"},
        {"board_vendor": "A", "board_model": "B550 X", "chipset_id": "c1", "cpu_id": "p1", "ram_gen": "DDR4",
         "bios_version": "2", "bios_date": "2025", "n_reports": 1, "source_commit": "abc"},
        {"board_vendor": "A", "board_model": "B550 Y", "chipset_id": "c1", "cpu_id": "p1", "ram_gen": "DDR4",
         "bios_version": "1", "bios_date": "2024", "n_reports": 1, "source_commit": "abc"},
    ])
    agg = release.aggregate_observed(combos, min_reports=3)
    assert agg["board_model"].tolist() == ["B550 X"]
    assert agg["n_reports"].tolist() == [3]
    assert "bios_version" not in agg.columns


def test_check_flags_forbidden_files_emails_and_local_paths(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ok.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "data" / "blender_samples.parquet").write_bytes(b"x")
    (tmp_path / "notes.md").write_text("contact someone@example.com at /Users/x/file", encoding="utf-8")
    problems = release.check(tmp_path)
    assert any("blender_samples" in p for p in problems)
    assert any("이메일" in p for p in problems)
    assert any("로컬 경로" in p for p in problems)
    assert not any("ok.csv" in p for p in problems)


def test_verification_summary_counts_only_true_false(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MANUAL", tmp_path)
    pd.DataFrame({"source_tier": ["OFFICIAL", "OFFICIAL", "WIKIPEDIA_UNCITED"],
                  "match": ["true", "false", "na"]}).to_csv(tmp_path / "verification_sample.csv", index=False)
    n = report.verification_numbers()
    assert n["verify_n_judged"] == 2
    assert "OFFICIAL 1/2" in n["verify_summary"]
    assert "1건 제외" in n["verify_summary"]


def test_verification_summary_is_tbd_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MANUAL", tmp_path)
    pd.DataFrame({"source_tier": ["OFFICIAL"], "match": [""]}).to_csv(tmp_path / "verification_sample.csv", index=False)
    assert report.verification_numbers()["verify_summary"].startswith("[TBD")


def test_real_templates_render_when_sheet_dependent_counts_hit_zero():
    # 시트를 다 채우면 NOT_LISTED·OTHER 등이 0이 된다 → 키가 사라지면 run_all이 report 단계에서 멈춘다
    from pccompat.labels import CAUSES
    from pccompat.schema import LABELS, SOURCE_TIERS

    tmpl_keys = set()
    for t in (t for g in report.TEMPLATE_GLOBS for t in report.TEMPLATES.glob(g)):
        tmpl_keys |= set(report.PLACEHOLDER.findall(t.read_text(encoding="utf-8")))
    vocab_keys = (
        {f"support_{v.lower()}" for v in report.SUPPORT_VALUES}
        | {f"tier_{t.lower()}" for t in SOURCE_TIERS} | {f"tier_{t.lower()}_pct" for t in SOURCE_TIERS}
        | {f"builds_{l.lower()}" for l in LABELS} | {f"builds_{l.lower()}_pct" for l in LABELS}
        | {f"builds_cause_{c.lower()}" for c in CAUSES}
        | {f"pair_{e.lower()}_{l.lower()}" for e in report.EDGE_TYPES for l in LABELS}
    )
    # 어휘로 만들 수 있는 키 중 템플릿이 쓰는 것은 모두 어휘에서 나와야 한다(값 존재 여부에 기대지 않음)
    used = {k for k in tmpl_keys if k.startswith(("support_", "tier_", "builds_", "pair_"))}
    assert used <= vocab_keys | {"pair_cpu_chipset", "pair_cpu_ram", "pair_chipset_ram"}


def test_tier_keys_exist_even_when_a_tier_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "OUT", tmp_path)
    pd.DataFrame({"source_tier": ["OFFICIAL"], "verified": [True]}).to_csv(tmp_path / "field_provenance.csv", index=False)
    n = report.provenance_numbers()
    assert n["tier_other"] == 0 and n["tier_wikipedia_cited_column_pct"] == "0.0%"


def test_upstream_counts_cited_wiki_values_whose_footnote_is_an_excluded_site(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "OUT", tmp_path)
    pd.DataFrame({
        "entity_id": ["gpu:a", "gpu:a", "gpu:b", "cpu:c", "cpu:d"],
        "source_tier": ["WIKIPEDIA_CITED_ROW", "WIKIPEDIA_CITED_CELL", "WIKIPEDIA_CITED_ROW",
                        "WIKIPEDIA_CITED_ROW", "OFFICIAL"],
        "source_url": ["https://www.techpowerup.com/gpu-specs/a", "https://www.techpowerup.com/gpu-specs/a",
                       "https://www.tomshardware.com/b", "https://www.amd.com/c", "https://www.techpowerup.com/x"],
        "verified": [False, False, False, False, True],
    }).to_csv(tmp_path / "field_provenance.csv", index=False)
    n = report.provenance_numbers()
    # 위키 각주 값만 센다(OFFICIAL 행의 URL은 상류 출처 문제가 아니다)
    assert n["n_cited_wiki"] == 4
    assert n["n_upstream_tpu"] == 2 and n["n_upstream_tpu_parts"] == 1
    assert n["n_upstream_excluded"] == 3


def test_rejudge_agreement_and_kappa(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MANUAL", tmp_path)
    pd.DataFrame({"source_tier": ["A"] * 4, "match": ["true", "true", "false", "false"]}).to_csv(
        tmp_path / "verification_sample.csv", index=False)
    pd.DataFrame({"row": [1, 2, 3, 4], "rejudge_match": ["true", "false", "false", "false"]}).to_csv(
        tmp_path / "verification_rejudge.csv", index=False)
    s = report.rejudge_numbers()["rejudge_summary"]
    assert "일치율 0.75" in s and "κ 0.50" in s


def test_rejudge_is_tbd_when_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MANUAL", tmp_path)
    pd.DataFrame({"row": [1], "rejudge_match": [""]}).to_csv(tmp_path / "verification_rejudge.csv", index=False)
    assert report.rejudge_numbers()["rejudge_summary"].startswith("[TBD")


def test_verification_cells_exist_for_every_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MANUAL", tmp_path)
    pd.DataFrame({"source_tier": ["WIKIPEDIA_FAMILY", "WIKIPEDIA_FAMILY"],
                  "match": ["true", "false"]}).to_csv(tmp_path / "verification_sample.csv", index=False)
    n = report.verification_numbers()
    assert n["verify_acc_wikipedia_family"].startswith("1/2 [")
    assert n["verify_acc_official"] == "—"


def test_tex_template_escapes_values_and_uses_percent_out_line(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "p.tmpl.tex").write_text("% out: docs/p.tex\nPASS {{p}} {{tier}}\n", encoding="utf-8")
    written = report.render_all({"p": "4.0%", "tier": "A_B"}, templates=tmp_path / "t", root=tmp_path)
    assert written[0].read_text(encoding="utf-8") == "PASS 4.0\\% A\\_B\n"


def test_ablation_counts_coexisting_labels_and_observed_contradictions(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "OUT", tmp_path)
    monkeypatch.setattr(report, "AN", tmp_path)
    t = pd.Timestamp("2022-01-01")
    pd.DataFrame({"label": ["PASS", "UNKNOWN", "UNKNOWN", "FAIL"], "unknown_cause_mask": pd.Series([0, 2, 6, 0], dtype="int8"),
                  "valid_from": [t, t, None, t]}).to_parquet(tmp_path / "build_labels.parquet")
    pd.DataFrame({"label": ["FAIL", "PASS", "UNKNOWN", "All"], "BOTH": [2, 3, 5, 10], "DOCUMENTED": [8, 7, 5, 20],
                  "All": [10, 10, 10, 30]}).to_csv(tmp_path / "e_crosstab.csv", index=False)
    for name in ("a_monthly.csv", "a_monthly_open_chipset.csv"):
        pd.DataFrame({"month": ["2022-01-01", "2022-02-01"], "FAIL": [1, 5], "PASS": [2, 3], "UNKNOWN": [0, 1]}).to_csv(
            tmp_path / name, index=False)
    n = report.ablation_numbers()
    assert (n["x_coexist_pass"], n["x_coexist_unknown"], n["x_coexist_fail"]) == (1, 1, 1)
    assert n["x_board_only"] == 1 and n["x_board_only_pct"] == "50.0%"
    assert n["x_obs_fail_if_unknown_fail"] == 7 and n["x_unproven_pass_if_unknown_pass"] == 10
    assert n["x_eol_base_peak_total"] == 9 and n["x_eol_base_peak_month"] == "2022-02"
