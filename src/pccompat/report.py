"""WP7: 산출물에서 수치를 뽑아 data/out/numbers.json에 쓰고, 문서 템플릿의 {{키}}를 그 값으로 채운다.

문서(데이터 시트·논문 초안)의 숫자는 손으로 적지 않는다. 시트가 바뀌면 run_all이 이 단계까지 다시 돌린다.
- 템플릿: docs/templates/*.tmpl.md → 결과: 각 템플릿 첫 줄 주석 `<!-- out: 경로 -->`가 가리키는 파일
- 템플릿에 없는 키를 쓰면 오류로 멈춘다(빈칸이 조용히 남지 않게).
- 사람 입력 시트가 덜 채워졌으면 그 시트에 기대는 수치 키에 † 를 붙인다(잠정값 표시).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pccompat.config import PROJECT_ROOT, load_config
from pccompat.labels import CAUSES
from pccompat.quality import accuracy_by_tier
from pccompat import paper_stats, validation_stats
from pccompat.analysis import _cpu_chipset_by_base
from pccompat.schema import CPU_CHIPSET_SUPPORT, LABELS, PAIR_LABELS, SOURCE_TIERS

OUT = PROJECT_ROOT / "data" / "out"
AN = OUT / "analysis"
MANUAL = PROJECT_ROOT / "data" / "manual"
TEMPLATES = PROJECT_ROOT / "docs" / "templates"
PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
# 첫 줄: Markdown은 <!-- out: 경로 -->, LaTeX는 % out: 경로
OUT_LINE = re.compile(r"^(?:<!--\s*out:\s*(\S+)\s*-->|%\s*out:\s*(\S+))\s*\n")
TEMPLATE_GLOBS = ("*.tmpl.md", "*.tmpl.tex")
TEX_SPECIALS = {"%": r"\%", "_": r"\_", "&": r"\&", "#": r"\#", "$": r"\$"}
PROVISIONAL_MARK = "†"
# 데이터 시트의 "제외한 출처"(직접 수집하지 않는 사이트). 위키 각주가 이들을 가리키면 상류 출처로 공개한다.
EXCLUDED_HOSTS = ("techpowerup.com", "tomshardware.com", "passmark.com", "geekbench.com", "3dmark.com")


def _allowed(table: Any, column: str) -> tuple[str, ...]:
    return next(c.allowed for c in table.columns if c.name == column)


# 값이 0이 돼도 키가 사라지지 않도록 고정 어휘로 센다(템플릿 KeyError 방지)
SUPPORT_VALUES = _allowed(CPU_CHIPSET_SUPPORT, "support")
EDGE_TYPES = _allowed(PAIR_LABELS, "edge_type")


def _filled(path: Path, col: str) -> tuple[int, int]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return int((df[col].str.strip() != "").sum()), len(df)


def sheet_status() -> dict[str, Any]:
    s: dict[str, Any] = {}
    for key, (name, col) in {"support": ("chipset_support_by_uarch.csv", "support"),
                             "verify": ("verification_sample.csv", "match"),
                             "release": ("release_dates_todo.csv", "release_date"),
                             "cache": ("cache_todo.csv", "l3_cache_mb")}.items():
        done, total = _filled(MANUAL / name, col)
        s[f"sheet_{key}_done"], s[f"sheet_{key}_total"] = done, total
    # 위키 값이 있는 빈칸은 라벨이 이미 정해져 있으므로 미입력으로 치지 않고, 공식 확인 대기로 따로 센다
    sup = pd.read_csv(MANUAL / "chipset_support_by_uarch.csv", dtype=str, keep_default_na=False)
    wiki_open = 0
    if "wiki_support" in sup:
        wiki_open = int(((sup["support"].str.strip() == "") & (sup["wiki_support"].str.strip() != "")).sum())
    s["sheet_support_total"] -= wiki_open
    s["sheet_support_wiki_open"] = wiki_open
    facts = pd.read_csv(MANUAL / "family_facts.csv", dtype=str, keep_default_na=False)
    s["facts_verified"] = int((facts["verified"].str.lower() == "true").sum())
    s["facts_total"] = len(facts)
    return s


def verification_numbers() -> dict[str, Any]:
    """검증 표본의 등급별 정확도(Wilson 95%). match가 true/false인 행만 센다(na 등은 '공식 값 없음')."""
    sample = pd.read_csv(MANUAL / "verification_sample.csv", dtype=str, keep_default_na=False)
    acc = accuracy_by_tier(sample)
    if acc.empty:
        return {"verify_summary": f"[TBD — 검증 표본 0/{len(sample)} 판정]", "verify_n_judged": 0,
                "verify_n_no_official": 0, **{f"verify_acc_{t.lower()}": "[TBD]" for t in SOURCE_TIERS}}
    parts = [f"{r.source_tier} {r.correct}/{r.n} ({r.accuracy:.2f}, {r.wilson_low:.2f}–{r.wilson_high:.2f})"
             for r in acc.itertuples()]
    no_official = int((~sample["match"].str.lower().isin(["true", "false", ""])).sum())
    # 등급별 표 칸: 표본이 없는 등급도 키를 둔다("—")
    cells = {f"verify_acc_{t.lower()}": "—" for t in SOURCE_TIERS}
    cells |= {f"verify_acc_{r.source_tier.lower()}": f"{r.correct}/{r.n} [{r.wilson_low:.2f}, {r.wilson_high:.2f}]"
              for r in acc.itertuples()}
    return {"verify_summary": "; ".join(parts) + f" (공식 값을 찾지 못한 {no_official}건 제외)",
            "verify_n_judged": int(acc["n"].sum()), "verify_n_no_official": no_official, **cells}


def rejudge_numbers() -> dict[str, Any]:
    """48시간 뒤 재판정(단일 검토자 자기 일치도): 일치율과 Cohen's κ. 재판정 시트가 비어 있으면 [TBD]."""
    path = MANUAL / "verification_rejudge.csv"
    if not path.exists():
        return {"rejudge_summary": "재판정 미실시(사용자 결정 2026-10-09)"}
    re_ = pd.read_csv(path, dtype=str, keep_default_na=False)
    done = re_[re_["rejudge_match"].str.lower().isin(["true", "false"])]
    if done.empty:
        return {"rejudge_summary": f"[TBD — 재판정 0/{len(re_)}]"}
    first = pd.read_csv(MANUAL / "verification_sample.csv", dtype=str, keep_default_na=False)["match"].str.lower()
    a = first.iloc[done["row"].astype(int) - 1].to_numpy() == "true"
    b = done["rejudge_match"].str.lower().to_numpy() == "true"
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    kappa = "정의 불가(한쪽 값만 있음)" if pe == 1 else f"{(po - pe) / (1 - pe):.2f}"
    return {"rejudge_summary": f"재판정 {len(done)}건 일치율 {po:.2f}, Cohen's κ {kappa}"}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def node_numbers() -> dict[str, Any]:
    parts = pd.read_csv(OUT / "parts.csv")
    chips = pd.read_csv(OUT / "chipsets.csv")
    sup = pd.read_csv(OUT / "cpu_chipset_support.csv")
    n = {"n_parts": len(parts), "n_cpu": int((parts["category"] == "CPU").sum()),
         "n_gpu": int((parts["category"] == "GPU").sum()), "n_partial": int((parts["record_status"] == "PARTIAL").sum()),
         "n_chipsets": len(chips),
         "n_chipsets_base": int(chips["base_chipset_id"].nunique()) if "base_chipset_id" in chips else len(chips),
         "n_chipsets_split": int(chips.loc[chips["board_memory"].notna(), "base_chipset_id"].nunique()) if "board_memory" in chips else 0,
         "n_ram": len(pd.read_csv(OUT / "ram_specs.csv")),
         "n_cpu_families": parts.loc[parts["category"] == "CPU", "family"].nunique(),
         "n_gpu_families": parts.loc[parts["category"] == "GPU", "family"].nunique(),
         "n_release_missing": int(parts["release_date"].isna().sum()),
         "n_msrp_missing": int(parts["msrp_usd"].isna().sum()), "n_support": len(sup)}
    counts = sup["support"].value_counts()
    for k in SUPPORT_VALUES:
        n[f"support_{k.lower()}"] = int(counts.get(k, 0))
    return n


def label_numbers() -> dict[str, Any]:
    pairs = pd.read_csv(OUT / "pair_labels.csv")
    n: dict[str, Any] = {"n_pairs": len(pairs)}
    by = pairs.groupby(["edge_type", "label"]).size()
    for edge in EDGE_TYPES:
        n[f"pair_{edge.lower()}"] = int((pairs["edge_type"] == edge).sum())
        for label in LABELS:
            n[f"pair_{edge.lower()}_{label.lower()}"] = int(by.get((edge, label), 0))
    pq = str(OUT / "build_labels.parquet")
    con = duckdb.connect()
    n["n_builds"] = con.execute(f"SELECT count(*) FROM '{pq}'").fetchone()[0]
    n["n_builds_coexist"] = con.execute(f"SELECT count(*) FROM '{pq}' WHERE valid_from IS NOT NULL").fetchone()[0]
    got = dict(con.execute(f"SELECT label, count(*) FROM '{pq}' GROUP BY 1").fetchall())
    for label in LABELS:
        v = int(got.get(label, 0))
        n[f"builds_{label.lower()}"] = v
        n[f"builds_{label.lower()}_pct"] = _pct(v / n["n_builds"]) if n["n_builds"] else "0%"
    for i, cause in enumerate(CAUSES):
        n[f"builds_cause_{cause.lower()}"] = con.execute(
            f"SELECT count(*) FROM '{pq}' WHERE (unknown_cause_mask >> {i}) & 1 = 1").fetchone()[0]
    return n


def provenance_numbers() -> dict[str, Any]:
    prov = pd.read_csv(OUT / "field_provenance.csv")
    n: dict[str, Any] = {"n_provenance": len(prov), "n_prov_verified": int(prov["verified"].astype(str).str.lower().eq("true").sum())}
    counts = prov["source_tier"].value_counts()
    n["n_tiers"] = len(SOURCE_TIERS)
    for tier in SOURCE_TIERS:
        v = int(counts.get(tier, 0))
        n[f"tier_{tier.lower()}"] = v
        n[f"tier_{tier.lower()}_pct"] = _pct(v / len(prov)) if len(prov) else "0%"
    return {**n, **upstream_numbers(prov)}


def upstream_numbers(prov: pd.DataFrame) -> dict[str, Any]:
    """위키 각주 값 중 각주가 제외 출처를 가리키는 값(사실값은 위키에서 왔지만 상류는 그 사이트)."""
    cited = prov[prov["source_tier"].astype(str).str.startswith("WIKIPEDIA_CITED")]
    url = cited.get("source_url", pd.Series("", index=cited.index)).fillna("").astype(str)
    host = url.str.extract(r"https?://(?:www\.)?([^/]+)", expand=False).fillna("")
    tpu = cited[host == "techpowerup.com"]
    entity = tpu.get("entity_id", pd.Series(dtype=str))
    return {"n_cited_wiki": len(cited), "n_upstream_tpu": len(tpu), "n_upstream_tpu_parts": int(entity.nunique()),
            "n_upstream_excluded": int(host.isin(EXCLUDED_HOSTS).sum())}


def split_numbers() -> dict[str, Any]:
    nodes = pd.read_csv(AN / "split_nodes.csv", index_col=0)
    n: dict[str, Any] = {}
    for cat, row in nodes.iterrows():
        assigned = sum(int(row.get(s, 0)) for s in ("train", "val", "test"))
        for s in ("train", "val", "test"):
            n[f"split_{cat.lower()}_{s}"] = int(row.get(s, 0))
            n[f"split_{cat.lower()}_{s}_pct"] = _pct(int(row.get(s, 0)) / assigned) if assigned else "0%"
        n[f"split_{cat.lower()}_unassigned"] = int(row.get("unassigned", 0))
    return n


def analysis_numbers() -> dict[str, Any]:
    n: dict[str, Any] = {}
    monthly = pd.read_csv(AN / "a_monthly.csv")
    monthly["total"] = monthly[["FAIL", "PASS", "UNKNOWN"]].sum(axis=1)
    peak = monthly.loc[monthly["PASS"].idxmax()]
    busiest = monthly.loc[monthly["total"].idxmax()]
    n.update({"a_peak_total_month": str(busiest["month"])[:7], "a_peak_total": int(busiest["total"]),
              "a_peak_total_pass": int(busiest["PASS"])})
    n.update({"a_n_months": len(monthly), "a_peak_pass_month": str(peak["month"])[:7], "a_peak_pass": int(peak["PASS"]),
              "a_last_month": str(monthly["month"].iloc[-1])[:7], "a_last_pass": int(monthly["PASS"].iloc[-1])})
    up = pd.read_csv(AN / "b_upgrade.csv")
    for fam, g in up.groupby("family"):
        key = re.sub(r"\W+", "", fam.lower())
        n[f"b_{key}_n_chipsets"] = len(g)
        n[f"b_{key}_n_with_pass"] = int((g["n_cpu_pass"] > 0).sum())
        n[f"b_{key}_max_span"] = f"{g['span_months'].max():.1f}" if g["span_months"].notna().any() else "—"
        n[f"b_{key}_max_families"] = int(g["n_cpu_families_pass"].max())
        n[f"b_{key}_n_max_families"] = int((g["n_cpu_families_pass"] == g["n_cpu_families_pass"].max()).sum())
    par = pd.read_csv(AN / "c_pareto_monthly.csv")
    live = par[par["n_pairs"] > 0]
    n.update({"c_median_frontier": f"{live['n_frontier'].median():.0f}", "c_max_frontier": int(live["n_frontier"].max()),
              "c_n_months": len(live)})
    d = pd.read_csv(AN / "d_validity.csv")
    n["d_n_combos"] = len(d)
    for _, r in d[d["primary"]].iterrows():
        key = f"d_{r['target']}_{r['source']}_{r['vendor']}".lower()
        n.update({f"{key}_rho": f"{r['rho']:.3f}", f"{key}_lo": f"{r['ci_low']:.3f}", f"{key}_hi": f"{r['ci_high']:.3f}",
                  f"{key}_n": int(r["n_models"])})
    ct = pd.read_csv(AN / "e_crosstab.csv", index_col=0)
    for label in ("PASS", "FAIL", "UNKNOWN"):
        n[f"e_{label.lower()}_observed"] = int(ct.loc[label, "BOTH"]) if label in ct.index else 0
        n[f"e_{label.lower()}_total"] = int(ct.loc[label, "All"]) if label in ct.index else 0
    n["e_fail_observed"] = len(pd.read_csv(AN / "e_fail_but_observed.csv"))
    n["e_resolution_candidates"] = len(pd.read_csv(AN / "e_unknown_resolution_candidates.csv"))
    bal = pd.read_csv(AN / "balanced_report.csv")
    n["balanced_n"] = int(bal["picked"].sum())
    return n


# 사람 입력 시트에 기대는 수치: 시트가 덜 차면 † 표시
PROVISIONAL_PREFIXES = ("builds_", "pair_cpu_chipset", "a_", "b_", "c_", "e_", "x_", "tier_", "n_prov_verified", "support_")


def ablation_numbers() -> dict[str, Any]:
    """논문 ablation: (1) 라벨 체계 3값 vs 2값 (2) 칩셋 단종일 가정. 이미 만든 산출물만 다시 센다."""
    b = pd.read_parquet(OUT / "build_labels.parquet", columns=["label", "unknown_cause_mask", "valid_from"])
    co = b[b["valid_from"].notna()]
    n: dict[str, Any] = {f"x_coexist_{l.lower()}": int((co["label"] == l).sum()) for l in LABELS}
    n["x_coexist_pass_if_unknown_pass"] = n["x_coexist_pass"] + n["x_coexist_unknown"]
    unk = b[b["label"] == "UNKNOWN"]
    board_only = int((unk["unknown_cause_mask"] == 1 << CAUSES.index("BOARD_DEPENDENT")).sum())
    n.update({"x_board_only": board_only, "x_board_only_pct": _pct(board_only / len(unk)) if len(unk) else "0.0%"})
    ct = pd.read_csv(AN / "e_crosstab.csv", index_col=0)
    for label in LABELS:
        obs, tot = (int(ct.loc[label, "BOTH"]), int(ct.loc[label, "All"])) if label in ct.index else (0, 0)
        n[f"x_obs_rate_{label.lower()}"] = f"{obs / tot * 100:.2f}%" if tot else "—"  # FAIL 0.03%가 0.0%로 보이지 않게
    fail_obs = int(ct.loc["FAIL", "BOTH"]) if "FAIL" in ct.index else 0
    unk_obs = int(ct.loc["UNKNOWN", "BOTH"]) if "UNKNOWN" in ct.index else 0
    n["x_obs_fail_if_unknown_fail"] = fail_obs + unk_obs
    n["x_unproven_pass_if_unknown_pass"] = int(ct.loc["UNKNOWN", "All"]) if "UNKNOWN" in ct.index else 0
    for key, name in (("base", "a_monthly.csv"), ("open", "a_monthly_open_chipset.csv")):
        m = pd.read_csv(AN / name)
        total = m[[c for c in LABELS if c in m]].sum(axis=1)
        n[f"x_eol_{key}_peak_total"], n[f"x_eol_{key}_peak_month"] = int(total.max()), str(m.loc[total.idxmax(), "month"])[:7]
        n[f"x_eol_{key}_pass_months"] = int(m["PASS"].sum())
        n[f"x_eol_{key}_peak_pass"] = int(m["PASS"].max())
    return n


def gpu_warning_numbers() -> dict[str, Any]:
    """G2 GPU 연결 경고: FAIL이 아닌 공존 조합 중 경고 수와 판단할 수 있었던 비율(플랫폼 사양을 사람이 확인한 만큼)."""
    b = pd.read_parquet(OUT / "build_labels.parquet", columns=["label", "valid_from", "warn_mask", "warn_unchecked_mask"])
    co = b[b["valid_from"].notna() & (b["label"] != "FAIL")]  # 장착 가능한(FAIL 아닌) 공존 조합만
    n: dict[str, Any] = {"g_coexist": len(co)}
    for i, name in enumerate(("pcie",)):
        warned = (co["warn_mask"].to_numpy() >> i) & 1
        checked = 1 - ((co["warn_unchecked_mask"].to_numpy() >> i) & 1)
        n[f"g_{name}_warn"] = int(warned.sum())
        n[f"g_{name}_checked_pct"] = _pct(checked.mean()) if len(co) else "0%"
    return n


def review_numbers() -> dict[str, Any]:
    """리뷰 대응 검정·재집계(paper_stats)."""
    sample = pd.read_csv(MANUAL / "verification_sample.csv", dtype=str, keep_default_na=False)
    builds = pd.read_parquet(OUT / "build_labels.parquet",
                             columns=["cpu_id", "chipset_id", "ram_id", "gpu_id", "label", "failed_rule_mask", "valid_from"])
    return {**paper_stats.tier_tests(sample), **paper_stats.same_socket(builds),
            **paper_stats.observed_tests(pd.read_csv(AN / "e_crosstab.csv", index_col=0)),
            **paper_stats.proxy_distribution(pd.read_csv(AN / "d_validity.csv")),
            **paper_stats.board_memory_stats(builds, pd.read_csv(OUT / "chipsets.csv"), pd.read_csv(OUT / "ram_specs.csv"))}


def validation_numbers() -> dict[str, Any]:
    """심사 2차 대응(validation_stats): UNKNOWN 원인별 관찰, PASS 근거, BIOS 날짜, 공식 단종일."""
    pairs, chips = pd.read_csv(OUT / "pair_labels.csv"), pd.read_csv(OUT / "chipsets.csv")
    cpu_chip = pairs[pairs["edge_type"] == "CPU_CHIPSET"]
    base = dict(zip(chips["chipset_id"], chips["base_chipset_id"]))
    basis = (cpu_chip.assign(dst=cpu_chip["dst"].map(lambda c: base.get(c, c)))
             .groupby(["src", "dst"])["basis_verified"].all().rename("basis_verified").reset_index())
    cc = _cpu_chipset_by_base(cpu_chip, chips).merge(basis, on=["src", "dst"], how="left")
    return {**validation_stats.unknown_cause_tests(cc), **validation_stats.pass_basis(cc),
            **validation_stats.bios_date_check(pd.read_csv(OUT / "observed_combos.csv"), cc,
                                               pd.read_csv(OUT / "parts.csv"), chips),
            **validation_stats.eol_stats(chips)}


def collect() -> dict[str, Any]:
    n: dict[str, Any] = {}
    for part in (sheet_status(), verification_numbers(), rejudge_numbers(), node_numbers(), label_numbers(), provenance_numbers(), split_numbers(), analysis_numbers(),
                 ablation_numbers(), review_numbers(), validation_numbers(), gpu_warning_numbers()):
        n.update(part)
    n["min_observed_reports"] = int(load_config()["release"]["min_observed_reports"])
    n["pending_sheets"] = n["sheet_support_done"] < n["sheet_support_total"] or n["sheet_verify_done"] < n["sheet_verify_total"]
    return n


def _fmt(key: str, value: Any, pending: bool) -> str:
    text = f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else str(value)
    is_count = key.startswith(PROVISIONAL_PREFIXES) and not key.endswith("_month")  # 달 이름에는 표시 안 함
    return text + PROVISIONAL_MARK if pending and is_count else text


def tex_escape(text: str) -> str:
    return "".join(TEX_SPECIALS.get(ch, ch) for ch in text)


def render(template: str, numbers: dict[str, Any], tex: bool = False) -> str:
    missing = sorted({k for k in PLACEHOLDER.findall(template) if k not in numbers})
    if missing:
        raise KeyError(f"템플릿 키가 numbers에 없음: {missing}")
    pending = bool(numbers.get("pending_sheets"))
    esc = tex_escape if tex else (lambda t: t)
    return PLACEHOLDER.sub(lambda m: esc(_fmt(m.group(1), numbers[m.group(1)], pending)), template)


def render_all(numbers: dict[str, Any], templates: Path = TEMPLATES, root: Path = PROJECT_ROOT) -> list[Path]:
    written = []
    for tmpl in sorted(t for g in TEMPLATE_GLOBS for t in templates.glob(g)):
        text = tmpl.read_text(encoding="utf-8")
        m = OUT_LINE.match(text)
        if not m:
            raise ValueError(f"{tmpl.name}: 첫 줄에 <!-- out: 경로 --> 또는 % out: 경로 가 없음")
        target = root / (m.group(1) or m.group(2))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(text[m.end():], numbers, tex=tmpl.suffix == ".tex"), encoding="utf-8")
        written.append(target)
    return written


def main() -> None:
    numbers = collect()
    (OUT / "numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    written = render_all(numbers)
    print(f"numbers.json {len(numbers)}개 키 / 잠정값 표시: {numbers['pending_sheets']} / 문서 {[p.name for p in written]}")


if __name__ == "__main__":
    main()
