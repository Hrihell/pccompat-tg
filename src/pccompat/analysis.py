"""3일차 분석 a·b·c·e + 분할 보고·균형 부분집합(WP1). DuckDB로 parquet을 직접 질의한다.

a 조합 수: 월마다 유효 기간이 겹치는 조합 수(라벨별), 부품별 참여 조합 수.
   칩셋 단종일 민감도: (나) 다음 소켓 세대 칩셋 첫 출시일(기본) vs 열린 구간(B450처럼 계속 판매).
b 업그레이드 여지: 칩셋별 호환(PASS) CPU 수, 그 CPU들의 첫~마지막 출시 간격(개월). BIOS·보드 의존(UNKNOWN)은 따로 센다.
c 가격 대비 성능 최적선: 월마다 PASS·공존 조합을 (CPU, GPU) 쌍으로 줄이고(칩셋·RAM은 가격·성능 없음)
   출시가 합↓, CPU 대리 성능↑, GPU 대리 성능↑ 3목표 파레토 최적 쌍을 센다.
e 공식 vs 관찰: 문서 라벨 × linuxhw 관찰 교차표, 문서 FAIL인데 관찰된 쌍, 관찰로 확인 필요를 해소할 후보(n≥3).
"""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pandas as pd

from pccompat.config import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "data" / "out"
AN = OUT / "analysis"
FIG = OUT / "figures"
# 논문 2단 폭(91.5mm = 3.6in)에 1:1로 넣는 그림. 양식 규정(글자 9pt 이상)을 그림 안 글자에도 맞춘다.
PAPER_FIG_WIDTH_IN = 3.6
PAPER_FIG_FONT_PT = 9


def months(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), pd.Timestamp(end), freq="MS")


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW b AS SELECT * FROM read_parquet('{OUT / 'build_labels.parquet'}')")
    con.execute(f"CREATE VIEW parts AS SELECT * FROM read_csv_auto('{OUT / 'parts.csv'}')")
    con.execute(f"CREATE VIEW chips AS SELECT * FROM read_csv_auto('{OUT / 'chipsets.csv'}')")
    return con


# ---------- WP1 분할 보고·균형 부분집합 ----------
def split_report(con, cfg) -> pd.DataFrame:
    from pccompat.labels import node_splits
    parts, chips = con.execute("SELECT * FROM parts").df(), con.execute("SELECT * FROM chips").df()
    sp = node_splits(parts, chips, cfg)
    p = parts.assign(split=parts["part_id"].map(sp))
    rep = p.groupby("category")["split"].value_counts().unstack(fill_value=0)
    rep["unassigned"] = p.groupby("category")["split"].apply(lambda s: int(s.isna().sum()))
    builds = con.execute("SELECT split, label, count(*) n FROM b GROUP BY 1, 2").df().pivot(
        index="split", columns="label", values="n").fillna(0).astype(int)
    return rep, builds


def balanced_subset(con, n: int, seed: int, label_mix: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시간상 공존하는 조합만. 라벨 목표 비율로 나누고, 라벨 안에서는 존재하는 난이도 층에 고르게 배분.

    난이도는 라벨에서 일부 결정되므로(UNKNOWN=HARD, PASS=MEDIUM) label×difficulty 곱 목표는 구조적으로 채울 수 없다
    (리뷰 반영). 층이 모자라면 있는 만큼만 뽑고 보고한다.
    """
    rng = np.random.default_rng(seed)
    pool = con.execute("SELECT build_id, label, difficulty, split FROM b WHERE valid_from IS NOT NULL").df()
    picks, rows = [], []
    for lab, lw in label_mix.items():
        cells = {d: g for d, g in pool[pool["label"] == lab].groupby("difficulty", observed=True)}
        target = int(round(n * lw))
        share = {d: target // len(cells) for d in cells} if cells else {}
        for d in list(share)[: target - sum(share.values())]:
            share[d] += 1
        for d, g in cells.items():
            k = min(share[d], len(g))
            picks.append(g.iloc[rng.choice(len(g), k, replace=False)])
            rows.append({"label": lab, "difficulty": d, "target": share[d], "available": len(g), "picked": k})
    sub = pd.concat(picks) if picks else pool.iloc[0:0]
    return sub.sort_values("build_id").reset_index(drop=True), pd.DataFrame(rows)


# ---------- a ----------
def monthly_counts(con, cfg, open_chipset: bool = False) -> pd.DataFrame:
    end = str(cfg["window"]["end"])
    chip_end = f"DATE '{end}'" if open_chipset else f"coalesce(h.eol_date, DATE '{end}')"
    q = f"""
    WITH w AS (
      SELECT b.label,
             greatest(c.release_date, h.release_date, g.release_date) AS s,
             least(coalesce(c.eol_date, DATE '{end}'), {chip_end}, coalesce(g.eol_date, DATE '{end}')) AS e
      FROM b JOIN parts c ON b.cpu_id = c.part_id JOIN chips h ON b.chipset_id = h.chipset_id
             JOIN parts g ON b.gpu_id = g.part_id
      WHERE c.release_date IS NOT NULL AND h.release_date IS NOT NULL AND g.release_date IS NOT NULL)
    SELECT m.month, w.label, count(*) AS n
    FROM w JOIN (SELECT unnest(generate_series(DATE '{cfg['window']['start']}', DATE '{end}', INTERVAL 1 MONTH))::DATE AS month) m
      ON w.s <= last_day(m.month) AND w.e >= m.month AND w.s <= w.e
    GROUP BY 1, 2 ORDER BY 1, 2"""
    df = con.execute(q).df().pivot(index="month", columns="label", values="n").fillna(0).astype(int)
    return df


def participation(con) -> pd.DataFrame:
    q = """
    SELECT part_id, kind, label, count(*) AS n FROM (
      SELECT cpu_id AS part_id, 'CPU' AS kind, label FROM b WHERE valid_from IS NOT NULL
      UNION ALL SELECT gpu_id, 'GPU', label FROM b WHERE valid_from IS NOT NULL
      UNION ALL SELECT chipset_id, 'CHIPSET', label FROM b WHERE valid_from IS NOT NULL)
    GROUP BY 1, 2, 3"""
    return con.execute(q).df().pivot_table(index=["part_id", "kind"], columns="label", values="n", fill_value=0).reset_index()


# ---------- b ----------
def upgrade_window(pairs: pd.DataFrame, parts: pd.DataFrame, chips: pd.DataFrame) -> pd.DataFrame:
    if "base_chipset_id" in chips:  # 보드 메모리 변형은 CPU 지원이 같으므로 칩셋 하나로 센다(첫 변형을 대표로)
        chips = chips.drop_duplicates("base_chipset_id").assign(name=lambda d: d["base_name"])
        pairs = pairs[~pairs["dst"].isin(set(pairs["dst"]) - set(chips["chipset_id"])) | (pairs["edge_type"] != "CPU_CHIPSET")]
    cc = pairs[pairs["edge_type"] == "CPU_CHIPSET"].merge(
        parts[["part_id", "release_date", "family"]], left_on="src", right_on="part_id")
    cc["release_date"] = pd.to_datetime(cc["release_date"])
    rows = []
    for chip, g in cc.groupby("dst"):
        ok = g[g["label"] == "PASS"]
        unk = g[g["label"] == "UNKNOWN"]
        span = (ok["release_date"].max() - ok["release_date"].min()).days / 30.44 if len(ok) and ok["release_date"].notna().any() else np.nan
        rows.append({"chipset_id": chip, "n_cpu_pass": len(ok), "n_cpu_unknown": len(unk),
                     "n_cpu_unknown_board_or_bios": int(unk["unknown_cause"].isin(["BOARD_DEPENDENT", "BIOS_REQUIRED"]).sum()),
                     "first_cpu": ok["release_date"].min(), "last_cpu": ok["release_date"].max(),
                     "span_months": round(span, 1) if span == span else np.nan,
                     "n_cpu_families_pass": ok["family"].nunique()})
    out = pd.DataFrame(rows).merge(chips[["chipset_id", "name", "family", "release_date", "eol_date"]], on="chipset_id")
    return out.sort_values(["family", "name"])


# ---------- c ----------
def pareto_mask(price: np.ndarray, perf1: np.ndarray, perf2: np.ndarray) -> np.ndarray:
    """최소화(price), 최대화(perf1, perf2). 다른 점에 약하게 모두 지고 하나 이상 엄격히 지면 지배됨."""
    n = len(price)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = (price <= price[i]) & (perf1 >= perf1[i]) & (perf2 >= perf2[i]) & \
              ((price < price[i]) | (perf1 > perf1[i]) | (perf2 > perf2[i]))
        keep[i] = not dom.any()
    return keep


def pareto_by_month(con, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = str(cfg["window"]["end"])
    pairs = con.execute(f"""
      SELECT DISTINCT b.cpu_id, b.gpu_id, b.valid_from, b.valid_to, c.msrp_usd AS cp, g.msrp_usd AS gp,
             c.proxy_perf AS cperf, g.proxy_perf AS gperf
      FROM b JOIN parts c ON b.cpu_id = c.part_id JOIN parts g ON b.gpu_id = g.part_id
      WHERE b.label = 'PASS' AND b.valid_from IS NOT NULL
      ORDER BY b.cpu_id, b.gpu_id, b.valid_from, b.valid_to""").df()  # 정렬 고정 → 재실행 해시 일치
    pairs = pairs.dropna(subset=["cp", "gp", "cperf", "gperf"])
    rows, members = [], []
    for m in months(str(cfg["window"]["start"]), end):
        me = m + pd.offsets.MonthEnd(0)
        cur = pairs[(pairs["valid_from"] <= me) & (pairs["valid_to"] >= m)].drop_duplicates(["cpu_id", "gpu_id"])
        if cur.empty:
            rows.append({"month": m, "n_pairs": 0, "n_frontier": 0})
            continue
        mask = pareto_mask((cur["cp"] + cur["gp"]).to_numpy(float), cur["cperf"].to_numpy(float), cur["gperf"].to_numpy(float))
        front = cur[mask]
        rows.append({"month": m, "n_pairs": len(cur), "n_frontier": len(front),
                     "min_price": float((front["cp"] + front["gp"]).min())})
        members.append(front.assign(month=m)[["month", "cpu_id", "gpu_id"]])
    mem = pd.concat(members) if members else pd.DataFrame(columns=["month", "cpu_id", "gpu_id"])
    return pd.DataFrame(rows), mem


# ---------- e ----------
def _cpu_chipset_by_base(cc: pd.DataFrame, chips: pd.DataFrame | None) -> pd.DataFrame:
    """보드 메모리 변형의 CPU–칩셋 쌍은 라벨이 같으므로 분할 전 칩셋 하나로 합친다(관찰 수는 더함). 두 번 세면 비율이 왜곡된다."""
    if chips is None or "base_chipset_id" not in chips:
        return cc
    base = dict(zip(chips["chipset_id"], chips["base_chipset_id"]))
    d = cc.assign(dst=cc["dst"].map(lambda x: base.get(x, x)))
    return (d.groupby(["src", "dst"], as_index=False)
            .agg(edge_type=("edge_type", "first"), label=("label", "first"), unknown_cause=("unknown_cause", "first"),
                 evidence=("evidence", "first"), n_observed=("n_observed", lambda s: s.sum(min_count=1)),
                 evidence_type=("evidence_type", lambda s: "BOTH" if (s == "BOTH").any() else "DOCUMENTED")))


def official_vs_observed(pairs: pd.DataFrame, chips: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cc = _cpu_chipset_by_base(pairs[pairs["edge_type"] == "CPU_CHIPSET"], chips)
    table = pd.crosstab(cc["label"], cc["evidence_type"], margins=True)
    fail_obs = cc[(cc["label"] == "FAIL") & (cc["evidence_type"] == "BOTH")][["src", "dst", "n_observed", "evidence"]]
    resolve = cc[(cc["label"] == "UNKNOWN") & (cc["n_observed"].fillna(0) >= 3)][
        ["src", "dst", "unknown_cause", "n_observed"]].sort_values("n_observed", ascending=False)
    return table, fail_obs, resolve


def upgrade_groups(upg: pd.DataFrame) -> pd.DataFrame:
    """같은 소켓에서 지원 구간이 같은 칩셋을 한 줄로 묶는다(논문 2단 폭에 칩셋 38개 막대는 읽히지 않음)."""
    total = upg.groupby("family")["chipset_id"].count().rename("n_total")
    keys = ["family", "first_cpu", "last_cpu", "n_cpu_families_pass", "span_months"]
    g = (upg.dropna(subset=["span_months"]).groupby(keys, as_index=False)
         .agg(n_chipsets=("name", "count"), names=("name", lambda s: "·".join(sorted(s)))))
    g = g.join(total, on="family")
    g["first_cpu"], g["last_cpu"] = pd.to_datetime(g["first_cpu"]), pd.to_datetime(g["last_cpu"])
    return g.sort_values(["first_cpu", "family"]).reset_index(drop=True)


def _upgrade_figure(plt, upg: pd.DataFrame) -> None:
    g = upgrade_groups(upg)
    fig, ax = plt.subplots(figsize=(PAPER_FIG_WIDTH_IN, 2.3))
    for i, r in g.iterrows():
        ax.barh(i, r["last_cpu"] - r["first_cpu"], left=r["first_cpu"], height=0.6, color=f"C{i}")
        ax.text(r["last_cpu"], i, f" {int(r['n_cpu_families_pass'])} gen",
                va="center", fontsize=PAPER_FIG_FONT_PT)
    labels = [f"{r.family} ({r.n_chipsets}/{r.n_total})" for r in g.itertuples()]
    ax.set_yticks(range(len(g)), labels, fontsize=PAPER_FIG_FONT_PT); ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=PAPER_FIG_FONT_PT)
    ax.set_xlim(pd.Timestamp("2020-07-01"), pd.Timestamp("2028-12-31"))
    years = range(2021, 2029, 2)  # 9pt에서 매년 눈금은 겹친다
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in years], [f"'{y % 100:02d}" for y in years])
    ax.set_xlabel("release of first / last PASS CPU", fontsize=PAPER_FIG_FONT_PT)
    fig.tight_layout(); fig.savefig(FIG / "b_upgrade.png", dpi=300); plt.close(fig)


def _figures(monthly: pd.DataFrame, monthly_open: pd.DataFrame, upg: pd.DataFrame, par: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for lab, st in (("PASS", "-"), ("UNKNOWN", "--")):
        if lab in monthly:
            ax.plot(monthly.index, monthly[lab], st, label=f"{lab} (chipset EOL = successor)")
        if lab in monthly_open:
            ax.plot(monthly_open.index, monthly_open[lab], st, alpha=0.4, label=f"{lab} (chipset open-ended)")
    ax.set_ylabel("valid builds"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "a_monthly.png", dpi=200)
    plt.close(fig)
    _upgrade_figure(plt, upg)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(par["month"], par["n_frontier"], label="Pareto-optimal (CPU, GPU) pairs")
    ax.plot(par["month"], par["n_pairs"], alpha=0.4, label="all PASS pairs")
    ax.set_yscale("log"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "c_pareto.png", dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    AN.mkdir(parents=True, exist_ok=True)
    (OUT / "splits").mkdir(parents=True, exist_ok=True)
    con = _con()
    rep, builds = split_report(con, cfg)
    rep.to_csv(AN / "split_nodes.csv"); builds.to_csv(AN / "split_builds.csv")
    sub, sub_rep = balanced_subset(con, 30000, int(cfg["seed"]), {"PASS": 0.4, "FAIL": 0.3, "UNKNOWN": 0.3})
    sub.to_parquet(OUT / "splits" / "balanced_30k.parquet", index=False); sub_rep.to_csv(AN / "balanced_report.csv", index=False)
    monthly = monthly_counts(con, cfg); monthly_open = monthly_counts(con, cfg, open_chipset=True)
    monthly.to_csv(AN / "a_monthly.csv"); monthly_open.to_csv(AN / "a_monthly_open_chipset.csv")
    participation(con).to_csv(AN / "a_participation.csv", index=False)
    pairs = pd.read_csv(OUT / "pair_labels.csv")
    upg = upgrade_window(pairs, pd.read_csv(OUT / "parts.csv"), pd.read_csv(OUT / "chipsets.csv"))
    upg.to_csv(AN / "b_upgrade.csv", index=False)
    par, mem = pareto_by_month(con, cfg)
    par.to_csv(AN / "c_pareto_monthly.csv", index=False); mem.to_csv(AN / "c_pareto_members.csv", index=False)
    table, fail_obs, resolve = official_vs_observed(pairs, pd.read_csv(OUT / "chipsets.csv"))
    table.to_csv(AN / "e_crosstab.csv"); fail_obs.to_csv(AN / "e_fail_but_observed.csv", index=False)
    resolve.to_csv(AN / "e_unknown_resolution_candidates.csv", index=False)
    _figures(monthly, monthly_open, upg, par)
    print("분할(노드):\n", rep.to_string())
    print("균형 부분집합:", len(sub), sub_rep[["label", "difficulty", "target", "available", "picked"]].to_dict("records"))
    peak = monthly.sum(axis=1).idxmax()
    print(f"a: 최대 공존 월 {peak.date()} / 그 달 PASS {int(monthly.loc[peak].get('PASS', 0)):,} "
          f"(열린 구간이면 {int(monthly_open.loc[peak].get('PASS', 0)):,})")
    print("b:", upg[["name", "n_cpu_pass", "n_cpu_unknown", "span_months"]].to_string(index=False))
    print(f"c: 최적선 쌍 수 중앙값 {par['n_frontier'].median()} / 최대 {par['n_frontier'].max()}")
    print("e:\n", table.to_string(), "\n문서 FAIL·관찰:", len(fail_obs), "/ 해소 후보(n≥3):", len(resolve))


if __name__ == "__main__":
    main()
