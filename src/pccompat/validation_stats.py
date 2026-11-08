"""심사 2차 대응: 라벨·시간 구간 검증 수치(report.collect가 numbers.json에 합친다).

- v_obs_*: UNKNOWN 원인별 실제 사용 관찰률, PASS·FAIL 대비와 BIOS 필요 vs 나머지 원인 Fisher 검정(7개 검정 Holm 보정)
- v_pass_*: PASS 쌍 가운데 사람이 공식 페이지에서 확인하지 않은 근거(위키 칩셋 표)의 비율
- v_bios_after_*: 관찰 보고의 BIOS 빌드 날짜가 CPU 출시일 이후인 비율(BIOS 필요 vs PASS)
- v_eol_*: 공식 단종일을 확인한 칩셋 수와 가정 단종일의 오차(개월)
- a_cl_*: 관찰률 차이(PASS−UNKNOWN, UNKNOWN−FAIL)의 CPU 단위 군집 부트스트랩(같은 CPU가 여러 쌍에 들어가는 점 고려)
CPU–칩셋 쌍은 analysis._cpu_chipset_by_base로 분할 전 칩셋 기준으로 합친 표를 받는다(논문 e 분석과 같은 분모).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pccompat.paper_stats import _fisher, _p, _sci, holm_adjusted

CAUSES = ("BIOS_REQUIRED", "BOARD_DEPENDENT", "MISSING")
DAYS_PER_MONTH = 30.4375
NA = "—"
# 라벨 감사 층별 표본 수(합 100). 층이 이보다 작으면 전부 뽑는다.


def _rate(k: int, n: int) -> str:
    if not n:
        return NA
    pct = k / n * 100
    return f"{pct:.2f}%" if 0 < pct < 1 else f"{pct:.1f}%"  # 0.03%가 0.0%로 뭉개지지 않게


def _kn(rows: pd.DataFrame, col: str = "observed") -> tuple[int, int]:
    return int(rows[col].sum()), len(rows)


def unknown_cause_tests(cc: pd.DataFrame) -> dict[str, Any]:
    d = cc.assign(observed=cc["evidence_type"] == "BOTH")
    pas, fail = _kn(d[d["label"] == "PASS"]), _kn(d[d["label"] == "FAIL"])
    n: dict[str, Any] = {"v_obs_pass_rate": _rate(*pas), "v_obs_fail_rate": _rate(*fail),
                         "v_obs_pass_n": pas[1], "v_obs_fail_n": fail[1]}
    groups = {c: _kn(d[(d["label"] == "UNKNOWN") & (d["unknown_cause"] == c)]) for c in CAUSES}
    tested = [c for c in CAUSES if groups[c][1] and pas[1] and fail[1]]
    keys = [(c, "pass") for c in tested] + [(c, "fail") for c in tested]
    raw = [_fisher(groups[c], pas) for c in tested] + [_fisher(groups[c], fail) for c in tested]
    # 원인끼리의 차이: BIOS 필요 vs 나머지 UNKNOWN 원인(같은 검정군에서 Holm 보정)
    rest = _kn(d[(d["label"] == "UNKNOWN") & (d["unknown_cause"] != "BIOS_REQUIRED")])
    bios = groups["BIOS_REQUIRED"]
    if bios[1] and rest[1]:
        keys.append(("BIOS_REQUIRED", "rest"))
        raw.append(_fisher(bios, rest))
    adj = dict(zip(keys, holm_adjusted(np.array(raw)))) if raw else {}
    n.update({"v_obs_unk_rest_k": rest[0], "v_obs_unk_rest_n": rest[1], "v_obs_unk_rest_rate": _rate(*rest),
              "v_obs_n_tests": len(raw),
              "v_obs_bios_vs_rest_p": _p(adj[("BIOS_REQUIRED", "rest")]) if ("BIOS_REQUIRED", "rest") in adj else NA})
    for c in CAUSES:
        key, (k, total) = f"v_obs_{c.lower()}", groups[c]
        n.update({f"{key}_k": k, f"{key}_n": total, f"{key}_rate": _rate(k, total)})
        if c in tested:
            mant, exp = _sci(adj[(c, "fail")])
            n.update({f"{key}_p_pass": _p(adj[(c, "pass")]), f"{key}_p_fail_mant": mant, f"{key}_p_fail_exp": exp})
        else:
            n.update({f"{key}_p_pass": NA, f"{key}_p_fail_mant": NA, f"{key}_p_fail_exp": "0"})
    return n


def cluster_bootstrap(cc: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    """CPU를 복원 추출해 그 CPU의 쌍을 통째로 넣는다. 점추정과 95% 백분위 구간(소수 셋째 자리)."""
    d = cc.assign(observed=cc["evidence_type"] == "BOTH")

    def diffs(rows: pd.DataFrame) -> dict[str, float]:
        rate = {lab: rows.loc[rows["label"] == lab, "observed"].mean() for lab in ("PASS", "UNKNOWN", "FAIL")}
        return {"pu": rate["PASS"] - rate["UNKNOWN"], "uf": rate["UNKNOWN"] - rate["FAIL"]}

    point = diffs(d)
    rng = np.random.default_rng(seed)
    cpus = d["src"].unique()
    groups = dict(tuple(d.groupby("src")))
    boot: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        sample = pd.concat([groups[c] for c in rng.choice(cpus, len(cpus))])
        for k, val in diffs(sample).items():
            if np.isfinite(val):  # 한 라벨이 표본에서 빠지면 그 회차는 건너뛴다
                boot[k].append(val)
    n: dict[str, Any] = {}
    for k in point:
        n.update({f"a_cl_{k}": f"{point[k]:.3f}", f"a_cl_{k}_lo": f"{np.percentile(boot[k], 2.5):.3f}",
                  f"a_cl_{k}_hi": f"{np.percentile(boot[k], 97.5):.3f}"})
    return n


def pass_basis(cc: pd.DataFrame) -> dict[str, Any]:
    pas = cc[cc["label"] == "PASS"]
    unverified = int((~pas["basis_verified"].astype(bool)).sum())
    return {"v_pass_unverified_n": unverified, "v_pass_total": len(pas),
            "v_pass_unverified_pct": _rate(unverified, len(pas))}


def bios_date_check(combos: pd.DataFrame, cc: pd.DataFrame, parts: pd.DataFrame, chips: pd.DataFrame) -> dict[str, Any]:
    """보고서 BIOS 빌드 날짜 ≥ CPU 출시일인 비율. BIOS 필요 쌍에서 높아야 라벨과 모순이 없다(강한 증거는 아님)."""
    base = dict(zip(chips["chipset_id"], chips["base_chipset_id"]))
    released = dict(zip(parts["part_id"], pd.to_datetime(parts["release_date"])))
    obs = combos[combos["chipset_id"].fillna("").astype(str).str.len() > 0].assign(
        dst=lambda d: d["chipset_id"].map(lambda c: base.get(c, c)),
        bios=lambda d: pd.to_datetime(d["bios_date"], errors="coerce"),
        released=lambda d: d["cpu_id"].map(released))
    obs = obs.merge(cc[["src", "dst", "label", "unknown_cause"]], left_on=["cpu_id", "dst"], right_on=["src", "dst"])
    obs = obs.dropna(subset=["bios", "released"]).assign(after=lambda d: d["bios"] >= d["released"])
    bios_rows = obs[(obs["label"] == "UNKNOWN") & (obs["unknown_cause"] == "BIOS_REQUIRED")]
    bios = _kn(bios_rows, "after")
    pas = _kn(obs[obs["label"] == "PASS"], "after")
    n: dict[str, Any] = {}
    for name, (k, total) in (("bios", bios), ("pass", pas)):
        n.update({f"v_bios_after_{name}_k": k, f"v_bios_after_{name}_n": total, f"v_bios_after_{name}_rate": _rate(k, total)})
    n["v_bios_after_bios_pairs"] = int(bios_rows[["cpu_id", "dst"]].drop_duplicates().shape[0])  # 행은 (보드, CPU, BIOS) 조합
    n["v_bios_after_p"] = _p(_fisher(bios, pas)) if bios[1] and pas[1] else NA
    return n


def eol_stats(chips: pd.DataFrame) -> dict[str, Any]:
    """분할 전 칩셋 기준. 단종일을 가정했던 칩셋(후속 세대가 없는 최신 칩셋은 제외) 가운데
    공식 단종일로 바뀐 수와, 그 칩셋들의 (공식 − 가정) 오차 개월."""
    base = chips.drop_duplicates("base_chipset_id")
    base = base[base["eol_assumed_date"].notna()]
    official = base[~base["eol_is_assumed"].astype(bool) & base["eol_assumed_date"].notna()]
    err = (pd.to_datetime(official["eol_date"]) - pd.to_datetime(official["eol_assumed_date"])).dt.days / DAYS_PER_MONTH
    n: dict[str, Any] = {"v_eol_official_n": len(official), "v_eol_total": len(base)}
    if len(err):
        n.update({"v_eol_err_median_mo": f"{err.median():.1f}", "v_eol_err_min_mo": f"{err.min():.1f}",
                  "v_eol_err_max_mo": f"{err.max():.1f}"})
    else:
        n.update({"v_eol_err_median_mo": NA, "v_eol_err_min_mo": NA, "v_eol_err_max_mo": NA})
    return n
