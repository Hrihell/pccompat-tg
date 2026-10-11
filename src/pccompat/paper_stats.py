"""논문 리뷰 대응용 검정·재집계 수치(report.collect가 numbers.json에 합친다).

- s_*: 출처 등급 간 정확도 차이의 Fisher 정확검정(필드 혼동을 줄인 출시일 내부 비교 포함)
- x_same_*: 소켓 불일치로 자명하게 FAIL인 조합을 뺀 재집계(같은 소켓, GPU 제외 단위 포함)
- x_obs_p_*: 실제 사용 관찰 비율이 라벨 사이에서 다른지(Fisher)
- d_*: 대리 지표 비교 전체의 분포(Holm 보정)
- x_chiplevel_*: 보드 메모리 변형을 칩셋 단위로 되돌려 센 UNKNOWN 비율(분할 전과 비교용)
- x_memmatch_*: 보드 메모리와 램 세대가 다른 변형(뻔한 FAIL)을 뺀 단위
이미 만든 산출물만 다시 센다. 새 실험은 없다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pccompat.labels import RULE_IDS

CITED = ("WIKIPEDIA_CITED_CELL", "WIKIPEDIA_CITED_ROW")
SOCKET_BIT = 1 << RULE_IDS.index("HC-001")
ALPHA = 0.05


def _p(p: float) -> str:
    return f"{p:.2g}"


def _sci(p: float) -> tuple[str, str]:
    mant, exp = f"{p:.1e}".split("e")
    return mant, str(int(exp))


def _counts(rows: pd.DataFrame) -> tuple[int, int]:
    return int((rows["match"] == "true").sum()), len(rows)


def _fisher(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1])


def tier_tests(sample: pd.DataFrame) -> dict[str, Any]:
    d = sample.assign(match=sample["match"].str.lower())
    d = d[d["match"].isin(["true", "false"])]
    cited, uncited = _counts(d[d["source_tier"].isin(CITED)]), _counts(d[d["source_tier"] == "WIKIPEDIA_UNCITED"])
    family = _counts(d[d["source_tier"] == "WIKIPEDIA_FAMILY"])
    rel = d[d["field"] == "release_date"]
    rel_other, rel_family = _counts(rel[rel["source_tier"] != "WIKIPEDIA_FAMILY"]), _counts(rel[rel["source_tier"] == "WIKIPEDIA_FAMILY"])
    fam_err = d[(d["source_tier"] == "WIKIPEDIA_FAMILY") & (d["match"] == "false")]
    n: dict[str, Any] = {"s_cited": f"{cited[0]}/{cited[1]}", "s_uncited": f"{uncited[0]}/{uncited[1]}",
                         "s_family": f"{family[0]}/{family[1]}", "s_rel_other": f"{rel_other[0]}/{rel_other[1]}",
                         "s_rel_family": f"{rel_family[0]}/{rel_family[1]}",
                         "s_family_fields": d.loc[d["source_tier"] == "WIKIPEDIA_FAMILY", "field"].nunique(),
                         "s_family_err_n": len(fam_err), "s_family_err_families": fam_err["family"].nunique()}
    ok = all(x[1] for x in (cited, uncited, family, rel_other, rel_family))
    n["s_p_cited_family"] = _p(_fisher(cited, family)) if ok else "—"
    n["s_p_cited_uncited"] = _p(_fisher(cited, uncited)) if ok else "—"
    n["s_p_release_family"] = _p(_fisher(rel_other, rel_family)) if ok else "—"
    return n


def same_socket(builds: pd.DataFrame) -> dict[str, Any]:
    socket_fail = (builds["failed_rule_mask"].astype(int) & SOCKET_BIT) > 0
    n: dict[str, Any] = {"x_socket_fail": int(socket_fail.sum()),
                         "x_socket_fail_share": f"{socket_fail.sum() / max((builds['label'] == 'FAIL').sum(), 1) * 100:.1f}%"}
    same = builds[~socket_fail]
    triples = same.drop_duplicates(["cpu_id", "chipset_id", "ram_id"])
    co = same[same["valid_from"].notna()]
    n["x_same_triples"] = len(triples)
    n["x_same_coexist"] = len(co)
    for lab in ("PASS", "UNKNOWN", "FAIL"):
        n[f"x_same_triples_{lab.lower()}"] = int((triples["label"] == lab).sum())
        n[f"x_same_coexist_{lab.lower()}_pct"] = f"{(co['label'] == lab).mean() * 100:.1f}%" if len(co) else "—"
    return n


def observed_tests(crosstab: pd.DataFrame) -> dict[str, Any]:
    def row(label: str) -> tuple[int, int]:
        return (int(crosstab.loc[label, "BOTH"]), int(crosstab.loc[label, "All"])) if label in crosstab.index else (0, 0)
    unk, pas, fail = row("UNKNOWN"), row("PASS"), row("FAIL")
    if not (unk[1] and pas[1] and fail[1]):
        return {"x_obs_p_unknown_pass": "—", "x_obs_p_unknown_fail_mant": "—", "x_obs_p_unknown_fail_exp": "0"}
    mant, exp = _sci(_fisher(unk, fail))
    return {"x_obs_p_unknown_pass": _p(_fisher(unk, pas)), "x_obs_p_unknown_fail_mant": mant, "x_obs_p_unknown_fail_exp": exp}


def _pct(s: pd.Series, lab: str) -> str:
    return f"{(s == lab).mean() * 100:.1f}%" if len(s) else "—"


def _collapse(rows: pd.DataFrame, keys: list[str]) -> pd.Series:
    """같은 칩셋의 보드 메모리 변형들을 하나로: 라벨이 모두 같으면 그 라벨, 다르면 보드마다 다름(UNKNOWN)."""
    g = rows.groupby(keys, observed=True)["label"]
    return pd.Series(np.where(g.nunique() == 1, g.first(), "UNKNOWN"))


def board_memory_stats(builds: pd.DataFrame, chips: pd.DataFrame, rams: pd.DataFrame) -> dict[str, Any]:
    base = dict(zip(chips["chipset_id"], chips.get("base_chipset_id", chips["chipset_id"])))
    board = dict(zip(chips["chipset_id"], chips.get("board_memory", pd.Series(pd.NA, index=chips.index))))
    gen = dict(zip(rams["ram_id"], rams["ddr_gen"]))
    same = builds[(builds["failed_rule_mask"].astype(int) & SOCKET_BIT) == 0]
    same = same.assign(label=same["label"].astype(str), base=same["chipset_id"].astype(str).map(base))
    bm = same["chipset_id"].astype(str).map(board)
    mismatch = bm.notna() & (bm != same["ram_id"].astype(str).map(gen))
    co = same["valid_from"].notna()
    triples = same.drop_duplicates(["cpu_id", "chipset_id", "ram_id"])
    chip_triples = _collapse(triples, ["cpu_id", "base", "ram_id"])
    chip_co = _collapse(same[co], ["cpu_id", "base", "ram_id", "gpu_id"])
    match = same[~mismatch]
    mt = match.drop_duplicates(["cpu_id", "chipset_id", "ram_id"])["label"]
    mc = match.loc[match["valid_from"].notna(), "label"]
    n: dict[str, Any] = {"x_chiplevel_same_unknown_pct": _pct(chip_triples, "UNKNOWN"),
                         "x_chiplevel_coexist_unknown_pct": _pct(chip_co, "UNKNOWN"),
                         "x_memmismatch_builds": int(mismatch.sum()),
                         "x_memmatch_triples": len(mt), "x_memmatch_coexist": len(mc)}
    for lab in ("PASS", "UNKNOWN", "FAIL"):
        n[f"x_memmatch_triples_{lab.lower()}"] = int((mt == lab).sum())
        n[f"x_memmatch_coexist_{lab.lower()}_pct"] = _pct(mc, lab)
    return n


def holm_significant(p_values: np.ndarray, alpha: float = ALPHA) -> int:
    p = np.sort(np.asarray(p_values, dtype=float))
    adjusted = np.maximum.accumulate(np.minimum(1.0, (len(p) - np.arange(len(p))) * p))
    return int((adjusted < alpha).sum())


def proxy_distribution(validity: pd.DataFrame) -> dict[str, Any]:
    n: dict[str, Any] = {"d_holm_sig": holm_significant(validity["p_value"].to_numpy()),
                         "d_rho_median": f"{validity['rho'].median():.3f}", "d_n_negative": int((validity["rho"] < 0).sum())}
    for (target, source), g in validity.groupby(["target", "source"]):
        key = f"d_{target}_{source}".lower()
        n.update({f"{key}_count": len(g), f"{key}_rho_min": f"{g['rho'].min():.3f}", f"{key}_rho_max": f"{g['rho'].max():.3f}"})
    neg = validity[validity["rho"] < 0]
    n["d_negative_sources"] = ", ".join(sorted(neg["source"].unique())) if len(neg) else "없음"
    return n
