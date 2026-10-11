"""데이터 품질 점검(템플릿 6항목을 이 데이터셋에 맞춰 적용). 실행: python -m pccompat.quality

1 결측 패턴: 필드×세대 결측률 + 결측 여부와 세대의 독립성 카이제곱(독립 기각 → MCAR 아님, 세대에 따른 MAR 근거).
  MNAR(값 자체 때문에 빠짐)은 관측만으로 판정할 수 없어 "검증 불가"로 보고한다.
2 이상치: 설정 범위(도메인) + 세대 내 robust z(중앙값·MAD) |z|>3.5 → 표시만, 삭제 없음.
3 중복·불일치: ID 중복, 정규화 이름 충돌(서로 다른 ID가 같은 이름), 두 출처(위키 vs GetDeploying) 값 일치율.
4 분포 편향: 제조사×카테고리, 출시 연도, 출처 등급 분포(전체·필드별).
5 누수: v1에는 예측 대상(target)이 없다. 대신 시간·분할 누수를 점검한다
  (미래 정보 열이 특징 목록에 없는지, 한 계열이 여러 분할에 걸치지 않는지).
6 수정 제안: 각 점검 결과 옆에 suggestion 열로 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pccompat.names import Catalog, MatchStatus, NameMatcher, normalize
from pccompat.split import assign_time_group_split, family_key

SPEC_FIELDS = ["cores", "threads", "boost_clock_mhz", "tdp_w", "l3_cache_mb", "shader_count",
               "fp32_tflops", "pcie_gen", "pcie_lanes", "release_date", "msrp_usd"]
# 클래스 불균형: 가장 작은 라벨이 이 비율보다 작으면 평가 때 정확도 대신 클래스별 지표·균형 표본을 쓰라고 제안한다.
MINORITY_WARN_SHARE = 0.10
FUTURE_COLUMNS = ["eol_date", "eol_is_assumed", "valid_to"]


@dataclass
class QualityReport:
    sections: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        out = ["# 데이터 품질 점검 보고서", ""]
        out += [f"- {n}" for n in self.notes] + [""]
        for title, df in self.sections.items():
            out += [f"## {title}", "", df.to_markdown(index=True) if not df.empty else "(해당 없음)", ""]
        return "\n".join(out)


def _applicable(parts: pd.DataFrame, f: str) -> pd.Series:
    """카테고리상 의미 있는 행만(GPU의 코어 수·CPU의 셰이더 수는 결측이 아니라 해당 없음)."""
    cpu_only = {"cores", "threads", "l3_cache_mb"}
    gpu_only = {"shader_count", "fp32_tflops", "pcie_gen", "pcie_lanes"}  # pcie_*: GPU 버스 인터페이스
    if f in cpu_only:
        return parts["category"] == "CPU"
    if f in gpu_only:
        return parts["category"] == "GPU"
    return pd.Series(True, index=parts.index)


def missingness(parts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rates, tests = {}, []
    for f in SPEC_FIELDS:
        if f not in parts.columns:
            continue
        sub = parts[_applicable(parts, f)]
        miss = sub[f].isna()
        rates[f] = miss.groupby(sub["family"]).mean()
        table = pd.crosstab(sub["family"], miss)
        if table.shape[1] == 2 and table.shape[0] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(table)
            verdict = "MCAR 기각(세대에 따라 결측 → MAR 근거)" if p < 0.05 else "세대와 독립(MCAR과 모순 없음)"
        else:
            p, verdict = np.nan, "결측 없음 또는 한쪽만" if table.shape[1] < 2 else "세대 1개"
        tests.append({"field": f, "missing_rate": round(float(miss.mean()), 3), "chi2_p": p, "verdict": verdict,
                      "suggestion": "세대 단위로 빠진 원천(표에 열 없음)을 사람 조회 시트로" if p == p and p < 0.05 else ""})
    return pd.DataFrame(rates).round(2), pd.DataFrame(tests).set_index("field")


def outliers(parts: pd.DataFrame, ranges: dict[str, list[float]], z_cut: float = 3.5) -> pd.DataFrame:
    rows = []
    for f, (low, high) in ranges.items():
        if f not in parts.columns:
            continue
        v = pd.to_numeric(parts[f], errors="coerce")
        for i in parts.index[(v < low) | (v > high)]:
            rows.append({"part_id": parts.at[i, "part_id"], "field": f, "value": v[i], "rule": f"범위 {low}~{high}",
                         "suggestion": "원출처 재확인(삭제하지 않음)"})
    for f in ["msrp_usd", "boost_clock_mhz", "tdp_w", "fp32_tflops"]:
        if f not in parts.columns:
            continue
        v = pd.to_numeric(parts[f], errors="coerce")
        for fam, idx in parts.groupby("family").groups.items():
            x = v.loc[idx].dropna()
            if len(x) < 5:
                continue
            mad = (x - x.median()).abs().median()
            if mad == 0:
                continue
            z = 0.6745 * (x - x.median()) / mad
            for i in z.index[z.abs() > z_cut]:
                rows.append({"part_id": parts.at[i, "part_id"], "field": f, "value": v[i],
                             "rule": f"세대 내 robust z={z[i]:.1f}", "suggestion": "원출처 재확인(플래그십이면 정상일 수 있음)"})
    return pd.DataFrame(rows, columns=["part_id", "field", "value", "rule", "suggestion"])


def duplicates(parts: pd.DataFrame, match_cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for pid in parts.loc[parts["part_id"].duplicated(keep=False), "part_id"].unique():
        rows.append({"kind": "ID 중복", "detail": pid, "suggestion": "변형 중복 제거 규칙 확인"})
    norm = parts["model_name"].map(lambda n: normalize(n, match_cfg))
    clash = parts.assign(norm=norm).groupby("norm")["part_id"].nunique()
    for n in clash[clash > 1].index:
        ids = sorted(parts.loc[norm == n, "part_id"])
        rows.append({"kind": "정규화 이름 충돌", "detail": f"{n}: {ids}", "suggestion": "이름 정규화 규칙 또는 원천 확인"})
    return pd.DataFrame(rows, columns=["kind", "detail", "suggestion"])


def cross_source_agreement(parts: pd.DataFrame, other: pd.DataFrame, field_map: dict[str, str],
                           match_cfg: dict[str, Any], rel_tol: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
    """다른 출처(예: GetDeploying)의 행을 확정 매칭으로 연결해 필드별 일치율을 낸다(상대 오차 rel_tol 이내 = 일치)."""
    gpus = parts[parts["category"] == "GPU"]
    matcher = NameMatcher(Catalog.from_rows(((r.part_id, r.model_name, []) for r in gpus.itertuples()), match_cfg), match_cfg)
    pairs = []
    for _, o in other.iterrows():
        r = matcher.match(str(o["name"]))
        if r.status is MatchStatus.CONFIRMED:
            pairs.append((r.model_id, o))
    detail, summary = [], []
    mine = gpus.set_index("part_id")
    for ours, theirs in field_map.items():
        agree = total = 0
        for pid, o in pairs:
            a, b = mine.at[pid, ours], o[theirs]
            if pd.isna(a) or pd.isna(b):
                continue
            total += 1
            if ours == "release_date":
                ok = str(a)[:10] == str(b)[:10]
            else:
                fa, fb = float(a), float(b)
                ok = abs(fa - fb) <= rel_tol * max(abs(fa), abs(fb), 1e-9)
            agree += ok
            if not ok:
                detail.append({"part_id": pid, "field": ours, "wikipedia": a, "other": b,
                               "suggestion": "공식 데이터시트로 판정 후 verified 기록"})
        summary.append({"field": ours, "pairs": total, "agree": agree,
                        "agreement": round(agree / total, 3) if total else np.nan})
    return pd.DataFrame(summary).set_index("field"), pd.DataFrame(detail)


def distribution(parts: pd.DataFrame, provenance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    year = pd.to_datetime(parts["release_date"], errors="coerce").dt.year
    return {
        "제조사×카테고리": pd.crosstab(parts["vendor"], parts["category"]),
        "출시 연도": pd.crosstab(year.fillna(-1).astype(int).replace(-1, "결측"), parts["category"]),
        "출처 등급(전체)": provenance["source_tier"].value_counts().to_frame("n").assign(
            share=lambda d: (d["n"] / d["n"].sum()).round(3)),
        "출처 등급(필드별)": pd.crosstab(provenance["field"], provenance["source_tier"]),
    }


def label_balance(units: dict[str, pd.Series]) -> pd.DataFrame:
    """라벨 클래스 분포(단위별). 소수 클래스가 MINORITY_WARN_SHARE 미만이면 평가 방법을 제안한다."""
    rows = []
    for unit, labels in units.items():
        counts = labels.astype(str).value_counts()
        n = {lab: int(counts.get(lab, 0)) for lab in ("PASS", "UNKNOWN", "FAIL")}
        share = min(n.values()) / max(sum(n.values()), 1)
        rows.append({"unit": unit, **n, "minority_share": round(share, 3),
                     "suggestion": "평가는 클래스별 재현율·macro-F1로, 학습·평가 표본은 균형 표본(balanced_30k) 사용"
                     if share < MINORITY_WARN_SHARE else ""})
    return pd.DataFrame(rows)


def leakage(parts: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    feats = set(cfg["features"]["numeric"]) | set(cfg["features"]["categorical"])
    bad = sorted(feats & set(FUTURE_COLUMNS + ["release_date"]))
    rows.append({"check": "미래 정보 열이 특징에 없음", "ok": not bad, "detail": str(bad),
                 "suggestion": "config.features에서 제거" if bad else ""})
    split = assign_time_group_split(parts.assign(release_date=pd.to_datetime(parts["release_date"], errors="coerce")),
                                    cfg["split"])
    spans = split.assign(k=family_key(split["family"])).dropna(subset=["split"]).groupby("k")["split"].nunique()
    multi = sorted(spans[spans > 1].index)
    rows.append({"check": "계열이 여러 분할에 걸치지 않음", "ok": not multi, "detail": str(multi),
                 "suggestion": "family-group 분할 규칙 확인" if multi else ""})
    unassigned = int(split["split"].isna().sum())
    rows.append({"check": "분할 미배정 행", "ok": unassigned == 0, "detail": str(unassigned),
                 "suggestion": "출시일 보완(release_dates_todo.csv)" if unassigned else ""})
    return pd.DataFrame(rows).set_index("check")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """정확도의 95% Wilson 신뢰구간(YAGO 방식). 표본이 작아도 0~1 밖으로 나가지 않는다."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(center - half, 4), round(center + half, 4))


def accuracy_by_tier(checked: pd.DataFrame) -> pd.DataFrame:
    """사람이 채운 검증 표본(match = true/false)에서 출처 등급별 정확도와 Wilson 구간."""
    done = checked[checked["match"].astype(str).str.lower().isin(["true", "false"])]
    rows = []
    for tier, g in done.groupby("source_tier"):
        k = int((g["match"].astype(str).str.lower() == "true").sum())
        lo, hi = wilson_interval(k, len(g))
        rows.append({"source_tier": tier, "n": len(g), "correct": k, "accuracy": round(k / len(g), 3),
                     "wilson_low": lo, "wilson_high": hi})
    return pd.DataFrame(rows, columns=["source_tier", "n", "correct", "accuracy", "wilson_low", "wilson_high"])


def verification_sample(provenance: pd.DataFrame, parts: pd.DataFrame, n: int, seed: int,
                        min_per_tier: int = 10) -> pd.DataFrame:
    """출처 등급별 최소 min_per_tier건(필드를 고르게) + 나머지는 등급×필드 층에서 무작위로 n건까지.

    등급별 정확도를 Wilson 구간으로 보고하려고 등급을 첫 기준으로 층화한다(provenance-practices.md 관행 4).
    """
    prov = provenance[provenance["entity_id"].isin(parts["part_id"])].merge(
        parts[["part_id", "family", "model_name"]], left_on="entity_id", right_on="part_id")
    prov = prov[prov["field"].isin(SPEC_FIELDS)]
    rng = np.random.default_rng(seed)
    picked: list[Any] = []
    for _, tier_rows in prov.groupby("source_tier"):
        by_field = [list(rng.permutation(idx)) for _, idx in tier_rows.groupby("field").groups.items()]
        quota = min(min_per_tier, len(tier_rows))
        while quota > 0:  # 필드를 돌아가며 하나씩
            for pool in by_field:
                if pool and quota > 0:
                    picked.append(pool.pop())
                    quota -= 1
    pools = [list(rng.permutation(idx)) for _, idx in prov.drop(index=picked).groupby(["source_tier", "field"]).groups.items()]
    rng.shuffle(pools)
    while len(picked) < n and any(pools):  # 층을 돌아가며 하나씩(층 크기 비례가 아니라 고르게)
        for pool in pools:
            if pool and len(picked) < n:
                picked.append(pool.pop())
    out = prov.loc[picked, ["entity_id", "model_name", "family", "field", "value", "source_url", "source_tier"]]
    for c in ["official_value", "official_url", "match", "checked_by", "checked_at", "note"]:
        out[c] = ""
    return out.sort_values(["source_tier", "family", "field"]).reset_index(drop=True)


def run(parts: pd.DataFrame, provenance: pd.DataFrame, cfg: dict[str, Any],
        other_gpu: pd.DataFrame | None = None, label_units: dict[str, pd.Series] | None = None) -> QualityReport:
    rep = QualityReport()
    rates, tests = missingness(parts)
    rep.sections["1. 결측률(세대별)"] = rates
    rep.sections["1. 결측 패턴 검정"] = tests
    rep.notes.append("MNAR(값 자체 때문에 빠짐)은 관측 데이터만으로 판정할 수 없다 — 사람 대조 표본으로 간접 점검.")
    rep.sections["2. 이상치(표시만, 삭제 없음)"] = outliers(parts, cfg["ranges"])
    rep.sections["3. 중복·이름 충돌"] = duplicates(parts, cfg["matching"])
    if other_gpu is not None:
        summary, detail = cross_source_agreement(
            parts, other_gpu, {"shader_count": "shader_cores", "tdp_w": "tdp_w", "release_date": "release_date",
                               "fp32_tflops": "fp32_dense_tflops"}, cfg["matching"])
        rep.sections["3. 두 출처 일치율(위키 vs GetDeploying)"] = summary
        rep.sections["3. 두 출처 불일치 목록"] = detail
    for title, df in distribution(parts, provenance).items():
        rep.sections[f"4. {title}"] = df
    if label_units:
        rep.sections["4. 라벨 클래스 분포(불균형)"] = label_balance(label_units)
    rep.sections["5. 누수 점검(v1은 예측 대상 없음 → 시간·분할 누수)"] = leakage(parts, cfg)
    return rep


def _label_units(out_dir: Any) -> dict[str, pd.Series]:
    """쌍 유형별 라벨과 논문 주 단위(소켓·보드 메모리가 맞는 CPU×칩셋×RAM) 라벨."""
    units: dict[str, pd.Series] = {}
    pairs_path, builds_path = out_dir / "pair_labels.csv", out_dir / "build_labels.parquet"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path, usecols=["edge_type", "label"])
        units.update({f"쌍({t})": g["label"] for t, g in pairs.groupby("edge_type")})
    if builds_path.exists():
        from pccompat.paper_stats import SOCKET_BIT
        b = pd.read_parquet(builds_path, columns=["cpu_id", "chipset_id", "ram_id", "label", "failed_rule_mask"])
        chips = pd.read_csv(out_dir / "chipsets.csv", usecols=["chipset_id", "board_memory"])
        rams = pd.read_csv(out_dir / "ram_specs.csv", usecols=["ram_id", "ddr_gen"])
        board = b["chipset_id"].astype(str).map(dict(zip(chips["chipset_id"], chips["board_memory"])))
        gen = b["ram_id"].astype(str).map(dict(zip(rams["ram_id"], rams["ddr_gen"])))
        keep = ((b["failed_rule_mask"].astype(int) & SOCKET_BIT) == 0) & ~(board.notna() & (board != gen))
        units["조합(소켓·보드 메모리 일치, CPU×칩셋×RAM)"] = b[keep].drop_duplicates(["cpu_id", "chipset_id", "ram_id"])["label"]
        units["조합(전체, GPU 포함)"] = b["label"]
    return units


def main() -> None:
    from pccompat.config import PROJECT_ROOT, load_config

    cfg = load_config()
    interim = PROJECT_ROOT / "data" / "interim"
    out_dir = PROJECT_ROOT / "data" / "out"
    # 공개 산출물이 있으면 그것을 점검한다(WP0 통합 출처). 없으면 위키 중간 산출물.
    parts = pd.read_csv(out_dir / "parts.csv") if (out_dir / "field_provenance.csv").exists() \
        else pd.read_csv(interim / "parts_wiki.csv")
    prov = pd.read_csv(out_dir / "field_provenance.csv") if (out_dir / "field_provenance.csv").exists() \
        else pd.read_csv(interim / "provenance_wiki.csv")
    gd = PROJECT_ROOT / "data" / "raw" / "getdeploying" / "gpu-models.csv"
    other = pd.read_csv(gd) if gd.exists() else None
    rep = run(parts, prov, cfg, other, _label_units(out_dir))
    out = PROJECT_ROOT / "data" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "quality_report.md").write_text(rep.to_markdown(), encoding="utf-8")
    sample_path = PROJECT_ROOT / "data" / "manual" / "verification_sample.csv"
    existing = pd.read_csv(sample_path, dtype=str, keep_default_na=False) if sample_path.exists() else None
    if existing is not None and (existing["match"].str.strip() != "").any():
        sample = existing  # 사람이 채우기 시작한 표본은 절대 덮어쓰지 않는다
        acc = accuracy_by_tier(existing)
        (out / "verification_accuracy.csv").write_text(acc.to_csv(index=False), encoding="utf-8")
    else:
        sample = verification_sample(prov, parts, n=60, seed=int(cfg["seed"]), min_per_tier=10)
        sample.to_csv(sample_path, index=False)
    print(f"보고서: {out / 'quality_report.md'} / 검증 표본 {len(sample)}건: data/manual/verification_sample.csv")


if __name__ == "__main__":
    main()
