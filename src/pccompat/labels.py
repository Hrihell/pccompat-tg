"""2일차 2단계: 쌍 호환 라벨과 완성 조합 라벨 생성 → data/out/pair_labels.csv, build_labels.parquet

- 쌍: CPU–칩셋, CPU–RAM, 칩셋–RAM(rules.py). 근거(evidence)와 규칙 ID, 판정 근거 값의 검증 여부(basis_verified).
- 유효 기간: 두 노드의 [출시일, 단종일(없으면 수집 범위 끝)] 교집합. 출시일을 모르면 비움.
- 조합: CPU × 칩셋 × RAM × GPU 전체. 라벨 = 세 쌍 중 가장 나쁜 것(FAIL > UNKNOWN > PASS).
  GPU는 PCIe 하위 호환이라 이번 범위에서 판정에 참여하지 않고 시간·가격·성능에만 쓴다(CPU-GPU 균형 보류).
- 분할: 노드는 시간 기준 family-group, 쌍·조합은 참여 노드 중 가장 늦은 분할(RAM은 시간 정보가 없어 제외).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from pccompat import rules
from pccompat.config import PROJECT_ROOT, load_config
from pccompat.split import ORDER, assign_time_group_split

OUT = PROJECT_ROOT / "data" / "out"
MANUAL = PROJECT_ROOT / "data" / "manual"
RULESET_VERSION = "pccompat-2026.10-v1"
CAUSES = ["BIOS_REQUIRED", "BOARD_DEPENDENT", "MISSING", "UNVERIFIED"]
RULE_IDS = ["HC-001", "HC-003", "HC-004", "HC-005"]
LABEL_CODE = {"PASS": 0, "UNKNOWN": 1, "FAIL": 2}
LABELS = ["PASS", "UNKNOWN", "FAIL"]


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _window(df: pd.DataFrame, end: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    start = _dates(df["release_date"])
    stop = _dates(df["eol_date"]).fillna(end) if "eol_date" in df else pd.Series(end, index=df.index)
    return start, stop.where(start.notna())


def _overlap(a_start, a_end, b_start, b_end) -> tuple[Any, Any]:
    if pd.isna(a_start) or pd.isna(b_start):
        return pd.NaT, pd.NaT
    lo, hi = max(a_start, b_start), min(a_end, b_end)
    return (lo, hi) if lo <= hi else (pd.NaT, pd.NaT)


def _verified_families() -> set[str]:
    facts = pd.read_csv(MANUAL / "family_facts.csv", dtype=str, keep_default_na=False)
    other_mem = facts.get("max_memory_source_url", pd.Series("", index=facts.index)).str.strip() != ""
    # 최대 메모리를 비공식 출처로만 확인한 세대는 판정 근거 전체가 검증된 것으로 보지 않는다
    has_url = facts["source_url"].str.strip() != ""  # provenance.py와 같은 기준: URL 없는 확인은 검증으로 치지 않음
    return set(facts.loc[(facts["verified"].str.lower() == "true") & has_url & ~other_mem, "family"])


def pair_labels(parts: pd.DataFrame, chips: pd.DataFrame, rams: pd.DataFrame, support: pd.DataFrame,
                cfg: dict[str, Any]) -> pd.DataFrame:
    end = pd.Timestamp(cfg["window"]["end"])
    verified = _verified_families()
    cpus = parts[parts["category"] == "CPU"].set_index("part_id")
    chips_i = chips.set_index("chipset_id")
    cs, ce = _window(cpus, end)
    hs, he = _window(chips_i, end)
    sup = support.set_index(["cpu_id", "chipset_id"])
    rows = []
    for cid, c in cpus.iterrows():
        for hid, h in chips_i.iterrows():
            key = (cid, hid)
            s_val = sup.at[key, "support"] if key in sup.index else None
            s_ver = bool(sup.at[key, "verified"]) if key in sup.index else False
            v = rules.cpu_chipset(c["socket"], h["socket"], s_val)
            lo, hi = _overlap(cs[cid], ce[cid], hs[hid], he[hid])
            # 소켓이 다르면 지원 근거 없이 FAIL이 정해지고, 같은 소켓이면 지원 근거(미지원 포함)가 검증돼야 한다
            socket_fail = pd.notna(c["socket"]) and pd.notna(h["socket"]) and c["socket"] != h["socket"]
            rows.append(_row("CPU_CHIPSET", cid, hid, v, lo, hi,
                             c["family"] in verified and h["family"] in verified and (socket_fail or s_ver)))
        for rid, r in rams.set_index("ram_id").iterrows():
            v = rules.cpu_ram(c["memory_gens"], c["max_memory_gb"], r["ddr_gen"], int(r["total_capacity_gb"]))
            rows.append(_row("CPU_RAM", cid, rid, v, cs[cid], ce[cid], c["family"] in verified))
    for hid, h in chips_i.iterrows():
        for rid, r in rams.set_index("ram_id").iterrows():
            v = rules.chipset_ram(h["memory_gen_options"], r["ddr_gen"])
            rows.append(_row("CHIPSET_RAM", hid, rid, v, hs[hid], he[hid], h["family"] in verified))
    out = pd.DataFrame(rows)
    out["ruleset_version"] = RULESET_VERSION
    return out


def _row(edge_type: str, src: str, dst: str, v: rules.Verdict, lo: Any, hi: Any, basis_verified: bool) -> dict:
    if pd.notna(lo) and pd.notna(hi) and lo > hi:  # 뒤집힌 노드 구간을 그대로 옮기지 않는다(_overlap과 같은 기준)
        lo, hi = pd.NaT, pd.NaT
    return {
        "edge_id": f"{src}|{dst}", "edge_type": edge_type, "src": src, "dst": dst, "label": v.label,
        "unknown_cause": v.cause, "evidence_type": "DOCUMENTED", "n_observed": None,
        "basis_verified": basis_verified, "rule_ids": ";".join(v.rule_ids),
        "evidence": json.dumps(v.evidence, ensure_ascii=False, default=str),
        "valid_from": None if pd.isna(lo) else lo.date().isoformat(),
        "valid_to": None if pd.isna(hi) else hi.date().isoformat(),
    }


def node_splits(parts: pd.DataFrame, chips: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, str]:
    model_level = set(cfg["split"].get("model_level_categories", []))
    grouped = parts.assign(release_date=_dates(parts["release_date"]),
                           split_group=parts["family"].where(~parts["category"].isin(model_level), parts["part_id"]))
    p = assign_time_group_split(grouped, {**cfg["split"], "group_column": "split_group"})
    c = assign_time_group_split(chips.assign(release_date=_dates(chips["release_date"])), cfg["split"])
    out = dict(zip(p["part_id"], p["split"]))
    out.update(zip(c["chipset_id"], c["split"]))
    return {k: v for k, v in out.items() if pd.notna(v)}


def build_labels(parts: pd.DataFrame, chips: pd.DataFrame, rams: pd.DataFrame, pairs: pd.DataFrame,
                 splits: dict[str, str], cfg: dict[str, Any],
                 warnings: tuple[np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    """CPU×칩셋×RAM×GPU 전체를 numpy로 펼친다(행 수 = 곱). 원인·실패 규칙은 비트마스크로 합친다."""
    end = np.datetime64(pd.Timestamp(cfg["window"]["end"]).date())
    cpus = parts[parts["category"] == "CPU"].reset_index(drop=True)
    gpus = parts[parts["category"] == "GPU"].reset_index(drop=True)
    chips = chips.reset_index(drop=True)
    rams = rams.reset_index(drop=True)
    nC, nH, nR, nG = len(cpus), len(chips), len(rams), len(gpus)

    def mats(edge: str, a_ids, b_ids):
        sub = pairs[pairs["edge_type"] == edge].set_index(["src", "dst"])
        lab = np.zeros((len(a_ids), len(b_ids)), np.int8)
        cause = np.zeros_like(lab, dtype=np.int16)
        fail = np.zeros_like(lab, dtype=np.int16)
        for i, a in enumerate(a_ids):
            for j, b in enumerate(b_ids):
                r = sub.loc[(a, b)]
                lab[i, j] = LABEL_CODE[r["label"]]
                if isinstance(r["unknown_cause"], str):
                    cause[i, j] = 1 << CAUSES.index(r["unknown_cause"])
                if r["label"] == "FAIL":
                    fail[i, j] = sum(1 << RULE_IDS.index(x) for x in r["rule_ids"].split(";"))
        return lab, cause, fail

    cc = mats("CPU_CHIPSET", cpus["part_id"], chips["chipset_id"])
    cr = mats("CPU_RAM", cpus["part_id"], rams["ram_id"])
    hr = mats("CHIPSET_RAM", chips["chipset_id"], rams["ram_id"])

    def window(df):
        s = _dates(df["release_date"]).to_numpy("datetime64[D]")
        e = _dates(df["eol_date"]).fillna(pd.Timestamp(end)).to_numpy("datetime64[D]") if "eol_date" in df else \
            np.full(len(df), end)
        return s, e

    (cS, cE), (hS, hE), (gS, gE) = window(cpus), window(chips), window(gpus)
    split_code = {k: ORDER[v] for k, v in splits.items()}
    cSp = np.array([split_code.get(x, -1) for x in cpus["part_id"]])
    hSp = np.array([split_code.get(x, -1) for x in chips["chipset_id"]])
    gSp = np.array([split_code.get(x, -1) for x in gpus["part_id"]])
    cM = pd.to_numeric(cpus["msrp_usd"], errors="coerce").to_numpy(float)
    gM = pd.to_numeric(gpus["msrp_usd"], errors="coerce").to_numpy(float)

    ci, hi_, ri, gi = np.meshgrid(np.arange(nC), np.arange(nH), np.arange(nR), np.arange(nG), indexing="ij")
    ci, hi_, ri, gi = ci.ravel(), hi_.ravel(), ri.ravel(), gi.ravel()
    label = np.maximum.reduce([cc[0][ci, hi_], cr[0][ci, ri], hr[0][hi_, ri]])
    cause = cc[1][ci, hi_] | cr[1][ci, ri] | hr[1][hi_, ri]
    cause = np.where(label == 1, cause, 0)  # FAIL이면 원인 표시 안 함(이미 판정됨)
    fail = cc[2][ci, hi_] | cr[2][ci, ri] | hr[2][hi_, ri]
    start = np.maximum.reduce([cS[ci], hS[hi_], gS[gi]])
    stop = np.minimum.reduce([cE[ci], hE[hi_], gE[gi]])
    known = ~(np.isnat(cS[ci]) | np.isnat(hS[hi_]) | np.isnat(gS[gi]))
    coexist = known & (start <= stop)
    sp = np.maximum.reduce([cSp[ci], hSp[hi_], gSp[gi]])
    sp = np.where((cSp[ci] < 0) | (hSp[hi_] < 0) | (gSp[gi] < 0), -1, sp)
    # 난이도: FAIL이 소켓·메모리 세대(규칙 하나로 결정)면 EASY, UNKNOWN이면 HARD, 나머지 MEDIUM
    easy_mask = (1 << RULE_IDS.index("HC-001")) | (1 << RULE_IDS.index("HC-004"))
    difficulty = np.where(label == 1, 2, np.where((label == 2) & ((fail & easy_mask) > 0), 0, 1))
    df = pd.DataFrame({
        "cpu_id": pd.Categorical.from_codes(ci, cpus["part_id"]),
        "chipset_id": pd.Categorical.from_codes(hi_, chips["chipset_id"]),
        "ram_id": pd.Categorical.from_codes(ri, rams["ram_id"]),
        "gpu_id": pd.Categorical.from_codes(gi, gpus["part_id"]),
        "label": pd.Categorical.from_codes(label, LABELS),
        "unknown_cause_mask": cause.astype(np.int8),
        "failed_rule_mask": fail.astype(np.int8),
        "valid_from": np.where(coexist, start, np.datetime64("NaT")),
        "valid_to": np.where(coexist, stop, np.datetime64("NaT")),
        "total_msrp_usd": cM[ci] + gM[gi],
        "difficulty": pd.Categorical.from_codes(difficulty, ["EASY", "MEDIUM", "HARD"]),
        "split": pd.Categorical.from_codes(np.where(sp < 0, -1, sp), list(ORDER)),
    })
    # G2 GPU 연결 경고(라벨과 별개). 비트 0 PCIe 대역폭. 미확인 = 플랫폼 사양을 사람이 아직 확인하지 않음.
    warn, unchecked = warnings if warnings is not None else (np.zeros((nC, nH, nG), np.int8),) * 2
    installable = label != 2  # 장착 자체가 FAIL인 조합에는 연결 경고가 의미 없다
    df["warn_mask"] = np.where(installable, warn[ci, hi_, gi], 0).astype(np.int8)
    df["warn_unchecked_mask"] = np.where(installable, unchecked[ci, hi_, gi], 0).astype(np.int8)
    df.insert(0, "build_id", np.arange(len(df), dtype=np.int64))
    return df


def decode_mask(mask: int, names: list[str]) -> list[str]:
    return [n for i, n in enumerate(names) if mask & (1 << i)]


def main() -> None:
    cfg = load_config()
    parts = pd.read_csv(OUT / "parts.csv")
    chips = pd.read_csv(OUT / "chipsets.csv")
    rams = pd.read_csv(OUT / "ram_specs.csv")
    support = pd.read_csv(OUT / "cpu_chipset_support.csv")
    pairs = pair_labels(parts, chips, rams, support, cfg)
    obs_path = OUT / "observed_pairs.csv"
    if obs_path.exists():  # linuxhw 관찰이 있으면 근거 유형을 BOTH로(라벨은 바꾸지 않음)
        from pccompat.observed import apply_to_pairs
        pairs = apply_to_pairs(pairs, rams, pd.read_csv(obs_path))
    splits = node_splits(parts, chips, cfg)
    sp = pairs.apply(lambda r: max((splits.get(r["src"]), splits.get(r["dst"])),
                                   key=lambda s: ORDER.get(s, -1) if s else -1), axis=1)
    pairs["split"] = sp.where(pairs.apply(lambda r: r["src"] in splits and (r["dst"] in splits or r["dst"].startswith("ram:")), axis=1))
    pairs.to_csv(OUT / "pair_labels.csv", index=False)
    from pccompat import gpu_platform as platform
    facts = platform.load_facts()
    cpus_, gpus_ = parts[parts["category"] == "CPU"].reset_index(drop=True), parts[parts["category"] == "GPU"].reset_index(drop=True)
    warnings = platform.warning_tables(cpus_, chips.reset_index(drop=True), gpus_, facts)
    builds = build_labels(parts, chips, rams, pairs, splits, cfg, warnings)
    builds.to_parquet(OUT / "build_labels.parquet", index=False)
    print(f"pair_labels {len(pairs):,}: " + str(pairs.groupby("edge_type")["label"].value_counts().unstack().fillna(0).astype(int).to_dict("index")))
    print(f"build_labels {len(builds):,}: {builds['label'].value_counts().to_dict()} / 시간상 공존 {int(builds['valid_from'].notna().sum()):,}")
    print("GPU 연결 경고(조합):", {w: int(((builds["warn_mask"].to_numpy() >> i) & 1).sum()) for i, w in enumerate(platform.WARNINGS)},
          "/ 미확인:", {w: int(((builds["warn_unchecked_mask"].to_numpy() >> i) & 1).sum()) for i, w in enumerate(platform.WARNINGS)})
    print("UNKNOWN 원인(조합):", {c: int(((builds["unknown_cause_mask"].to_numpy() >> i) & 1).sum()) for i, c in enumerate(CAUSES)})


if __name__ == "__main__":
    main()
