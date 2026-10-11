"""WP9: 공개 패키지(release/pccompat-tg-<버전>/)를 만들고 금지 항목 검사를 통과해야만 끝난다.

넣는 것: data/out의 공개 표, 모델별 실측 중앙값, 분할 ID 목록, 분석 결과·그림, 데이터 시트, 고지문, 라이선스.
빼는 것: 위키 원문·API 응답, linuxhw 원 보고서, Blender 원 표본, 내부 매칭표·제외 목록, 사람 입력 시트, _workspace.
linuxhw 관찰 조합은 min_observed_reports 이상 모인 묶음만 넣는다.

uv run python -m pccompat.release
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from pccompat.config import PROJECT_ROOT, load_config
from pccompat.fetch_files import sha256

OUT = PROJECT_ROOT / "data" / "out"
INTERIM = PROJECT_ROOT / "data" / "interim"

# (원본, 공개본 안의 경로)
TABLES = [
    (OUT / "parts.csv", "data/parts.csv"),
    (OUT / "chipsets.csv", "data/chipsets.csv"),
    (OUT / "ram_specs.csv", "data/ram_specs.csv"),
    (OUT / "cpu_chipset_support.csv", "data/cpu_chipset_support.csv"),
    (OUT / "pair_labels.csv", "data/pair_labels.csv"),
    (OUT / "build_labels.parquet", "data/build_labels.parquet"),
    (OUT / "field_provenance.csv", "data/field_provenance.csv"),
    (OUT / "observed_pairs.csv", "data/observed_pairs.csv"),
    (INTERIM / "perf_blender.csv", "data/perf_blender.csv"),
    (INTERIM / "perf_kishonti.csv", "data/perf_kishonti.csv"),
    (OUT / "splits" / "balanced_30k.parquet", "data/splits/balanced_30k.parquet"),
    (OUT / "numbers.json", "numbers.json"),
    (PROJECT_ROOT / "docs" / "DATASHEET.md", "DATASHEET.md"),
    (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
]
GLOBS = [(OUT / "analysis", "*.csv", "analysis"), (OUT / "figures", "*.png", "figures")]

FORBIDDEN_NAME = re.compile(
    r"(_workspace|/raw/|/interim/|/manual/|\.zip$|blender_samples|name_matches|excluded_wiki|"
    r"verification_|_todo\.csv$|cache_linuxhw|provenance_wiki|wikipedia/)", re.IGNORECASE)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
LOCAL_PATH = re.compile(r"/Users/|/home/[a-z]")
TEXT_SUFFIXES = {".csv", ".md", ".json", ".txt"}
# 공개 고지문에 원래 있어야 하는 주소(라이선스 문구 속 연락처 등)는 여기에 둔다. 지금은 없음.
ALLOWED_EMAILS: frozenset[str] = frozenset()


def aggregate_observed(combos: pd.DataFrame, min_reports: int) -> pd.DataFrame:
    """BIOS 버전·날짜를 빼고 (보드, 칩셋, CPU, RAM 세대)로 묶어 건수가 min_reports 이상인 것만 남긴다."""
    keys = ["board_vendor", "board_model", "chipset_id", "cpu_id", "ram_gen"]
    agg = (combos.fillna({"chipset_id": ""}).groupby(keys, as_index=False)
           .agg(n_reports=("n_reports", "sum"), source_commit=("source_commit", "first")))
    return agg[agg["n_reports"] >= min_reports].reset_index(drop=True)


def node_split_table(cfg: dict[str, Any]) -> pd.DataFrame:
    from pccompat.labels import node_splits
    parts, chips = pd.read_csv(OUT / "parts.csv"), pd.read_csv(OUT / "chipsets.csv")
    sp = node_splits(parts, chips, cfg)
    return pd.DataFrame({"node_id": list(sp.keys()), "split": list(sp.values())}).sort_values("node_id")


def check(root: Path) -> list[str]:
    """공개본 안의 금지 항목을 찾아 문제 목록으로 돌려준다(비어 있어야 통과)."""
    problems = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = "/" + path.relative_to(root).as_posix()
        if FORBIDDEN_NAME.search(rel):
            problems.append(f"금지 파일: {rel}")
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            emails = {m.group(0) for m in EMAIL.finditer(text)} - ALLOWED_EMAILS
            if emails:
                problems.append(f"이메일 주소: {rel} {sorted(emails)}")
            if LOCAL_PATH.search(text):
                problems.append(f"로컬 경로: {rel}")
    return problems


def _readme(version: str) -> str:
    return f"""# PCCompat-TG {version}

2020-10~2026-10 데스크톱 PC 부품(CPU·GPU·칩셋·RAM 규격)의 시간 호환성 그래프. 값마다 출처 URL과 출처 등급이 붙어 있다.

- 설명·수치: `DATASHEET.md`
- 표: `data/` (CSV, 조합 라벨은 Parquet)
- 분할: `data/splits/nodes.csv`(노드별 train/val/test), `data/splits/balanced_30k.parquet`
- 분석 결과: `analysis/`, `figures/`
- 파일 해시: `MANIFEST.csv`(sha256)

라이선스: 데이터 CC BY 4.0(https://creativecommons.org/licenses/by/4.0/). 원천별 표기는 `THIRD_PARTY_NOTICES.md`.
라벨은 제조사·위키 문서 기준 판정이며 실제 조립 성공을 보증하지 않는다.
권리자 요청: 어떤 값이 권리를 침해한다고 보면 저장소 이슈로 알려 주세요. 확인 후 해당 값을 지우고 다음 판에 반영합니다.
"""


def write_manifest(root: Path) -> Path:
    """공개본 파일 자체의 해시. 파이프라인 내부 run_manifest와 달리 공개본에서 바뀐 파일(집계한 관찰 조합 등)도 맞는다."""
    rows = [{"file": p.relative_to(root).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "MANIFEST.csv"]
    path = root / "MANIFEST.csv"
    pd.DataFrame(rows, columns=["file", "sha256", "bytes"]).to_csv(path, index=False)
    return path


def build(cfg: dict[str, Any]) -> tuple[Path, list[str]]:
    rel_cfg = cfg["release"]
    root = PROJECT_ROOT / rel_cfg["dir"]
    if root.exists():
        shutil.rmtree(root)
    for src, dst in TABLES:
        if not src.exists():
            raise FileNotFoundError(f"공개본 원본이 없음: {src} (run_all을 먼저 돌릴 것)")
        (root / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, root / dst)
    for folder, pattern, dst in GLOBS:
        (root / dst).mkdir(parents=True, exist_ok=True)
        for f in sorted(folder.glob(pattern)):
            shutil.copy2(f, root / dst / f.name)
    combos = pd.read_csv(OUT / "observed_combos.csv")
    aggregate_observed(combos, int(rel_cfg["min_observed_reports"])).to_csv(root / "data" / "observed_combos.csv", index=False)
    node_split_table(cfg).to_csv(root / "data" / "splits" / "nodes.csv", index=False)
    (root / "README.md").write_text(_readme(rel_cfg["version"]), encoding="utf-8")
    (root / "LICENSE").write_text(
        "Data: Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "https://creativecommons.org/licenses/by/4.0/legalcode\n\n"
        "Third-party sources keep their own licenses; see THIRD_PARTY_NOTICES.md.\n", encoding="utf-8")
    write_manifest(root)
    return root, check(root)


def main() -> None:
    root, problems = build(load_config())
    n_files = sum(1 for p in root.rglob("*") if p.is_file())
    if problems:
        raise SystemExit("공개본 검사 실패:\n" + "\n".join(problems))
    print(f"공개본 {root.relative_to(PROJECT_ROOT)}: 파일 {n_files}개, 금지 항목 0")


if __name__ == "__main__":
    main()
