"""WP0: 재실행 체인. 사람 입력(시트)이 바뀌면 이것 하나로 공개 산출물을 다시 만든다.

uv run python -m pccompat.run_all            # assemble → labels → provenance → quality → 분석 → report
uv run python -m pccompat.run_all --observed # linuxhw 관찰 조합 재집계 포함(3.4GB 순회, 수 분)
끝나면 data/out/run_manifest.csv에 산출 파일별 sha256을 남긴다(같은 입력 → 같은 해시면 재현).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

from pccompat import analysis, analysis_d, assemble, labels, observed, provenance, quality, report
from pccompat.config import PROJECT_ROOT
from pccompat.fetch_files import sha256

OUT = PROJECT_ROOT / "data" / "out"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observed", action="store_true", help="linuxhw 관찰 조합도 다시 집계")
    args = ap.parse_args(argv)
    steps = [("assemble", assemble.main)]
    if args.observed or not (OUT / "observed_combos.csv").exists():
        steps.append(("observed", observed.main))
    else:  # 칩셋 노드(보드 메모리 변형)가 바뀌어도 쌍 관찰이 낡지 않게 집계본에서 다시 센다
        steps.append(("observed_pairs", observed.refresh_pairs))
    steps += [("labels", labels.main), ("provenance", provenance.main), ("quality", quality.main),
              ("analysis_d", analysis_d.main), ("analysis", analysis.main)]
    steps.append(("report", report.main))  # 수치 → numbers.json, 문서 템플릿 → DATASHEET·논문 초안
    for name, fn in steps:
        print(f"== {name}")
        fn()
    rows = [{"file": p.relative_to(OUT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "run_manifest.csv"]
    with open(OUT / "run_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256", "bytes"])
        w.writeheader()
        w.writerows(rows)
    print(f"완료 {datetime.now(timezone.utc).isoformat(timespec='seconds')} / 산출 {len(rows)}개 해시 기록")


if __name__ == "__main__":
    main()
