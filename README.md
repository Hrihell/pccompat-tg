# PCCompat-TG

2020-10~2026-10 데스크톱 PC 부품(CPU·GPU·칩셋·RAM 규격)의 시간 호환성 그래프 데이터셋과 이를 만드는 파이프라인. 값마다 출처 URL과 출처 등급이 붙어 있다.

## 구성

| 경로 | 내용 | 라이선스 |
| --- | --- | --- |
| `release/pccompat-tg-2026.10/` | 공개 데이터셋(표·조합 라벨·분할·분석 결과·그림, `MANIFEST.csv` sha256) | CC BY 4.0 (폴더 안 `LICENSE`) |
| `src/`, `tests/`, `config/` | 정제·라벨·분석·공개본 생성 파이프라인 | MIT (`LICENSE`) |
| `docs/` | 데이터 시트, 파이프라인·출처·라이선스 설명 | CC BY 4.0 |

원천별 표기는 `THIRD_PARTY_NOTICES.md`. 데이터셋 설명과 수치는 `release/pccompat-tg-2026.10/DATASHEET.md`.

라벨은 제조사·위키 문서 기준 판정이며 실제 조립 성공을 보증하지 않는다.

## 실행

```bash
uv sync
uv run pytest -q
uv run python -m pccompat.run_all      # 조립 → 라벨 → 출처 → 품질 → 분석 → 수치·문서 생성
uv run python -m pccompat.release      # 공개본 생성 + 금지 항목 검사
```

내려받은 원본·중간 산출물(`data/raw`, `data/interim`, `data/out`)과 사람 입력 시트(`data/manual`)는 저장소에 넣지 않는다. 원본은 `config/files.yml`·`config/wikipedia.yml`의 출처에서 다시 받는다.

## 권리자 요청

어떤 값이 권리를 침해한다고 보면 저장소 이슈로 알려 주세요. 확인 후 해당 값을 지우고 다음 판에 반영합니다.
