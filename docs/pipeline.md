# PCCompat-TG 정제 파이프라인 설계

> 코드: `src/pccompat/` · 설정: `config/pipeline.yml`(값·컬럼명은 모두 여기서 읽음) · 테스트: `tests/`(49개)
> 설계 근거: `_workspace/research/pccompat-dataset-design.md`

## 데이터

- **원천.** 위키백과 사실값(API), 제조사 공식 페이지(사람이 확인), Blender Open Data(퍼블릭 도메인), Kishonti GFXBench·CompuBench(CC BY 4.0), 손 입력한 CPU–칩셋 지원 기록
- **대상 테이블(8개).** `parts`, `chipsets`, `ram_specs`, `cpu_chipset_support`, `pair_labels`, `build_labels`, `perf_measurements`, `field_provenance`
- **원천 CSV의 형태.** 모두 문자열 열이다. list 필드는 `;`로 구분한다(예: `DDR4;DDR5`).

## 해결할 문제

1. 문자열로 들어온 스펙을 스키마 타입으로 바꾼다. 실패하면 조용히 버리지 않고 보고한다.
2. 결측을 **채우지 않고** 드러낸다. 규칙 엔진이 결측 값을 UNKNOWN/MISSING으로 판정해야 하기 때문이다.
3. 범위 밖 값은 지우지 않고 사람이 확인할 목록으로 보낸다.
4. 사용자가 업로드한 실측 표본에서 측정 이상치를 잘라낸 뒤 모델별 중앙값을 낸다.
5. 형제 모델 누수가 없게 시간 기준으로 분할한다.
6. v2 모델 평가용 특징 행렬은 train에만 fit한다.

---

## 1. 파이프라인 단계

템플릿 순서는 EDA → 결측 → 이상치 → 타입 → 인코딩 → 분할 → 스케일링이다. 이 데이터에 맞춰 아래처럼 실행 순서를 바꿨고, 해당하지 않는 단계는 그렇다고 표시했다.

| # | 단계 | 목적 | 입력 → 출력 | 주의점 |
|---|---|---|---|---|
| 1 | **EDA** (`validate.profile`) | 정제 전 상태 기록 | 원본 DF → 컬럼별 건수·결측률·고유값·수치 요약 | 원본은 문자열이다. 값의 90% 이상이 숫자로 읽히는 열만 수치로 요약한다. |
| 2 | **타입** (`clean.coerce_types`) | 스키마 타입으로 변환 | 원본 → 타입 변환된 DF + `issues` | **결측·이상치보다 먼저** 한다. 문자열 상태로는 범위 검사를 할 수 없고, 파싱 실패가 새 결측을 만들기 때문이다. 실패는 행·열·값과 함께 보고한다. 키 결측과 키 중복도 여기서 잡는다. |
| 3 | **결측** (`clean.flag_missing`) | 결측 드러내기 | 타입 DF → `record_status`, `missing_required`, 결측 보고서 | **대치하지 않는다.** 카테고리별 필수 필드(`REQUIRED_BY_CATEGORY`)가 비면 `PARTIAL`로 표시한다. |
| 3' | **단종일** (`clean.assign_eol`) | 단종일 가정 적용 | 타입 DF → `eol_date`, `eol_is_assumed` | **비어 있는 칸만** 같은 계열 다음 세대의 최소 출시일로 채운다. 출처 값은 덮어쓰지 않는다. 다음 세대 행이 없으면 비워 둔다. eol_date는 미래 정보이므로 특징에 넣지 않는다. |
| 4 | **이상치** (`clean.flag_out_of_range`, `clean.trim_samples_iqr`) | 스펙: 확인 대상 표시 / 실측: 업로드 이상치 제거 | 스펙 → `review_flags`(행 유지). 실측 원표본 → 유지 표본 + 그룹별 제거 수 | 스펙 값은 지우지 않는다. 실측 표본만 모델별 IQR(k=1.5)로 자른다. 표본이 low_n(10) 미만이거나 IQR=0이면 자르지 않는다. NaN·키 없음은 이상치와 따로 센다(`len(kept)+n_removed+n_nan+n_no_key = 전체`). |
| 5 | **인코딩** (`features`, v2 전용) | 모델 베이스라인 입력 | 특징 DF → 원-핫 | **공개 CSV에는 적용하지 않는다.** 처음 보는 범주는 0 벡터로 둔다. |
| 6 | **분할** (`split`) | 누수 없는 평가 분할 | 노드 → `split`. 간선·조합은 최신 노드의 분할을 물려받는다. | 계열 첫 출시일로 계열 전체를 묶는다. 계열명은 공백·대소문자 무시. 자기 출시일이 없어도 계열을 따른다. 기준 날짜가 없으면 비우고 `unassigned`로 보고. |
| 7 | **스케일링** (`features`, v2 전용) | 수치 특징 정규화 | train fit → val/test transform | RobustScaler를 쓴다. 플래그십 제품 때문에 분포 꼬리가 길다. 결측 표시 열은 모든 수치 열에 둔다(`features="all"`). 정제 결과의 pd.NA는 변환기 입구에서 np.nan으로 바꾼다. |
| 8 | **검증** (`validate.compare_profiles`, `distribution_shift`) | 전후·분할 간 비교 | 전/후 프로파일 → 차이표, KS·TVD | 결측이 늘었으면 파싱 실패, 평균이 이동했으면 단위 오류를 의심한다. |

## 2. 재현성과 누수 방지

- **시드:** `config.seed = 20261012`. 균형 부분집합 추출과 테스트 난수에 쓴다. 분할 자체는 날짜 규칙이라 무작위성이 없다.
- **fit은 train에만 한다.** `build_feature_transformer(...).fit(train)` 뒤 val·test는 `transform`만 한다. 테스트 `TestFeatureLeakage`가 대치 중앙값이 train 중앙값인지 검사한다.
- **형제 모델 누수:** family-group 분할로 막는다. 7800X3D와 7950X3D는 같은 분할에 들어간다.
- **조합 누수:** 조합의 분할은 참여 노드 중 가장 늦은 분할이다. 그래서 test 조합에는 test 노드가 반드시 들어간다.
- **실측 값은 라벨 계산에 쓰지 않는다.** 성능 값과 호환 라벨은 입력이 분리돼 있다.
- **불변성:** 모든 단계는 입력 DF를 바꾸지 않고 새 DF를 돌려준다. 테스트 `test_input_frame_is_not_mutated`가 확인한다.

## 3. 코드

### 사용 예

```python
import pandas as pd
from pccompat.config import load_config
from pccompat.clean import CleaningPipeline
from pccompat.schema import TABLES
from pccompat.split import assign_time_group_split, split_ratios

cfg = load_config()
raw = pd.read_csv("data/interim/parts_raw.csv", dtype=str, keep_default_na=False)

result = CleaningPipeline(cfg).run(raw, TABLES["parts"])
parts = assign_time_group_split(result.frame, cfg["split"])

result.reports["issues"]        # 타입 변환 실패·키 중복 → 사람이 고칠 목록
result.reports["comparison"]    # 정제 전후 건수·결측·평균
parts[parts["review_flags"].map(len) > 0]   # 범위 밖 값 → 원출처 재확인
split_ratios(parts)             # 60/15/25 목표와 비교(허용 오차 ±5%p)
```

### 결측 처리 (`clean.py`에서 발췌)

```python
def flag_missing(df: pd.DataFrame, table: Table) -> tuple[pd.DataFrame, pd.DataFrame]:
    """결측은 채우지 않는다. 필수 필드 결측 행은 PARTIAL + missing_required 목록으로 표시."""
    # 근거: 스펙 값은 사실이다. 추정으로 채우면 규칙 엔진이 "모르는 것을 아는 척"하는 라벨을 만든다.
    # 비워 두면 판정이 UNKNOWN/MISSING이 되고, 이 결과가 데이터셋의 3값 라벨 설계와 맞는다.
```

### 이상치 처리 (`clean.py`에서 발췌)

```python
def flag_out_of_range(df, ranges):        # 스펙: 범위(config.ranges, 설계값) 밖 → review_flags, 행 유지
    # 근거: 범위 밖 값이 출처 오기인지, 실제 극단값(예: 신형 플래그십)인지는 원출처를 봐야 안다.
def trim_samples_iqr(samples, group_col, value_col, k):   # 실측 원표본만 모델별 IQR 절단
    # 근거: 사용자 업로드 표본의 극단값은 측정 설정 차이·오류일 가능성이 높다.
    # 중앙값 전에 잘라내되, 잘라낸 수를 그룹별로 반환해 데이터 시트에 보고한다.
```

### 분할 (`split.py`에서 발췌)

```python
first = out.groupby(group_col)[date_col].transform("min")   # 계열 첫 출시일
out["split"] = first.map(lambda d: _bucket(d, train_end, val_end))
```

### 인코딩·스케일링 (`features.py`, v2 전용)

```python
ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", RobustScaler())]), numeric_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols),
])
# 수치 대치는 이 특징 행렬에만 쓴다. 결측 표시 열을 함께 넣어 "값이 없었다"는 정보를 남긴다.
```

## 4. 검증

| 점검 | 코드 | 기대 |
|---|---|---|
| 정제 전후 건수·결측·평균 | `reports["comparison"]` | 결측 증가분 = `issues`의 변환 실패 수 |
| 타입·허용값·키 | `reports["issues"]` | 0건이 목표. 남은 건은 원천을 고친다. |
| 범위 밖 | `review_flags` | 사람이 원출처를 확인한 뒤 값을 고치거나 그대로 두고 메모를 남긴다. |
| 실측 이상치 | `trim_samples_iqr`가 돌려준 제거 수 | 모델별 제거 비율을 데이터 시트에 보고한다. |
| 분할 간 분포 | `distribution_shift(train, test, numeric, categorical)` | 시간 분할이라 차이가 나는 게 정상이다. 크기를 보고한다. |
| 분할 비율 | `split_ratios` | 60/15/25 ±5%p를 벗어나면 경계를 분기 단위로 조정한다. |

## 5. 결정 근거 요약

- **결측 = 대치 없음:** 데이터셋의 기여가 "출처와 검증 상태가 추적되는 사실값"이다. 대치값은 출처가 없다.
- **이상치 = 표시만 (스펙) / IQR 절단 (실측 표본):** 스펙의 극단값은 정보일 수 있다. 업로드 표본의 극단값은 노이즈일 가능성이 높다.
- **스케일러 = RobustScaler:** 성능·가격 분포가 오른쪽으로 긴 꼬리를 가진다.
- **분할 = 시간 + 계열:** 실제 쓰임이 "과거 데이터로 신제품을 판정"하는 것이다. 무작위로 나누면 형제 모델로 정답이 샌다.

## 6. 코드 리뷰 반영 (2026-10-08)

- **단종일:** 출처 값은 덮어쓰지 않고, 빈칸만 가정으로 채운다.
- **다중 모델 판정:** "RTX 4070/4080"처럼 둘째 모델을 번호만 적은 줄임 표기도 다중 모델로 제외한다. 기준은 3자리 이상 모델 번호가 2개 이상인 경우다.
- **분할:** 계열명의 대소문자·공백 차이를 무시하고, 자기 출시일이 없는 행은 계열을 따른다. 기준 날짜가 없어 분할을 못 받은 행은 `unassigned`로 보고한다.
- **결측 표시 열:** 모든 수치 열에 만든다. train에서 전부 결측인 열도 버리지 않는다.
- **dtype 처리:** 정제 결과(pd.NA)로 특징 변환기를 fit해도 터지지 않게 했다.
- **IQR 절단:** NaN과 키 없는 행은 따로 센다. 작은 그룹(표본 수 < low_n)이나 IQR=0 그룹은 자르지 않는다.
- **변환 오류 보고:** 1e30 같은 거대 정수는 실행을 멈추지 않고 변환 실패로 보고한다. 중복 인덱스가 있어도 위치 번호로 보고한다.
- **표기 보정:** CSV의 1.0/0.0은 bool로 읽고, enum·list 값의 대소문자는 표준 표기로 맞춘다.
- **모바일 제외:** HK·HX3D·P 접미사도 모바일로 제외한다.

## 7. 출처 관행 (2026-10-08 채택, 근거: `docs/notes/provenance-practices.md`)

| # | 관행 | 구현 위치 |
|---|---|---|
| 1 | "검증"을 둘로 나눠 쓴다. **출처 충실성**은 고정한 위키 리비전과 값이 일치함을 뜻한다. **현실 정확성**은 공식 사양과 일치함을 뜻하며, 출처 등급과 사람 대조로 나타낸다. | 데이터 시트 문구, `source_tier` |
| 2 | 리비전 ID와 수집 시각을 고정한다. 위키 원문 스냅샷은 CC BY-SA 동일조건 때문에 **고정 리비전 링크(oldid)**로 대신 배포한다. | `collect_wiki.py`의 `revid`·`fetched_at`·`url` |
| 3 | 각주 없는 값은 숨기지 않고 `WIKIPEDIA_UNCITED`로 표시한다(Wikidata P143 관례). | `normalize_wiki.py` 등급 |
| 4 | 출처 등급별로 최소 10건을 층화 추출해 사람이 대조하고, 등급별 정확도를 Wilson 95% 구간으로 보고한다. | `quality.verification_sample`(60건), `quality.accuracy_by_tier` |
| 5 | 두 출처의 일치는 한쪽이 **제조사 1차 자료**일 때만 검증으로 인정한다. GetDeploying은 행마다 NVIDIA 데이터시트를 걸어 두므로 1차 자료에 준한다고 명시하고, 2차 출처끼리의 일치는 "일치율"로만 보고한다. | `quality.cross_source_agreement` |
| 6 | 필드별 출처 등급 분포를 공개한다(선례를 찾지 못했으므로 논문 기여점). 데이터 시트에서 오류·잡음과 외부 자원 보존 질문에 답한다. | `data/out/quality_report.md` §4 |

## 8. CPU 캐시 (A 기본 + B 교차 확인 + C 사람 대조)

- **A (기본):** linuxhw DMI 보고서 5,821건에서 모델별 (L1, L2, L3) 최빈값을 낸다. 보고서가 3건 이상인 **92개 모델을 채택**했다. 최빈값 비율은 평균 0.99, 최소 0.86이다.
  - 위키에 값이 있는 필드는 위키를 1차 출처로 둔다. 관찰값은 위키가 빈 필드(L1·L2)만 `OPEN_DATASET`(CC BY 4.0)으로 채운다.
- **B (교차 확인):** 위키 L3와 관찰 L3가 **100% 일치**했다(채택 모델 전체).
- **C (사람 대조):** 채택하지 못한 **86개 모델**은 `data/manual/cache_todo.csv`로 보낸다. 2025-01 이후 출시 모델, 보고서가 3건 미만인 모델, T 모델 등이다. 사람이 공식 페이지를 보고 한 행씩 기입한다.
- **주의:** AMD L1은 데이터·명령 합계만 있다. 인텔 하이브리드는 P·E 세트의 L1·L2를 합하고, L3는 공유 캐시라 합하지 않는다.
