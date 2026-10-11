# 원천별 실제 형식 (표본 확인: 2026-10-08)

로더는 이 문서에 적은 형식을 기준으로 작성한다. 형식을 기억에 의존해 쓰지 않는다.

| 원천 | 받는 방법 | 확인한 형식 | 메모 |
|---|---|---|---|
| **위키백과** | MediaWiki API `action=parse&page=Template:…&prop=text\|revid` (User-Agent에 연락처 포함, 요청 사이 1초) | 스펙 표는 목록 문서가 아니라 **세대별 템플릿**(예: `Template:AMD Ryzen 7000 Series`, revid 1374758833)에 있다. `pandas.read_html`로 표를 읽으면 2단 헤더가 나온다(`Branding and model`, `Cores (threads)`, `Clock rate (GHz) Base/Boost`, `TDP`, `Release date`, `MSRP` 등). | 셀 값에 각주 표시(`[iii]`)가 섞이고, MSRP는 `US $699` 형식이라 정리가 필요하다. 각주 원출처 URL은 HTML의 `sup.reference → li#cite_note-…`에서 따로 추출한다. `List of Intel Core processors`는 리다이렉트(47바이트)라서 인텔은 세대별 템플릿 이름을 따로 조사해야 한다. |
| **GetDeploying** | `https://getdeploying.com/dataset/gpu-models/gpu-models.csv` (robots 허용) | 113행 36열: `gpu_slug,name,vendor,release_date,…,fp32_dense_tflops,…,shader_cores,…,tdp_w,…,market_class,…,datasheet_url` | 범위 안 NVIDIA 소비자용만 쓴다(`market_class`로 거른다). `datasheet_url`이 공식 출처가 된다. |
| **Kishonti** | GitHub raw: `GFXBench-5.0-results.csv`(51MB), `CompuBench-2.0-Desktop-results.csv`(4MB). 마지막 푸시 2025-11-28 | `test id,test name,device id,device name,best/median/average result (raw/formatted/fps),test unit,screen size,os,api,cpu architecture,hardware type,hardware name,soc name,form factor` | **원표본이 아니라 집계값**(최근 100회의 최대·중앙·평균)이라 IQR 절단을 할 수 없다. 그래서 `median result (raw)`를 그대로 쓴다. 장치명이 `AMD 66A0:00` 같은 **PCI ID**인 경우가 있어 pci.ids·amdgpu.ids로 풀어야 한다. `form factor=desktop`과 `hardware type=dGPU`로 거른다. LICENSE 1절은 코드에 BSD-3을, 2절은 비코드 자산에 CC BY 4.0을 적용한다. |
| **Blender Open Data** | 사용자가 브라우저로 받은 `opendata-latest.zip`(2026-10-07 스냅샷, sha256 f1005528…a2c7)을 `data/raw/blender/`에 둔다 | zip 안에 `LICENSE.txt`(**CC0 1.0**)·README·jsonl(1.9GB, 제출 432,224건)이 있다. 스키마 v4 344,376건만 `stats.samples_per_minute`를 가진다(v3·v1은 렌더 시간뿐이라 제외). 장면은 monster·junkshop·classroom, Blender 3.4~5.2. | 에이전트·스크립트는 `/snapshots/`를 받지 않는다(robots). 한 제출에 장치가 둘 이상이면 어느 장치 점수인지 몰라 제외한다. 점수는 (모델, 장치 종류, 장면, Blender 버전)별로만 묶고 섞지 않는다. |
| **linuxhw DMI** | `git clone --depth 1` (커밋 57b21bc6, 2025-01-06) | **3.4GB**(리포트에 적힌 340MB가 아니다). 보고서 경로는 `{종류}/{제조사}/{계열}/{모델}/{HWID}` | macOS는 대소문자를 구분하지 않아 경로가 겹치는 보고서 약 40쌍 중 한쪽이 작업 트리에서 빠진다. 전수 집계가 필요하면 `git ls-tree`·`git show`로 읽는다. |
| **linuxhw LsPCI** | `git clone --depth 1` (커밋 716d5be960f, 2025-01-07) | 크기 16G | DMI와 같은 HWID 경로 구조 |
| **pci.ids / amdgpu.ids / Mesa / LLVM** | 파일 1개씩 받는다 | (다음 단계에서 표본 확인) | pci.ids는 상류 파일을 BSD-3 조건으로 쓴다. |

## CPU 캐시(L1·L2·L3) 출처 조사 (2026-10-08, 법률 자문 아님)

| 원천 | 내용 | 판단 |
|---|---|---|
| AMD 제품 페이지 | L1(합계, 데이터·명령 구분 없음)·L2·L3 표시(7800X3D: 512KB/8MB/96MB). robots는 제품 페이지 허용 | 약관: "only for your personal use … may not reproduce … for any … public purpose". 대량 출처로 쓰지 말고, 사람 확인·인용용 |
| 인텔 ARK | Smart Cache(L3)·Total L2만, **L1 없음** | 약관 4.1.14/16/17: 자동·체계적 수집 금지(문구는 요약 도구 경유, 원문 재확인 필요) |
| **linuxhw DMI**(CC BY 4.0) | dmidecode Cache Information. 7800X3D·13700K·12600K 표본이 공식값과 일치. 하이브리드 인텔은 P·E 두 세트(L1·L2 합산, L3는 공유라 합산 금지) | 모델별 최빈값(n≥3) 사용 가능. 펌웨어 오류(X3D L3 192MB 등)·코어 비활성 보고 섞임 → 최빈값으로 걸러짐. 2025-01 이후 모델 없음 |
| linuxhw LsCPU | 2022-03 이후 갱신 없음, 코어당·합계 혼재 | 사용 안 함 |
| Wikidata | 캐시 속성 없음 | 사용 안 함 |
| 인텔 최적화 매뉴얼 | 코어당 캐시 표 일부(Golden Cove 등), "unmodified copy" 외 이용 허락 없음 | 사실 인용만 |

조사 스크립트·출력: `docs/notes/cache_probe_linuxhw.py`, `docs/notes/cache_probe_out.txt`

## 검증 표본용 공식 페이지 개별 조회 (2026-10-08)
- 대상: `verification_sample.csv`의 공식 값 후보 찾기. 사람이 링크를 열어 확인한 행만 `match`를 채운다.
- 사이트: www.amd.com 제품 페이지, ir.amd.com 보도자료, www.nvidia.com 제품 페이지, nvidianews.nvidia.com 보도자료. 네 곳 robots.txt가 해당 경로를 허용함을 확인.
- 방법: 사이트맵(`en.sitemap.xml`, `ir.amd.com/sitemap.xml`)으로 주소를 찾고, 행마다 한 번씩 2초 간격으로 요청. 일반 UA, 연락처 정보 없음. 대량 수집이 아니다.
- 11600 계열 출시일(4개): 위키 `List of Intel Core desktop processors`(oldid 1371607091) Rocket Lake-S 표의 Release date. 공식 API로 한 번 조회. 각주가 없어 `WIKIPEDIA_UNCITED` 등급(2026-10-08). 인텔 사이트는 스크립트 접근이 403으로 막혀 조회하지 않음.
