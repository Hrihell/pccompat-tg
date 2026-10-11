# 2차 이용 점검표: 논문과 공개 데이터셋 (2026-10-08, 법률 자문 아님)

- 우리 산출물: 데이터는 **CC BY 4.0**, 코드는 **MIT**.
- 원천 파일과 해시: `data/raw/files_manifest.csv`
- 근거 조사:
  - `_workspace/research/benchmark-licenses.md`
  - `_workspace/research/secondary-use-law.md`
  - `docs/sources.md`

## 1. 원천별 의무와 준수 방법

| 원천 | 쓰는 것 | 라이선스·조건 | 우리 의무 | 준수 방법 | 상태 |
|---|---|---|---|---|---|
| 위키백과 | 표의 **사실값**만(문구·표 형식 X) | 본문은 CC BY-SA 4.0. 기여자 DB권 포기("facts … may be reused freely without attribution") | 사실만 쓰면 의무는 없음. 단, 문구·표를 복사하면 동일조건(SA)이 붙음 | 원문 HTML은 내부 작업용으로만 두고 비공개. 공개본에는 값 + 고정 리비전 링크(oldid)만 | ✅ |
| Blender Open Data | 실측 표본 → 모델별 중앙값 | **CC0 1.0**(zip 안 LICENSE.txt 확인) | 없음 | 관례상 인용: "Blender Open Data, 2026-10-07 snapshot, CC0" | ✅ |
| Kishonti GFXBench·CompuBench | 장치별 중앙값 | LICENSE §2 "non-code assets … CC BY 4.0" | 저작자 표시, 라이선스 링크, **변경 사항 표시** | THIRD_PARTY_NOTICES에 표기. "우리 모델 ID로 매칭·집계함"을 명시 | ✅ 2026-10-08 사용자 보고: Kishonti에 'CC BY 4.0 scope of GFXBench/CompuBench results CSVs — academic dataset' 문의 → **허락 받음**. 회신 원문은 `docs/notes/kishonti-permission.md`에 보관(사용자가 붙여 넣을 것). 논문·데이터 시트에는 'LICENSE §2(CC BY 4.0) + 저작권자 확인 회신' 명시 |
| GetDeploying | NVIDIA 20종 스펙(교차 확인) + 데이터시트 URL | "CC BY 4.0: Credit GetDeploying and link to the license … covers the published dataset files only and implies no endorsement" | 저작자 표시, 라이선스 링크, 변경 표시, **보증 암시 금지** | 공개된 CSV 파일만 사용하고 사이트·API는 쓰지 않음(robots도 /api/ 금지). 표기 문구 고정 | ✅ |
| linuxhw DMI·LsPCI | 모델별 캐시 최빈값, (2일차) 관찰 조합 집계 | CC BY 4.0(저장소 LICENSE) | 저작자 표시, 링크, 변경 표시 | **원 보고서는 공개하지 않고 집계만**(개인 PC 덤프, 익명화 불완전 가능성 — hw-probe README가 직접 인정). 커밋 해시 고정 | ✅ |
| pci.ids | 장치 ID → 제품명(Kishonti 장치명 해석) | GPL-2.0+ 또는 BSD-3 중 **BSD-3 선택**. 헤더: 저작권은 "aggregation and formatting"에만 미침 | BSD-3: 저작권 고지·조건·면책 문구 유지. 저작권자 이름으로 보증 표시 금지 | 파일 자체는 재배포하지 않음. 쓴 값의 출처로 고지문 + BSD-3 본문을 THIRD_PARTY_NOTICES에 수록 | ✅ |
| amdgpu.ids (libdrm) | (계획) AMD 장치+리비전 → SKU | MIT(프로젝트 표기, 파일 헤더 없음) | MIT 고지 | **gitlab.freedesktop.org의 robots.txt가 Claude 계열 에이전트를 전면 금지** → 에이전트로 받은 파일은 삭제했다. 필요하면 사람이 브라우저로 받는다 | ⛔ 보류 |
| 제조사 공식 페이지(AMD·인텔·NVIDIA) | 사람이 한 행씩 확인한 값(출시일·캐시·검증 표본) | AMD: 개인 용도. 인텔 4.1.14/16/17: 자동·체계적 수집 금지 | 자동 수집 금지, 대량 복제 금지 | 사람이 몇 건씩 확인해 **인용**하고 URL을 기록. 대량 원천으로 쓰지 않음 | ✅(운영 규칙) |
| 보드 제조사·JEDEC | (계획) 몇 건의 공식 BIOS 사실·표준 상수 | All rights reserved, JEDEC 재배포 금지. jedec.org는 AI 에이전트 차단 | 원문·PDF 재배포 금지 | 사람이 읽고 사실 몇 건만 인용 | ✅(운영 규칙) |

## 2. 논문 본문에서 쓰는 방식

| 사용 | 판단 | 조치 |
|---|---|---|
| 표·그림에 원천 값을 가공해 싣기(예: 세대별 L3 분포, Blender 중앙값 상관 그래프) | CC BY·CC0·사실값이라 가능 | 캡션이나 각주에 출처를 적는다. CC BY 원천은 "adapted from …, CC BY 4.0" |
| 원천의 표·그래프를 그대로 캡처 | 원천마다 다름(위키 표 = BY-SA) | **하지 않는다.** 그림은 우리 데이터로 새로 그린다 |
| 라이선스·약관 문구 인용 | 짧은 인용은 저작권법 제28조 범위 | 출처 URL과 함께 인용 |
| 제품명·상표(GeForce, Ryzen, Core 등) | 제품을 가리키는 지명적 사용 | 데이터 시트에 상표 고지 1줄: "All product names are trademarks of their respective owners; no endorsement implied." |
| 사용자 리뷰·닉네임 | 수집하지 않음 | — |

## 3. 우리 CC BY 4.0과의 호환

- CC0, BSD-3, MIT, CC BY 4.0 입력은 모두 CC BY 4.0 산출물에 넣을 수 있다. 조건은 각 고지를 유지하는 것이다.
- 쓰지 않는 것:
  - CC BY-SA(위키 문구)
  - CC BY-NC(SPEC 등)
  - TechPowerUp 사이트·DB에서 **직접** 수집한 데이터(약관상 재배포 불가). 접속·조회해 옮겨 적는 것도 포함한다.
    - 위키 표의 사실값 중 각주가 TechPowerUp을 가리키는 값은 쓰되, 상류 출처로 공개한다(데이터 시트 출처 절, `field_provenance.csv`의 `source_url`).
    - 근거(2026-10-10 결정): 개별 사실값은 저작물이 아니고, 저작권법 제93조 ②은 DB의 개별 소재를 "상당한 부분"으로 보지 않으며 ④은 보호가 소재 자체에 미치지 않는다고 정한다. 우리는 TechPowerUp을 반복·체계적으로 수집하지 않았다(대법원 2021도1533: 반복·체계 복제는 결과적으로 상당한 부분 복제와 같을 때만 침해).
    - 계보를 숨기면 Longpre 등(2024)이 지적한 license laundering이 된다. 그래서 값마다 각주 URL을 배포한다.
    - 권리자가 요청하면 해당 값을 지운다(README·데이터 시트에 안내).
- CC BY 4.0 §2(a)(5)는 하류 이용자에게 기술적 제한을 추가하지 못하게 한다. 그래서 공개본에는 접근 제한을 두지 않는다.
- 영리 재사용 주의: CC BY는 하류 이용자의 영리 재사용도 허용한다.
  - 저작권법 제94조 연구 예외는 비영리 이용에만 적용되므로, 하류 이용자의 영리 재사용까지 덮어 주지 않는다.
  - 그래서 공개본에는 **사실값과 우리가 계산한 파생값만** 넣는다.
  - 원천의 표현물(문구·원 덤프)은 넣지 않는다.

## 4. 남은 조치

1. ~~Kishonti에 라이선스 확인 메일 보내기~~ 완료(2026-10-08 허락). 회신 원문을 `docs/notes/kishonti-permission.md`에 붙여 넣기
2. amdgpu.ids가 필요하면 사람이 브라우저로 받기. 지금은 없어도 된다(Kishonti의 AMD PCI ID 장치는 확정 매칭에서 제외됨).
3. 공개 전에 한국저작권위원회 무료 상담이나 학교 법무팀의 확인을 권장한다.
