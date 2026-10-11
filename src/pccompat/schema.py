"""8개 테이블 스키마(설계서 §1). 검증기·정제기가 같은 정의를 쓴다(단일 원천)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["str", "int", "float", "date", "bool", "enum", "list", "json"]


@dataclass(frozen=True)
class Column:
    name: str
    kind: Kind
    required: bool = False          # 레코드 완결성(COMPLETE) 판단에 쓰는 필수 필드
    allowed: tuple[str, ...] = ()   # enum·list 원소 허용값
    nullable: bool = True           # False면 결측 자체가 형식 오류(키 등)


@dataclass(frozen=True)
class Table:
    name: str
    key: tuple[str, ...]
    columns: tuple[Column, ...]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name}에 {name} 컬럼 없음")

    def of_kind(self, *kinds: Kind) -> list[str]:
        return [c.name for c in self.columns if c.kind in kinds]


VENDORS = ("AMD", "INTEL", "NVIDIA")
MEM_GENS = ("DDR4", "DDR5")
LABELS = ("PASS", "FAIL", "UNKNOWN")
UNKNOWN_CAUSES = ("BIOS_REQUIRED", "BOARD_DEPENDENT", "MISSING", "UNVERIFIED")
SPLITS = ("train", "val", "test")
DATE_TYPES = ("LAUNCH", "ANNOUNCE", "UNKNOWN")
EVIDENCE = ("DOCUMENTED", "OBSERVED", "BOTH")  # 스펙·지원 문서 기반 규칙 판정 / 실제 PC 관찰(linuxhw) / 둘 다
# 출처 등급(정밀도 순): 공식 > 값 셀 각주 > 행 각주 > 열 각주 > 각주 없음(리비전만) / 공개 데이터셋 / 기타
SOURCE_TIERS = ("OFFICIAL", "WIKIPEDIA_CITED_CELL", "WIKIPEDIA_CITED_ROW", "WIKIPEDIA_CITED_COLUMN",
                "WIKIPEDIA_FAMILY", "WIKIPEDIA_UNCITED", "OPEN_DATASET", "OTHER")  # FAMILY = 세대 첫 출시일(문서 단위)

PARTS = Table(
    "parts",
    key=("part_id",),
    columns=(
        Column("part_id", "str", required=True, nullable=False),
        Column("category", "enum", required=True, allowed=("CPU", "GPU"), nullable=False),
        Column("vendor", "enum", required=True, allowed=VENDORS, nullable=False),
        Column("family", "str", required=True, nullable=False),
        Column("model_name", "str", required=True, nullable=False),
        Column("socket", "str"),
        Column("memory_gens", "list", allowed=MEM_GENS),
        Column("max_memory_gb", "int"),
        Column("cores", "int"),
        Column("threads", "int"),
        Column("boost_clock_mhz", "int"),
        Column("tdp_w", "int"),
        Column("l1_cache_kb", "float"),   # 칩 전체 합(표에 있을 때만, 없으면 결측 — 추정 안 함)
        Column("l2_cache_mb", "float"),
        Column("l3_cache_mb", "float"),
        Column("shader_count", "int"),
        Column("fp32_tflops", "float"),
        # GPU 버스 인터페이스(G2 PCIe 대역폭 경고용). 위키 GPU 표의 Bus interface 열.
        Column("pcie_gen", "int"),
        Column("pcie_lanes", "int"),
        Column("proxy_perf", "float"),
        Column("release_date", "date", required=True),
        Column("release_date_type", "enum", allowed=DATE_TYPES),  # 출처마다 발표일·판매일이 섞임(RTX 4090 예)
        Column("eol_date", "date"),
        Column("eol_is_assumed", "bool"),
        Column("msrp_usd", "int"),
        Column("record_status", "enum", allowed=("COMPLETE", "PARTIAL")),
    ),
)

# 카테고리별 추가 필수 필드: 판정·분석에 꼭 필요한 값(없으면 PARTIAL, 판정은 UNKNOWN/MISSING)
REQUIRED_BY_CATEGORY = {
    "CPU": ("socket", "memory_gens", "max_memory_gb", "cores", "boost_clock_mhz"),
    "GPU": ("shader_count", "boost_clock_mhz"),
}

CHIPSETS = Table(
    "chipsets",
    key=("chipset_id",),
    columns=(
        Column("chipset_id", "str", required=True, nullable=False),
        Column("vendor", "enum", required=True, allowed=("AMD", "INTEL"), nullable=False),
        Column("name", "str", required=True, nullable=False),
        Column("socket", "str", required=True),
        Column("memory_gen_options", "list", required=True, allowed=MEM_GENS),
        Column("release_date", "date", required=True),
        Column("eol_date", "date"),
        Column("eol_is_assumed", "bool"),
        Column("record_status", "enum", allowed=("COMPLETE", "PARTIAL")),
        # 보드 메모리 분할(v2): DDR4·DDR5 겸용 칩셋은 보드 메모리 종류별 노드로 나뉜다. 나뉘지 않은 칩셋은 board_memory가 비고
        # base_chipset_id·base_name이 자기 자신이다.
        Column("base_chipset_id", "str"),
        Column("base_name", "str"),
        Column("board_memory", "enum", allowed=MEM_GENS),
    ),
)

RAM_SPECS = Table(
    "ram_specs",
    key=("ram_id",),
    columns=(
        Column("ram_id", "str", required=True, nullable=False),
        Column("ddr_gen", "enum", required=True, allowed=MEM_GENS, nullable=False),
        Column("total_capacity_gb", "int", required=True, nullable=False),
        Column("release_date", "date"),
    ),
)

CPU_CHIPSET_SUPPORT = Table(
    "cpu_chipset_support",
    key=("cpu_id", "chipset_id"),
    columns=(
        Column("cpu_id", "str", required=True, nullable=False),
        Column("chipset_id", "str", required=True, nullable=False),
        Column("support", "enum", required=True,
               allowed=("SUPPORTED", "BIOS_REQUIRED", "VARIES", "UNSUPPORTED", "NOT_LISTED"), nullable=False),
        Column("source_url", "str", required=True, nullable=False),
        Column("verified", "bool", required=True),
        Column("note", "str"),
    ),
)

PAIR_LABELS = Table(
    "pair_labels",
    key=("edge_id",),
    columns=(
        Column("edge_id", "str", required=True, nullable=False),
        Column("edge_type", "enum", required=True, allowed=("CPU_CHIPSET", "CPU_RAM", "CHIPSET_RAM"), nullable=False),
        Column("src", "str", required=True, nullable=False),
        Column("dst", "str", required=True, nullable=False),
        Column("label", "enum", required=True, allowed=LABELS, nullable=False),
        Column("unknown_cause", "enum", allowed=UNKNOWN_CAUSES),
        Column("evidence_type", "enum", required=True, allowed=EVIDENCE),
        Column("n_observed", "int"),          # 이 쌍이 관찰된 linuxhw 보고서 수(집계, 원 보고서 비공개)
        Column("basis_verified", "bool"),     # 판정에 쓴 값이 모두 사람 검증(verified)됐는가(라벨은 바꾸지 않고 표시만)
        Column("rule_ids", "list"),
        Column("evidence", "json"),
        Column("valid_from", "date"),
        Column("valid_to", "date"),
        Column("split", "enum", allowed=SPLITS),
        Column("ruleset_version", "str", required=True),
    ),
)

BUILD_LABELS = Table(
    "build_labels",
    key=("build_id",),
    columns=(
        Column("build_id", "str", required=True, nullable=False),
        Column("cpu_id", "str", required=True, nullable=False),
        Column("chipset_id", "str", required=True, nullable=False),
        Column("ram_id", "str", required=True, nullable=False),
        Column("gpu_id", "str", required=True, nullable=False),
        Column("label", "enum", required=True, allowed=LABELS, nullable=False),
        Column("unknown_causes", "list", allowed=UNKNOWN_CAUSES),
        Column("failed_rule_ids", "list"),
        Column("valid_from", "date"),
        Column("valid_to", "date"),
        Column("total_msrp_usd", "int"),
        Column("proxy_perf_cpu", "float"),
        Column("proxy_perf_gpu", "float"),
        Column("difficulty", "enum", allowed=("EASY", "MEDIUM", "HARD")),
        Column("split", "enum", allowed=SPLITS),
    ),
)

PERF_MEASUREMENTS = Table(
    "perf_measurements",
    key=("part_id", "source", "benchmark_version", "device_type", "scene"),
    columns=(
        Column("part_id", "str", required=True, nullable=False),
        Column("source", "enum", required=True,
               allowed=("BLENDER_OPEN_DATA", "GFXBENCH", "COMPUBENCH"), nullable=False),
        Column("benchmark_version", "str", required=True, nullable=False),
        Column("device_type", "str", required=True, nullable=False),
        Column("scene", "str", required=True, nullable=False),
        Column("median_samples_per_min", "float", required=True),
        Column("n_samples", "int", required=True),
        Column("n_removed_outliers", "int"),
        Column("iqr", "float"),
        Column("low_n_flag", "bool"),
        Column("matched_device_names", "list"),
        Column("match_method", "enum", allowed=("EXACT", "ALIAS", "FUZZY_REVIEWED")),
    ),
)

FIELD_PROVENANCE = Table(
    "field_provenance",
    key=("entity_id", "field"),
    columns=(
        Column("entity_id", "str", required=True, nullable=False),
        Column("field", "str", required=True, nullable=False),
        Column("value", "str", required=True),
        Column("source_url", "str", required=True, nullable=False),
        Column("archive_url", "str"),
        Column("source_tier", "enum", required=True,
               allowed=SOURCE_TIERS, nullable=False),
        Column("source_revision", "str"),
        Column("license_scope", "enum", required=True,
               allowed=("CC0", "CC_BY_4_0", "BSD_3", "MIT", "APACHE_2_0", "FACT_CITED"), nullable=False),
        Column("verified", "bool", required=True),
        Column("verified_by", "str"),
        Column("verified_at", "date"),
    ),
)

# linuxhw 관찰 조합의 공개용 집계(원 덤프는 개인정보 때문에 공개하지 않는다)
OBSERVED_COMBOS = Table(
    "observed_combos",
    key=("board_model", "cpu_id", "ram_gen", "bios_version"),
    columns=(
        Column("board_model", "str", required=True, nullable=False),
        Column("chipset_id", "str"),
        Column("cpu_id", "str", required=True, nullable=False),
        Column("ram_gen", "enum", allowed=MEM_GENS),
        Column("bios_version", "str", required=True, nullable=False),
        Column("bios_date", "date"),
        Column("gpu_id", "str"),
        Column("n_reports", "int", required=True, nullable=False),
        Column("source_commit", "str", required=True, nullable=False),
    ),
)

TABLES: dict[str, Table] = {
    t.name: t
    for t in (OBSERVED_COMBOS, PARTS, CHIPSETS, RAM_SPECS, CPU_CHIPSET_SUPPORT, PAIR_LABELS, BUILD_LABELS, PERF_MEASUREMENTS, FIELD_PROVENANCE)
}

LIST_SEPARATOR = ";"  # CSV 안의 list 필드 구분자
