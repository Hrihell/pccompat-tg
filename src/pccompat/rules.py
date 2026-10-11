"""쌍 호환 판정 규칙(설계서 §1·§4). 백엔드 규칙 엔진의 HC-001·003·004·005를 데이터셋 노드 단위로 옮긴 것.

엔진과 다른 점(의도된 차이, 데이터 시트에 명시)
- 엔진은 보드 제품(SKU) 단위, 데이터셋은 칩셋·RAM 규격 단위다.
  칩셋 하나에 DDR4·DDR5 보드가 공존하면 BOARD_DEPENDENT(확인 필요).
- 엔진 HC-003은 "BIOS 업데이트 필요"를 실패(경고)로 낸다. 데이터셋은 업데이트로 해소될 수 있으므로
  UNKNOWN/BIOS_REQUIRED로 둔다. 위키 칩셋 표의 "Varies/Partial"(보드마다 다름)은 UNKNOWN/BOARD_DEPENDENT.
- 엔진 HC-005는 보드 슬롯·최대 용량을 본다. 데이터셋은 보드가 없으므로 CPU 공식 최대 메모리로 판정한다.
값이 없으면 추정하지 않고 UNKNOWN/MISSING.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS, UNKNOWN, FAIL = "PASS", "UNKNOWN", "FAIL"
SEVERITY = {PASS: 0, UNKNOWN: 1, FAIL: 2}


@dataclass(frozen=True)
class Verdict:
    label: str
    cause: str | None = None
    rule_ids: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip())


def gens(v: Any) -> tuple[str, ...]:
    if _blank(v):
        return ()
    return tuple(v) if isinstance(v, (tuple, list)) else tuple(x.strip() for x in str(v).split(";") if x.strip())


def cpu_chipset(cpu_socket: Any, chip_socket: Any, support: Any) -> Verdict:
    ev = {"cpu.socket": cpu_socket, "chipset.socket": chip_socket, "support": support}
    if _blank(cpu_socket) or _blank(chip_socket):
        return Verdict(UNKNOWN, "MISSING", ("HC-001",), ev)
    if cpu_socket != chip_socket:
        return Verdict(FAIL, None, ("HC-001",), ev)
    if support == "SUPPORTED":
        return Verdict(PASS, None, ("HC-001", "HC-003"), ev)
    if support == "UNSUPPORTED":
        return Verdict(FAIL, None, ("HC-003",), ev)
    if support == "BIOS_REQUIRED":
        return Verdict(UNKNOWN, "BIOS_REQUIRED", ("HC-003",), ev)
    if support == "VARIES":
        return Verdict(UNKNOWN, "BOARD_DEPENDENT", ("HC-003",), ev)
    return Verdict(UNKNOWN, "MISSING", ("HC-003",), ev)


def cpu_ram(cpu_gens: Any, max_memory_gb: Any, ram_gen: str, ram_capacity_gb: int) -> Verdict:
    g = gens(cpu_gens)
    ev = {"cpu.memory_gens": list(g), "cpu.max_memory_gb": None if _blank(max_memory_gb) else max_memory_gb,
          "ram.ddr_gen": ram_gen, "ram.total_capacity_gb": ram_capacity_gb}
    if not g:
        return Verdict(UNKNOWN, "MISSING", ("HC-004",), ev)
    if ram_gen not in g:
        return Verdict(FAIL, None, ("HC-004",), ev)
    if _blank(max_memory_gb):
        return Verdict(UNKNOWN, "MISSING", ("HC-005",), ev)
    if ram_capacity_gb > float(max_memory_gb):
        return Verdict(FAIL, None, ("HC-005",), ev)
    return Verdict(PASS, None, ("HC-004", "HC-005"), ev)


def chipset_ram(options: Any, ram_gen: str) -> Verdict:
    g = gens(options)
    ev = {"chipset.memory_gen_options": list(g), "ram.ddr_gen": ram_gen}
    if not g:
        return Verdict(UNKNOWN, "MISSING", ("HC-004",), ev)
    if ram_gen not in g:
        return Verdict(FAIL, None, ("HC-004",), ev)
    if len(g) > 1:
        return Verdict(UNKNOWN, "BOARD_DEPENDENT", ("HC-004",), ev)
    return Verdict(PASS, None, ("HC-004",), ev)


# ---------- GPU 연결 경고(G2) ----------
# 3값 라벨과 별개다. 꽂히고 동작하지만 성능이 떨어질 수 있는 조합을 표시한다.
# 판단 입력(CPU·칩셋의 PCIe 사양)은 사람이 공식 자료로 확인한 값만 쓰고, 없으면 UNCHECKED로 둔다.
OK, WARN, UNCHECKED = "OK", "WARN", "UNCHECKED"
# 경고 경계: 실효 대역폭이 GPU 자체보다 줄었고 PCIe 3.0 x8(= 4.0 x4) 상당 이하일 때.
# x16 카드가 한 세대 낮은 슬롯에 꽂히는 경우(예: 4.0 x16 → 3.0 x16)는 손실이 작아 경고하지 않는다.
PCIE_WARN_MAX_UNITS = 8


def _pcie_units(gen: int, lanes: int) -> int:
    return lanes * 2 ** (gen - 3)  # PCIe 3.0 한 레인 = 1


def gpu_link(gpu_gen: Any, gpu_lanes: Any, cpu_gen: Any, cpu_lanes: Any, slot_gen: Any) -> str:
    """G2: GPU 슬롯의 실효 PCIe 대역폭(세대는 GPU·CPU·보드 슬롯 중 최소, 레인은 GPU·CPU 중 최소)."""
    if any(_blank(v) for v in (gpu_gen, gpu_lanes, cpu_gen, cpu_lanes, slot_gen)):
        return UNCHECKED
    gg, gl = int(gpu_gen), int(gpu_lanes)
    eff = _pcie_units(min(gg, int(cpu_gen), int(slot_gen)), min(gl, int(cpu_lanes)))
    return WARN if eff < _pcie_units(gg, gl) and eff <= PCIE_WARN_MAX_UNITS else OK

