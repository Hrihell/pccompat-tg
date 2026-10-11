import pytest

from pccompat.rules import FAIL, PASS, UNKNOWN, chipset_ram, cpu_chipset, cpu_ram


@pytest.mark.parametrize(
    "cpu, chip, support, label, cause",
    [
        ("AM5", "AM4", "SUPPORTED", FAIL, None),          # 소켓 불일치 → 지원표 볼 필요 없음
        ("AM5", "AM5", "SUPPORTED", PASS, None),
        ("AM4", "AM4", "VARIES", UNKNOWN, "BOARD_DEPENDENT"),
        ("AM4", "AM4", "BIOS_REQUIRED", UNKNOWN, "BIOS_REQUIRED"),
        ("LGA1200", "LGA1200", "UNSUPPORTED", FAIL, None),
        ("LGA1700", "LGA1700", "NOT_LISTED", UNKNOWN, "MISSING"),
        ("", "AM5", "SUPPORTED", UNKNOWN, "MISSING"),
    ],
)
def test_cpu_chipset(cpu, chip, support, label, cause):
    v = cpu_chipset(cpu, chip, support)
    assert (v.label, v.cause) == (label, cause)


def test_cpu_ram():
    assert cpu_ram("DDR5", 192, "DDR4", 32).label == FAIL
    assert cpu_ram("DDR4;DDR5", 128, "DDR5", 192).label == FAIL          # 용량 초과
    assert cpu_ram("DDR4;DDR5", None, "DDR5", 32).cause == "MISSING"      # 최대 메모리 모름 → 추정 안 함
    assert cpu_ram("DDR5", 192, "DDR5", 96).label == PASS


def test_chipset_ram():
    assert chipset_ram("DDR4;DDR5", "DDR5").cause == "BOARD_DEPENDENT"
    assert chipset_ram("DDR5", "DDR4").label == FAIL
    assert chipset_ram("DDR5", "DDR5").label == PASS
    assert chipset_ram("", "DDR5").cause == "MISSING"


def test_gpu_link_warns_only_when_a_narrow_link_is_cut_further():
    from pccompat.rules import gpu_link
    # RX 6500 XT(PCIe 4.0 x4)를 PCIe 3.0 플랫폼에: 4.0x4(8) → 3.0x4(4), 3.0 x8 상당(8) 이하로 깎임 → 경고
    assert gpu_link(4, 4, 3, 16, 4) == "WARN"
    # RTX 4060(4.0 x8)을 PCIe 3.0 보드에: 4.0x8(16) → 3.0x8(8) → 경고
    assert gpu_link(4, 8, 4, 16, 3) == "WARN"
    # RTX 4090(4.0 x16)을 3.0 x16에: 32 → 16, 여전히 3.0 x8 상당보다 넓음 → 경고 안 함
    assert gpu_link(4, 16, 3, 16, 4) == "OK"
    # 깎이지 않으면 OK, CPU가 GPU 레인을 x4만 주면(예: 일부 APU) 경고
    assert gpu_link(4, 8, 5, 16, 5) == "OK"
    assert gpu_link(4, 8, 4, 4, 4) == "WARN"
    # 플랫폼 사양을 사람이 확인하지 않았으면 판단하지 않는다
    assert gpu_link(4, 8, None, 16, 4) == "UNCHECKED"

