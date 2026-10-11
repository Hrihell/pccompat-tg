from pathlib import Path

from pccompat.load_open_files import parse_pci_ids, resolve_pci_name

PCI = """# header
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t73bf  Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]
\t7480  Navi 33 [Radeon RX 7600/7600 XT/7600M XT/7700S / Mobile]
\t744c  Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]
\t\t1002 0e3b  subsystem line
\t1586  Single Product [Radeon RX 9999 XT]
10de  NVIDIA Corporation
\t2786  AD104 [GeForce RTX 4070]
C 00  Unclassified device
"""


def test_parse_and_resolve(tmp_path: Path):
    p = tmp_path / "pci.ids"
    p.write_text(PCI)
    pci = parse_pci_ids(p)
    assert pci[("10de", "2786")] == "AD104 [GeForce RTX 4070]"
    assert resolve_pci_name("AMD 744C:C8", pci) is None            # 여러 제품 묶음 → 확정 안 함
    assert resolve_pci_name("AMD 1586:00", pci) == "Radeon RX 9999 XT"
    assert resolve_pci_name("AMD FFFF:00", pci) is None            # 모르는 ID
    assert resolve_pci_name("NVIDIA GeForce RTX 4070", pci) == "NVIDIA GeForce RTX 4070"  # PCI ID 형식 아님 → 그대로
