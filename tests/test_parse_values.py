"""셀 파서 테스트: 입력은 2026-10-08 수집본에서 실제로 본 셀 형식."""

from datetime import date

import pytest

from pccompat import parse_values as pv


@pytest.mark.parametrize("s, want", [("16 (32)", (16, 32)), ("8 (8)", (8, 8)), ("— N/a", (None, None))])
def test_cores_threads(s, want):
    assert pv.cores_threads(s) == want


@pytest.mark.parametrize("s, want", [("5.7", 5700), ("4.6", 4600), ("— N/a", None)])
def test_ghz(s, want):
    assert pv.ghz_to_mhz(s) == want


@pytest.mark.parametrize("s, want", [("1830 (2460)", 2460), ("1000 2040", 2040), ("2450", 2450), ("", None)])
def test_boost(s, want):
    assert pv.boost_mhz(s) == want


@pytest.mark.parametrize("s, want", [("US $699", 699), ("$110", 110), ("— N/a", None), ("Jan 4, 2022 OEM", None)])
def test_usd(s, want):
    assert pv.usd(s) == want


@pytest.mark.parametrize(
    "s, want",
    [("2304:72:32: 18:72:", 2304), ("768:48:32:12 12 CU", 768), ("6 Xe cores 768:32:", 768), ("12 CUs 768:48:24:12", 768)],
)
def test_shaders(s, want):
    assert pv.shader_count(s) == want


@pytest.mark.parametrize(
    "s, want",
    [
        ("Feb 28, 2023", (date(2023, 2, 28), "DAY")),
        ("April 22, 2026", (date(2026, 4, 22), "DAY")),
        ("Apr 16, 2024 [ citation needed ]", (date(2024, 4, 16), "DAY")),
        ("April 2024", (date(2024, 4, 1), "MONTH")),
        ("Aug, 2025", (date(2025, 8, 1), "MONTH")),
        ("Q4 2021", (date(2021, 10, 1), "QUARTER")),
        # OEM 선출시 뒤 리테일 출시: 리테일 데스크톱 범위라 리테일 날짜를 쓴다(5600G·5700G)
        ("Apr 13, 2021 (OEM), Aug 5, 2021 (retail)", (date(2021, 8, 5), "DAY")),
        ("", (None, None)),
    ],
)
def test_release(s, want):
    assert pv.release(s) == want


def test_watts_and_slug():
    assert pv.watts("120 W") == 120
    assert pv.slug("Core i9-12900KS") == "core-i9-12900ks"


@pytest.mark.parametrize(
    "s,want",
    [("PCIe 4.0 ×16", (4, 16)), ("PCIe 4 x8", (4, 8)), ("PCIe 5 x16", (5, 16)), ("PCIe 4.0 ×4", (4, 4)),
     ("", (None, None)), ("N/A", (None, None))],
)
def test_pcie_bus_interface(s, want):
    assert pv.pcie(s) == want
