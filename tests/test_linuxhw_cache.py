from pccompat.load_linuxhw_cache import parse_report

AMD = """Handle 0x0004, DMI type 4, 48 bytes
Processor Information
\tVersion: AMD Ryzen 7 7800X3D 8-Core Processor

Handle 0x0010, DMI type 7, 27 bytes
Cache Information
\tSocket Designation: L1 - Cache
\tConfiguration: Enabled, Not Socketed, Level 1
\tInstalled Size: 512 kB
\tSystem Type: Unified

Handle 0x0011, DMI type 7, 27 bytes
Cache Information
\tSocket Designation: L2 - Cache
\tConfiguration: Enabled, Not Socketed, Level 2
\tInstalled Size: 8 MB

Handle 0x0012, DMI type 7, 27 bytes
Cache Information
\tSocket Designation: L3 - Cache
\tConfiguration: Enabled, Not Socketed, Level 3
\tInstalled Size: 96 MB
"""


def _block(level, size, typ=""):
    return f"Handle 0x1, DMI type 7\nCache Information\n\tConfiguration: Enabled, Level {level}\n\tInstalled Size: {size}\n\tSystem Type: {typ}\n"


def test_amd_unified_report():
    version, cache = parse_report(AMD)
    assert version.startswith("AMD Ryzen 7 7800X3D")
    assert cache == (512.0, 8 * 1024.0, 96 * 1024.0)


def test_hybrid_intel_sums_l1_l2_but_not_l3():
    text = "Handle 0x0\nProcessor Information\n\tVersion: 13th Gen Intel(R) Core(TM) i7-13700K\n\n" + "\n".join([
        _block(1, "384 kB", "Data"), _block(1, "256 kB", "Instruction"), _block(2, "16 MB"), _block(3, "30 MB"),
        _block(1, "256 kB", "Data"), _block(1, "512 kB", "Instruction"), _block(2, "8 MB"), _block(3, "30 MB"),
    ])
    _, cache = parse_report(text)
    assert cache == (1408.0, 24 * 1024.0, 30 * 1024.0)


def test_zero_or_missing_cache_is_dropped():
    text = "Handle 0x0\nProcessor Information\n\tVersion: X\n\n" + "\n".join(
        [_block(1, "0 kB"), _block(2, "0 kB"), _block(3, "0 kB")])
    assert parse_report(text)[1] is None
