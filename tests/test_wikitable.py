from pccompat.wikitable import parse_tables, reference_urls

HTML = """
<div>
<table class="wikitable">
<tr><th rowspan="2">Model</th><th colspan="2">Clock rate (GHz)</th><th rowspan="2">MSRP</th></tr>
<tr><th>Base</th><th>Boost</th></tr>
<tr><td>7950X<sup class="reference"><a href="#cite_note-a-1">[1]</a></sup></td><td rowspan="2">4.5</td><td>5.7</td><td>US $699<sup class="reference"><a href="#cite_note-p-2">[2]</a></sup></td></tr>
<tr><td>7900X</td><td>5.6[iii]</td><td>US $549</td></tr>
</table>
<ol class="references">
<li id="cite_note-a-1"><a class="external text" href="https://www.amd.com/en/products/7950x">AMD</a></li>
<li id="cite_note-p-2"><a class="external text" href="https://ir.amd.com/news">PR</a><a href="/wiki/X">wiki</a></li>
</ol>
</div>
"""


def test_expands_spans_and_joins_headers():
    (t,) = parse_tables(HTML)
    assert t.headers == ["Model", "Clock rate (GHz) / Base", "Clock rate (GHz) / Boost", "MSRP"]
    assert [c.text for c in t.rows[0]] == ["7950X", "4.5", "5.7", "US $699"]
    assert [c.text for c in t.rows[1]] == ["7900X", "4.5", "5.6", "US $549"]  # rowspan 4.5 복제, 각주 표시 제거


def test_cell_citations_resolve_to_external_urls():
    (t,) = parse_tables(HTML)
    refs = reference_urls(HTML)
    assert t.rows[0][0].cite_ids == ("cite_note-a-1",)
    assert refs["cite_note-a-1"] == ["https://www.amd.com/en/products/7950x"]
    assert refs["cite_note-p-2"] == ["https://ir.amd.com/news"]
