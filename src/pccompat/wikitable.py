"""위키백과 HTML 표 → 셀 격자(rowspan·colspan 펼침) + 셀별 각주 출처 URL.

pandas.read_html은 각주 링크를 버리므로 직접 펼친다. 셀마다 (텍스트, 각주 id 목록)을 남기고,
각주 id는 문서의 참고문헌 목록(li#cite_note-…)에서 외부 URL로 바꾼다.
표 문구를 저장하지 않으려고, 결과는 정규화 단계에서 사실값만 뽑은 뒤 버린다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import html as lxml_html

_FOOTNOTE_MARK = re.compile(r"\[(?:[a-z]{1,4}|\d{1,3}|note \d+|N\d+|i{1,3}|iv|v|vi{0,3}|[a-z] \d+)\]", re.I)


@dataclass(frozen=True)
class Cell:
    text: str
    cite_ids: tuple[str, ...] = ()
    is_header: bool = False


@dataclass
class Table:
    headers: list[str]                       # 머리글 행들을 열마다 " / "로 이은 이름
    rows: list[list[Cell]] = field(default_factory=list)
    header_cites: list[tuple[str, ...]] = field(default_factory=list)  # 열 머리글에 붙은 각주(열 전체의 출처)
    caption: str = ""


def _text(el) -> str:
    for bad in el.xpath(".//sup[contains(@class,'reference')]|.//style|.//span[contains(@class,'sortkey')]"):
        bad.drop_tree()
    raw = " ".join(el.itertext())
    raw = _FOOTNOTE_MARK.sub("", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _cite_ids(el) -> tuple[str, ...]:
    ids = []
    for a in el.xpath(".//sup[contains(@class,'reference')]//a[@href]"):
        href = a.get("href", "")
        if href.startswith("#cite_note"):
            ids.append(href[1:])
    return tuple(dict.fromkeys(ids))


def _expand(table_el) -> list[list[Cell]]:
    grid: list[list[Cell | None]] = []
    pending: dict[tuple[int, int], Cell] = {}
    for r, tr in enumerate(table_el.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")):
        row: list[Cell | None] = []
        c = 0
        cells = tr.xpath("./td|./th")
        i = 0
        while i < len(cells) or (r, c) in pending:
            if (r, c) in pending:
                row.append(pending.pop((r, c)))
                c += 1
                continue
            el = cells[i]
            i += 1
            cite = _cite_ids(el)
            cell = Cell(_text(el), cite, el.tag == "th")
            rowspan = int(re.sub(r"\D", "", el.get("rowspan", "1")) or 1)
            colspan = int(re.sub(r"\D", "", el.get("colspan", "1")) or 1)
            for dc in range(colspan):
                row.append(cell)
                for dr in range(1, rowspan):
                    pending[(r + dr, c + dc)] = cell
            c += colspan
        grid.append(row)
    width = max((len(r) for r in grid), default=0)
    return [[x if x is not None else Cell("") for x in r] + [Cell("")] * (width - len(r)) for r in grid]


def parse_tables(html: str) -> list[Table]:
    doc = lxml_html.fromstring(html)
    tables = []
    for t in doc.xpath("//table[contains(@class,'wikitable')]"):
        grid = _expand(t)
        n_head = 0
        while n_head < len(grid) and all(c.is_header or not c.text for c in grid[n_head]):
            n_head += 1
        if not grid or n_head == 0:
            continue
        width = len(grid[0])
        headers, header_cites = [], []
        for col in range(width):
            parts, cites = [], []
            for r in range(n_head):
                cell = grid[r][col] if col < len(grid[r]) else Cell("")
                if cell.text and (not parts or parts[-1] != cell.text):
                    parts.append(cell.text)
                cites += [c for c in cell.cite_ids if c not in cites]
            headers.append(" / ".join(parts))
            header_cites.append(tuple(cites))
        caption = _text(t.xpath("./caption")[0]) if t.xpath("./caption") else ""
        body = [row for row in grid[n_head:] if any(c.text for c in row)]
        tables.append(Table(headers, body, header_cites, caption))
    return tables


def reference_urls(html: str) -> dict[str, list[str]]:
    """각주 id → 외부 URL 목록(위키 내부 링크 제외)."""
    doc = lxml_html.fromstring(html)
    out: dict[str, list[str]] = {}
    for li in doc.xpath("//li[starts-with(@id,'cite_note')]"):
        urls = [
            a.get("href")
            for a in li.xpath(".//a[contains(@class,'external')][@href]")
            if a.get("href", "").startswith("http")
        ]
        out[li.get("id")] = list(dict.fromkeys(urls))
    return out
