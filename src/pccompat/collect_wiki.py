"""위키백과 수집기: 설정의 템플릿·절을 MediaWiki API로 받아 원본 HTML과 리비전을 저장한다.

- 공식 API만 사용, User-Agent에 연락처, 요청 사이 최소 간격(위키미디어 이용 정책).
- 저장: data/raw/wikipedia/{id}.json = {id, page, section_index, revid, fetched_at, url, html}
  원본 HTML은 내부 작업용이며 공개하지 않는다(공개 산출물은 정규화된 사실값 + 출처 URL뿐).
- 이미 받은 파일은 다시 받지 않는다(--refresh로 강제).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from pccompat.config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wikipedia"
CONFIG = PROJECT_ROOT / "config" / "wikipedia.yml"


class WikiClient:
    def __init__(self, api: str, user_agent: str, min_interval: float):
        self.api = api
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, **params: Any) -> dict[str, Any]:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(self.api, params={"format": "json", "formatversion": 2, **params}, timeout=60)
        self._last = time.monotonic()
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MediaWiki API 오류: {data['error']}")
        return data

    def sections(self, page: str) -> list[dict[str, Any]]:
        return self.get(action="parse", page=page, prop="sections", redirects=1)["parse"]["sections"]

    def parse(self, page: str, section: str | None = None) -> dict[str, Any]:
        params = {"action": "parse", "page": page, "prop": "text|revid", "redirects": 1, "disableeditsection": 1}
        if section is not None:
            params["section"] = section
        return self.get(**params)["parse"]


def _plain(line: str) -> str:
    return re.sub(r"<[^>]+>", "", line).strip()


def resolve_section(sections: list[dict[str, Any]], title: str, under: str | None = None) -> str:
    """제목 정규식(+상위 절 정규식)에 맞는 첫 절의 index. 없거나 여럿이면 오류(조용히 엉뚱한 절을 받지 않게)."""
    stack: list[tuple[int, str]] = []
    hits = []
    for s in sections:
        level = int(s["level"])
        while stack and stack[-1][0] >= level:
            stack.pop()
        name = _plain(s["line"])
        ancestors = [n for _, n in stack]
        if re.search(title, name) and (under is None or any(re.search(under, a) for a in ancestors)):
            hits.append((s["index"], name, ancestors))
        stack.append((level, name))
    # 같은 절의 하위 절이 함께 걸리면 가장 바깥 절을 쓴다(하위 표까지 포함됨)
    outer = [h for h in hits if not any(o[1] in h[2] for o in hits if o is not h)]
    if len(outer) != 1:
        raise LookupError(f"절 선택이 모호하거나 없음: title={title!r} under={under!r} → {[(h[0], h[1]) for h in hits]}")
    return str(outer[0][0])


def fetch_source(client: WikiClient, src: dict[str, Any], refresh: bool = False) -> Path:
    out = RAW_DIR / f"{src['id']}.json"
    if out.exists() and not refresh:
        return out
    if "template" in src:
        page, section = src["template"], None
    else:
        page = src["page"]
        sec = src["section"]
        section = resolve_section(client.sections(page), sec["title"], sec.get("under"))
    parsed = client.parse(page, section)
    record = {
        "id": src["id"],
        "page": parsed["title"],
        "section_index": section,
        "revid": parsed["revid"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": f"https://en.wikipedia.org/w/index.php?title={parsed['title'].replace(' ', '_')}&oldid={parsed['revid']}",
        "html": parsed["text"],
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="위키백과 스펙 표 원본 수집")
    ap.add_argument("--only", nargs="*", help="수집할 source id")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    client = WikiClient(cfg["api"], cfg["user_agent"], float(cfg["min_interval_sec"]))
    failures = []
    for src in cfg["sources"]:
        if args.only and src["id"] not in args.only:
            continue
        try:
            path = fetch_source(client, src, args.refresh)
            print(f"ok   {src['id']:24} {path.name}")
        except (LookupError, RuntimeError, requests.RequestException) as e:
            failures.append(src["id"])
            print(f"FAIL {src['id']:24} {e}")
    if failures:
        raise SystemExit(f"실패 {len(failures)}건: {failures}")


if __name__ == "__main__":
    main()
