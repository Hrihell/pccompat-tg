"""config/files.yml의 파일 원천을 받아 sha256·크기·시각을 manifest에 남긴다(이미 있으면 건너뜀)."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone

import requests
import yaml

from pccompat.config import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "config" / "files.yml"
MANIFEST = PROJECT_ROOT / "data" / "raw" / "files_manifest.csv"
FIELDS = ["id", "url", "dest", "sha256", "bytes", "fetched_at", "license", "attribution"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = {}
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            rows = {r["id"]: r for r in csv.DictReader(f)}
    session = requests.Session()
    session.headers["User-Agent"] = cfg["user_agent"]
    for spec in cfg["files"]:
        dest = PROJECT_ROOT / spec["dest"]
        if dest.exists() and not args.refresh and spec["id"] in rows:
            print(f"skip {spec['id']}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with session.get(spec["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        rows[spec["id"]] = {
            "id": spec["id"], "url": spec["url"], "dest": spec["dest"], "sha256": sha256(dest),
            "bytes": dest.stat().st_size, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "license": spec["license"], "attribution": spec["attribution"],
        }
        print(f"ok   {spec['id']} {rows[spec['id']]['bytes']:,} bytes")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows.values())


if __name__ == "__main__":
    main()
