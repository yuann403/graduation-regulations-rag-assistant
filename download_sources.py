"""讀 data/sources.csv，把 Google Drive / PDF 來源抓到 data/raw/，並記錄 sha256。

用法:
    python scripts/download_sources.py            # 只抓還沒抓過的
    python scripts/download_sources.py --force    # 全部重抓

HTML 來源不在這裡處理：系網有幾頁是 Elementor JS 分頁（資管系辦法規範、
論文考試流程表單），requests 抓不到內容，要用瀏覽器展開後另存。
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "sources.csv"
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "raw_manifest.json"

DRIVE_ID = re.compile(r"/d/([A-Za-z0-9_-]+)")


def direct_url(url: str) -> str | None:
    """把來源網址轉成可直接下載的網址；不是檔案就回 None。"""
    m = DRIVE_ID.search(url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "work=down" in url:          # 中興教務處法規頁的下載端點
        return url
    if url.lower().endswith((".pdf", ".doc", ".docx", ".odt")):
        return url
    return None


def sniff_ext(content: bytes, fallback: str = ".bin") -> str:
    if content[:4] == b"%PDF":
        return ".pdf"
    if content[:2] == b"PK":          # docx / odt / xlsx 都是 zip
        return ".docx"
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text("utf-8")) if MANIFEST.exists() else {}

    with SRC.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    session = requests.Session()
    session.headers["User-Agent"] = "nchu-mis-rag/0.1 (student project)"

    ok = skipped = failed = 0
    for row in rows:
        doc_id, url = row["doc_id"], row["url"]
        role = (row.get("role") or "content").strip()
        if role != "content":
            # index = 只拿來挖連結，external_link = 只填進 checklist，兩者都不進向量庫
            print(f"[{role[:4]}] {doc_id}: 依 role 欄略過，不需下載")
            skipped += 1
            continue
        dl = direct_url(url)
        if not dl:
            print(f"[skip] {doc_id}: 非直接下載來源（{row['fmt']}），需另行處理")
            skipped += 1
            continue
        if doc_id in manifest and not args.force:
            print(f"[have] {doc_id}")
            skipped += 1
            continue

        try:
            r = session.get(dl, timeout=60, allow_redirects=True)
            r.raise_for_status()
            body = r.content
            # Drive 對大檔會先回一頁確認頁，而不是檔案本身
            if body[:15].lower().startswith(b"<!doctype html") or b"<html" in body[:200].lower():
                print(f"[warn] {doc_id}: 收到 HTML 而非檔案，可能需要手動下載 -> {url}")
                failed += 1
                continue
            ext = sniff_ext(body)
            path = RAW / f"{doc_id}{ext}"
            path.write_bytes(body)
            manifest[doc_id] = {
                "title": row["title"],
                "url": url,
                "file": path.name,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            print(f"[ok]   {doc_id} -> {path.name} ({len(body):,} bytes)")
            ok += 1
        except Exception as exc:                      # noqa: BLE001
            print(f"[fail] {doc_id}: {exc}")
            failed += 1
        time.sleep(0.5)

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n完成：{ok} 下載 / {skipped} 略過 / {failed} 失敗")
    print(f"清單寫入 {MANIFEST}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
