"""raw → processed：每份 role=content 的來源轉成一份正規化 Markdown。

用法:
    python -m src.ingest

輸出 data/processed/<doc_id>.md，格式固定：

    ---
    doc_id: ...
    revision_date: 114.12.19
    source_method: pdf_text | markdown | manual_transcription
    ---
    # 標題
    ## 沿革            ← chunker 會跳過
    ## 第一條
    ...

每個 `## ` 標題就是一個切塊單位，chunker 只需要認 `## `。結構判斷（第X條 / 一、）
全部在這裡做，這樣 processed/ 可以直接打開來人工檢查切得對不對。

PDF 用 PyMuPDF 的純文字輸出而非 pymupdf4llm：實測 pymupdf4llm 會在數字外面
包反引號、把條文當成 list item，畢業條件明細表的兩欄表格也會錯位；純文字依
閱讀順序輸出，反而乾淨。掃描檔（無文字層）改讀 data/manual/ 的人工轉錄稿。
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

from src.config import MANUAL_DIR, PROCESSED_DIR, RAW_DIR, SOURCES_CSV

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "兩": 2}
CN_UNITS = {"十": 10, "百": 100}
CN_NUM = "零〇一二三四五六七八九十百兩"


def cn2int(s: str) -> int:
    """中文數字 → int（支援到百位：一、十、十五、二十三、一百零二）。"""
    total, cur = 0, 0
    for ch in s:
        if ch in CN_DIGITS:
            cur = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            total += (cur or 1) * CN_UNITS[ch]
            cur = 0
    return total + cur


# ---------- 錨點 ----------
RE_ARTICLE = re.compile(rf"^第\s*([{CN_NUM}]+)\s*條(之[{CN_NUM}]+)?(?=\s|$|[一-鿿（(])")
RE_CHAPTER = re.compile(rf"^【?\s*第\s*([{CN_NUM}]+)\s*章\s*([^】]*)】?$")   # 學則寫成「【第二章 招 生】」
RE_ITEM = re.compile(rf"^([{CN_NUM}]+)\s*、")
# 只用來判斷「這行是新的一行」而不是上一行的折行
RE_LINE_START = re.compile(
    rf"^(第\s*[{CN_NUM}]+\s*[條章]|[{CN_NUM}]+\s*、|[（(][{CN_NUM}\d]+[)）]|\d+\s*[.、]|•|※|【)"
)
RE_PAGE_NO = re.compile(r"^\s*(-?\s*\d+\s*-?|第\s*\d+\s*頁\s*[，,]?\s*共\s*\d+\s*頁)\s*$")
RE_DATE = re.compile(r"(?<![\d.])(\d{2,3})\.(\d{1,2})\.(\d{1,2})(?![\d.])")
RE_PUA = re.compile(r"[-]")
# PDF 常把「六」「⼀」編成相容表意字（U+F900–FAFF）或康熙部首（U+2F00–2FDF），
# 肉眼一樣但 regex / BM25 都對不上，要轉回標準字。只轉這兩段，全形標點保留。
RE_COMPAT = re.compile(r"[豈-﫿⼀-⿟]")
CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def normalize(s: str) -> str:
    s = RE_COMPAT.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), s)
    s = RE_PUA.sub("•", s).strip()
    return re.sub(r"[ \t　\xa0]+", " ", s)


def join_lines(lines: list[str]) -> list[str]:
    """把 PDF 的折行接回邏輯行。中文之間直接相連，英數之間補空白。"""
    out: list[str] = []
    for raw in lines:
        line = normalize(raw)
        if not line or RE_PAGE_NO.match(line):
            continue
        # 「第一條 目的」這種條名行獨立成行，不要和下一行內文接起來
        titled = out and RE_ARTICLE.match(out[-1]) and len(out[-1]) <= 14 and not re.search(r"[，。；：]", out[-1])
        if not out or RE_LINE_START.match(line) or titled:
            out.append(line)
            continue
        prev = out[-1]
        sep = "" if (CJK.match(prev[-1]) or CJK.match(line[0])) else " "
        out[-1] = prev + sep + line
    return out


def pdf_lines(path: Path) -> list[str]:
    doc = pymupdf.open(path)
    lines: list[str] = []
    for page in doc:
        lines.extend(page.get_text().splitlines())
    return lines


def latest_date(lines: list[str]) -> str | None:
    dates = [tuple(map(int, m.groups())) for ln in lines for m in RE_DATE.finditer(ln)]
    dates = [d for d in dates if 80 <= d[0] <= 130 and 1 <= d[1] <= 12 and 1 <= d[2] <= 31]
    if not dates:
        return None
    y, m, d = max(dates)
    return f"{y}.{m}.{d}"


def date_from_title(title: str) -> str | None:
    """教務處法規標題帶 (1141219) 這種修訂日期。"""
    m = re.search(r"\((\d{3})(\d{2})(\d{2})\)", title)
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    return f"{y}.{mo}.{d}"


def structure(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]], str]:
    """把邏輯行分成 (前言/沿革, [(單位標題, 內文行)], 模式)。

    模式 article：以「第X條」為單位，「第X章」併入標題路徑。
    模式 item：以「一、二、三、」為單位（要點類、畢業條件明細表）。
    錨點必須依序遞增，避免把條文中的「依第三條規定」誤判成新條文。
    """
    n_article = sum(1 for ln in lines if RE_ARTICLE.match(ln))
    mode = "article" if n_article >= 2 else "item"

    preamble: list[str] = []
    units: list[tuple[str, list[str]]] = []
    expect = 1
    chapter = ""
    for ln in lines:
        if mode == "article":
            mc = RE_CHAPTER.match(ln)
            if mc:
                name = re.sub(r"\s+", "", mc.group(2) or "")
                chapter = f"第{mc.group(1)}章 {name}".strip()
                continue
            ma = RE_ARTICLE.match(ln)
            if ma:
                n = cn2int(ma.group(1))
                is_sub = bool(ma.group(2))          # 第十四條之一
                if n == expect or (is_sub and n == expect - 1):
                    label = ma.group(0).replace(" ", "")
                    rest = ln[ma.end():].strip()
                    # 第一條  目的 → 標題保留條名
                    if rest and len(rest) <= 12 and not re.search(r"[，。；：]", rest):
                        label, rest = f"{label} {rest}", ""
                    head = f"{chapter} > {label}" if chapter else label
                    units.append((head, [rest] if rest else []))
                    expect = n + 1 if not is_sub else expect
                    continue
        else:
            mi = RE_ITEM.match(ln)
            if mi and cn2int(mi.group(1)) == expect:
                units.append((ln if len(ln) <= 60 else ln[: mi.end()], [] if len(ln) <= 60 else [ln[mi.end():]]))
                expect += 1
                continue
        if units:
            units[-1][1].append(ln)
        else:
            preamble.append(ln)
    return preamble, units, mode


def join_lines_y(lines: list[tuple[float, str]]) -> list[list]:
    """join_lines 的版本，保留每個邏輯行起始的 y 座標。"""
    out: list[list] = []
    for y, raw in lines:
        line = normalize(raw)
        if not line or RE_PAGE_NO.match(line):
            continue
        if not out or RE_LINE_START.match(line):
            out.append([y, line])
            continue
        prev = out[-1][1]
        sep = "" if (CJK.match(prev[-1]) or CJK.match(line[0])) else " "
        out[-1][1] = prev + sep + line
    return out


def grad_req_units(path: Path) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """畢業條件明細表：兩欄表格（項目｜備註），依 x 座標分欄、依 y 座標把備註歸到對應項目。

    純文字輸出會把左右欄串成一行（「抵免學分：不限依本校抵免學分辦法…」），
    數字和條件黏在一起，LLM 讀得懂但 BM25 與人工覆核都很難看。
    """
    header: list[str] = []
    items: list[dict] = []          # {key:(page,y), left:[(y,t)], right:[(y,t)], credits:[(y,n)]}
    expect = 1
    right_lines: list[tuple[int, float, str]] = []
    for pno, page in enumerate(pymupdf.open(path)):
        W, H = page.rect.width, page.rect.height
        raw = []
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                t = normalize("".join(s["text"] for s in ln["spans"]))
                y = ln["bbox"][1]
                # 頁碼只出現在頁首頁尾；表格裡的純數字是「學分數」欄，不能丟
                if t and not (RE_PAGE_NO.match(t) and (y > 0.9 * H or y < 0.06 * H)):
                    raw.append((ln["bbox"][0] / W, y, t))
        raw.sort(key=lambda r: (r[1], r[0]))
        top = next((y for x, y, t in raw if re.fullmatch(r"項\s*目", t)), -1) if pno == 0 else -1
        for x, y, t in raw:
            if y <= top + 2 or re.fullmatch(r"備\s*註|科目名稱|學分數", t) or "承辦人" in t:
                if y < top:
                    header.append(t)
                continue
            if x >= 0.48:
                right_lines.append((pno, y, t))     # 第二輪再分配，避免同列備註比項目早 0.1pt 而歸錯
                continue
            m = RE_ITEM.match(t)
            if m and cn2int(m.group(1)) == expect:
                items.append({"key": (pno, y), "left": [], "right": [], "credits": []})
                expect += 1
            if not items:
                header.append(t)
            elif x > 0.35 and re.fullmatch(r"\d+", t):          # 必修科目表的「學分數」欄
                items[-1]["credits"].append((pno * 10000 + y, t))
            else:
                items[-1]["left"].append((pno * 10000 + y, t))

    # 備註歸給「起始位置在它上方（或同列）」的最後一個項目；跨頁時延續上一頁的項目
    for pno, y, t in right_lines:
        owner = [it for it in items if it["key"] <= (pno, y + 3)]
        if owner:
            owner[-1]["right"].append((pno * 10000 + y, t))
        else:
            header.append(t)

    units = []
    for it in items:
        left = join_lines_y(it["left"])
        for cy, n in it["credits"]:                 # 學分數接到它所屬的科目那一行後面
            course = [ln for ln in left if re.match(r"^\d+\s*\.", ln[1]) and ln[0] <= cy + 5]
            if course:
                course[-1][1] += f"（{n}學分）"
        lines = [t for _, t in left]
        head, body = lines[0], lines[1:]
        right = [t for _, t in join_lines_y(it["right"])]
        if right:
            body += ["", "【備註】", *right]
        units.append((head, body))
    return header, units


def render(meta: dict, title: str, preamble: list[str], units: list[tuple[str, list[str]]]) -> str:
    out = ["---"] + [f"{k}: {v}" for k, v in meta.items() if v is not None] + ["---", "", f"# {title}", ""]
    if preamble:
        out += ["## 沿革", "", *preamble, ""]
    for head, body in units:
        out += [f"## {head}", "", *body, ""]
    return "\n".join(out)


def ingest_pdf(row: dict, path: Path, report: dict) -> str:
    lines = join_lines(pdf_lines(path))
    if not lines:
        raise ValueError("PDF 無文字層（掃描檔），請在 data/manual/ 放人工轉錄稿")
    doc_id = row["doc_id"]
    if doc_id.startswith("grad-req-"):
        preamble, units = grad_req_units(path)
        mode = "table"
    else:
        preamble, units, mode = structure(lines)
        preamble = preamble[1:]                     # 第一行是法規名稱，標題已用 sources.csv 的

    revision = date_from_title(row["title"])
    if doc_id.startswith("grad-req-"):
        m = re.search(r"(\d{2,3}\.\d{1,2}\.\d{1,2})\s*版", " ".join(preamble))
        revision = m.group(1) if m else revision
        # 檢查表頭學年度和 sources.csv 是否一致
        ym = re.search(r"(\d{3})\s*學年度", " ".join(preamble))
        if ym and row["year_from"] and int(ym.group(1)) != int(row["year_from"]):
            report["warnings"].append(
                f"{doc_id}: 表頭為 {ym.group(1)} 學年度，sources.csv year_from={row['year_from']}")
    revision = revision or latest_date(preamble)

    report["docs"][doc_id] = {"method": "pdf_text", "mode": mode, "units": len(units),
                              "preamble_lines": len(preamble), "revision_date": revision}
    if len(units) < 2:
        report["warnings"].append(f"{doc_id}: 只切出 {len(units)} 個單位，請檢查 processed 檔")
    meta = {"doc_id": doc_id, "revision_date": revision, "source_method": "pdf_text", "mode": mode}
    return render(meta, row["title"], preamble, units)


def ingest_markdown(row: dict, path: Path, method: str, report: dict) -> str:
    """已轉存的 Markdown（網頁擷取 / 人工轉錄）：去掉來源資訊前言，保留 ## 結構。"""
    text = path.read_text("utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    body = text[text.index("\n## "):] if "\n## " in text else text
    body = re.sub(r"\n-{3,}\n", "\n", body)            # 分隔線
    units = len(re.findall(r"^## ", body, re.M))
    m = re.search(r"擷取日期：(\S+)", text)
    revision = date_from_title(row["title"]) or (m.group(1) if m else None)
    if method == "manual_transcription":
        dm = re.findall(r"(\d{2,3}\.\d{1,2}\.\d{1,2})\s*版|\((\d{2,3}\.\d{1,2}\.\d{1,2})\)", text)
        revision = latest_date([" ".join(a or b for a, b in dm)]) or revision
    report["docs"][row["doc_id"]] = {"method": method, "mode": "heading", "units": units,
                                     "revision_date": revision}
    meta = {"doc_id": row["doc_id"], "revision_date": revision, "source_method": method, "mode": "heading"}
    return render(meta, row["title"], [], []) + body.strip() + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCES_CSV.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("role") or "content") == "content"]

    report: dict = {"docs": {}, "warnings": [], "errors": []}
    for row in rows:
        doc_id = row["doc_id"]
        manual = MANUAL_DIR / f"{doc_id}.md"
        raw_md = RAW_DIR / f"{doc_id}.md"
        raw_pdf = RAW_DIR / f"{doc_id}.pdf"
        try:
            if manual.exists():
                out = ingest_markdown(row, manual, "manual_transcription", report)
            elif raw_md.exists():
                out = ingest_markdown(row, raw_md, "markdown", report)
            elif raw_pdf.exists():
                out = ingest_pdf(row, raw_pdf, report)
            else:
                report["errors"].append(f"{doc_id}: 找不到原始檔")
                continue
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{doc_id}: {exc}")
            continue
        (PROCESSED_DIR / f"{doc_id}.md").write_text(out, "utf-8")
        d = report["docs"][doc_id]
        print(f"[ok] {doc_id:32s} {d['method']:22s} {d['mode']:8s} units={d['units']:3d} rev={d['revision_date']}")

    (PROCESSED_DIR / "_ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    for w in report["warnings"]:
        print("[warn]", w)
    for e in report["errors"]:
        print("[err] ", e)
    print(f"\n{len(report['docs'])} 份完成 / {len(report['warnings'])} 警告 / {len(report['errors'])} 錯誤")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
