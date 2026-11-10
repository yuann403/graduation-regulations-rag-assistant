"""processed → chunks.jsonl，並產出人工覆核表。

用法:
    python -m src.chunker

切塊：每個 `## ` 標題（條 / 項 / 段）一塊；超過 MAX_CHARS 才依子項（一、/(一)/1.）二次切，
子塊在標題補上「（續 2/3）」。

學年度 metadata 的來源與優先順序（高 → 低）：
    1. data/chunk_overrides.csv   人工覆核後對單一 chunk 的覆寫
    2. data/doc_overrides.csv     人工覆核後對整份文件的修正（例：系網連結標錯學年度）
    3. data/sources.csv           Phase 1 從系網索引頁標的值

為什麼「條文內出現 XXX 學年度」不自動覆寫 chunk 的學年度：
    實際跑過一遍，條文裡的學年度幾乎都是下面兩種，自動覆寫反而會把對的改成錯的——
    (a) 同一句話自己區分新舊制：「105學年度(含)入學者5小時；106學年度(含)以後6小時」。
        這塊要讓所有學年度都撈得到，由 LLM 依使用者學年度挑句子；若標 year_from=106，
        105 入學的人就查不到自己的規定。
    (b) 學年度指的是「施行年度」不是「入學年度」：「114學年度（含）起，學生應於學位
        考試前提出申請論文延後公開」——這是 114 學年度起所有考試的人，不分入學年。
    因此 chunker 只「偵測 + 建議」，寫進 data/review/chunks_review.csv 的 flags 與
    suggestion 欄；確認後由人填進 chunk_overrides.csv 才生效。
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.config import (CHUNK_OVERRIDES_CSV, CHUNKS_JSONL, DOC_OVERRIDES_CSV, MAX_CHARS,
                        PROCESSED_DIR, REVIEW_DIR, SOURCES_CSV)
from src.schema import Chunk

SKIP_SECTIONS = {"沿革"}

TOPICS = {
    "畢業學分": r"畢業總?學分|最低畢業",
    "抵免": r"抵免",
    "外系學分": r"外系|校際選課",
    "必修": r"必修",
    "補修": r"補修",
    "選課": r"選課|加退選|大學部相關課程|進階課程|合開",
    "英文門檻": r"英文能力|英檢|英文檢定|托福|TOEIC|多益|IELTS|全英",
    "學術倫理": r"學術倫理|倫理教育",
    "論文考試": r"學位考試|論文考試|口試|考試委員",
    "指導教授": r"指導教授",
    "修業年限": r"修業年限|休學|退學",
    "提前畢業": r"提前畢業",
    "離校": r"離校",
    "論文繳交": r"圖書館|電子學位論文|送印|延後公開|授權書",
    "原創性比對": r"Turnitin|原創性比對|論文比對",
    "成績": r"成績|及格",
    "獎助學金": r"獎助學金|助學金",
    "碩專班": r"在職專班|碩專班",
    "逕讀博士": r"逕修讀博士|逕讀博士",
    "資訊能力": r"程式能力檢定|CPE|資訊能力",
}
FORM_SUFFIX = r"(?:申請書|申請表|同意書|通知單|通知書|評分單|檢核表|審核頁|手續單|約定書|授權書|回條)"
RE_FORM_QUOTED = re.compile(rf"「([^「」]{{2,30}}{FORM_SUFFIX})」")
RE_FORM_BARE = re.compile(rf"([一-鿿]{{2,14}}{FORM_SUFFIX})")
FORM_LEADING = re.compile(r"^.*?(?:填寫|填具|繳交|檢附|提出|列印|附|及|與|或|之|須|需|另有|並)")

RE_YEAR = re.compile(r"(\d{2,3})\s*學年度")
RE_YEAR_FROM = re.compile(r"(\d{2,3})\s*學年度\s*[（(]?\s*含\s*[)）]?\s*(?:起|以後|之後|後)?\s*入學")
RE_YEAR_FROM2 = re.compile(r"(\d{2,3})\s*學年度\s*(?:起|以後|之後)\s*入學")
RE_YEAR_TO = re.compile(r"(\d{2,3})\s*學年度\s*[（(]?\s*含\s*[)）]?\s*(?:以前|之前|前)\s*入學")
# 「本辦法經教務會議通過後公告施行」「依據學則第二十條訂定本辦法」這類條文沒有資訊量，
# 但用詞和真正條文很像，會在檢索時佔名額
# 錨定在條文「開頭」：只比對主詞是本法規自己的句子，避免誤殺「⋯報校長核定後實施之」這種真條文
_SELF = r"本(?:校)?(?:作業)?(?:辦法|細則|要點|章程|準則|學則|規定|原則)"
RE_BOILERPLATE = re.compile(
    rf"^(?:{_SELF}[^。]*(?:通過|核定|核可|發布)[^。]*(?:施行|實施)[^。]*。"     # 本X經OO通過後實施，修正時亦同。
    rf"|{_SELF}(?:如有)?未盡事宜[^。]*。"                                   # 本X未盡事宜，依⋯辦理。
    rf"|[^。]{{0,90}}訂定(?:本\S{{0,4}}|之)。)"                             # 本X依據⋯訂定之。／為⋯，特訂定本辦法。
    rf"(?:[^。]*(?:修正|修訂)時亦同。)?$")                                   # 整塊只有這一句才算
RE_ARTICLE_PREFIX = re.compile(r"^(?:第[一二三四五六七八九十百]+條(?:之[一二三四五六七八九十]+)?(?:\s+\S{1,10}\n)?|[一二三四五六七八九十]+、)\s*")
RE_SUBSPLIT =re.compile(r"^(?:[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[)）]|\d+\s*\.)")


# ---------------- 讀檔 ----------------
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def parse_year(v: str | None) -> int | None | str:
    """空字串 = 不改；'none' = 清成 None；數字 = 該值。"""
    v = (v or "").strip()
    if not v:
        return "keep"
    if v.lower() == "none":
        return None
    return int(v)


def parse_processed(path: Path) -> tuple[dict, str, list[tuple[str, str]]]:
    text = path.read_text("utf-8")
    meta: dict = {}
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        text = text[m.end():]
    title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    sections = []
    for i in range(1, len(parts), 2):
        head, body = parts[i].strip(), parts[i + 1].strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        sections.append((head, body))
    return meta, title, sections


# ---------------- 切塊 ----------------
def split_long(head: str, body: str) -> list[tuple[str, str]]:
    """單條超過 MAX_CHARS 才切；切點優先找子項錨點，其次段落。"""
    if len(head) + len(body) <= MAX_CHARS:
        return [(head, body)]
    lines = body.split("\n")
    blocks: list[str] = []
    for ln in lines:
        if not blocks or RE_SUBSPLIT.match(ln.strip()) or (ln.strip() == "" and blocks[-1].strip()):
            blocks.append(ln)
        else:
            blocks[-1] += "\n" + ln
    parts: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if parts and len(parts[-1]) + len(b) + 1 <= MAX_CHARS:
            parts[-1] += "\n" + b
        else:
            parts.append(b)
    if len(parts) == 1:
        return [(head, body)]
    n = len(parts)
    return [(f"{head}（續 {i}/{n}）" if i > 1 else f"{head}（{i}/{n}）", p) for i, p in enumerate(parts, 1)]


def split_paragraphs(head: str, body: str, min_len: int = 60) -> list[tuple[str, str]]:
    """網頁轉存的流程頁：一段 = 一個步驟（「完成學術倫理認證課程：⋯」「Turnitin 比對：⋯」）。

    一整節塞五件事時，BM25 會被其他文件裡重複出現的同一個詞壓過去（實測「學術倫理要修
    幾小時」輸給了學術倫理*違反*處理原則）。所以長節依段落切，太短的段落併到前一段，
    標題保留節名，「提出論文考試申請前」這種時間脈絡不會丟。
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    groups: list[str] = []
    for p in paras:
        if groups and (len(p) < min_len or len(groups[-1]) < min_len):
            groups[-1] += "\n" + p
        else:
            groups.append(p)
    if len(groups) <= 1:
        return [(head, body)]
    n = len(groups)
    return [(f"{head}（{i}/{n}）" if i == 1 else f"{head}（續 {i}/{n}）", g) for i, g in enumerate(groups, 1)]


def article_label(head: str) -> str:
    """section_path 的最後一段，去掉續號，再截短成條次。"""
    last = head.split(" > ")[-1]
    last = re.sub(r"（(續 )?\d+/\d+）$", "", last)
    m = re.match(r"^(第[一二三四五六七八九十百]+條(之[一二三四五六七八九十]+)?)", last)
    if m:
        return m.group(1)
    m = re.match(r"^([一二三四五六七八九十]+、)", last)
    if m:
        return m.group(1)
    return last[:30]


def find_forms(text: str) -> list[str]:
    found = [m.group(1) for m in RE_FORM_QUOTED.finditer(text)]
    for m in RE_FORM_BARE.finditer(text):
        name = FORM_LEADING.sub("", m.group(1))
        if len(name) >= 4:
            found.append(name)
    out: list[str] = []
    for f in found:
        if not any(f in o or o in f for o in out):
            out.append(f)
    return out


def year_suggestion(text: str) -> tuple[list[int], str]:
    years = sorted({int(y) for y in RE_YEAR.findall(text) if 90 <= int(y) <= 130})
    if not years:
        return [], ""
    froms = {int(y) for y in RE_YEAR_FROM.findall(text) + RE_YEAR_FROM2.findall(text)}
    tos = {int(y) for y in RE_YEAR_TO.findall(text)}
    if not froms and not tos:
        return years, "學年度非指入學年（可能是施行年度/會議紀錄），勿覆寫"
    if len(froms | tos) > 1 or (froms and tos):
        return years, "條文自行區分新舊制，建議維持不分學年度"
    if froms:
        return years, f"可能 year_from={froms.pop()}（請確認整塊都只適用該學年度起）"
    return years, f"可能 year_to={tos.pop()}（請確認整塊都只適用該學年度前）"


def header(title: str, rev: str | None, yf, yt, program: list[str], office: str, path: str) -> str:
    if yf is None and yt is None:
        scope = "不分入學學年度"
    elif yt is None:
        scope = f"{yf} 學年度(含)以後入學"
    elif yf is None:
        scope = f"{yt} 學年度(含)以前入學"
    elif yf == yt:
        scope = f"{yf} 學年度入學"
    else:
        scope = f"{yf}–{yt} 學年度入學"
    prog = "、".join(program)
    rev_s = f"（{rev} 修訂）" if rev and re.match(r"\d", rev) and "-" not in rev else ""
    return (f"【文件】{title}{rev_s}\n【適用】{scope}之{prog}\n【主責】{office}\n【條次】{path}\n---\n")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sources = {r["doc_id"]: r for r in read_csv(SOURCES_CSV) if (r.get("role") or "content") == "content"}
    doc_ovr = {r["doc_id"]: r for r in read_csv(DOC_OVERRIDES_CSV)}
    chunk_ovr = {r["chunk_id"]: r for r in read_csv(CHUNK_OVERRIDES_CSV)}

    chunks: list[Chunk] = []
    review: list[dict] = []
    seen: dict[tuple, str] = {}
    dropped: list[tuple[str, str]] = []
    boilerplate: list[str] = []

    for doc_id, src in sources.items():
        path = PROCESSED_DIR / f"{doc_id}.md"
        if not path.exists():
            print(f"[miss] {doc_id}: 沒有 processed 檔，先跑 python -m src.ingest")
            continue
        meta, _, sections = parse_processed(path)

        yf = int(src["year_from"]) if src["year_from"] else None
        yt = int(src["year_to"]) if src["year_to"] else None
        program = [p.strip() for p in (src["program"] or "全部").split("|")]
        doc_flags: list[str] = []
        if doc_id in doc_ovr:
            o = doc_ovr[doc_id]
            v = parse_year(o.get("year_from"))
            yf = yf if v == "keep" else v
            v = parse_year(o.get("year_to"))
            yt = yt if v == "keep" else v
            if (o.get("program") or "").strip():
                program = [p.strip() for p in o["program"].split("|")]
            doc_flags.append("doc_override")
        if meta.get("source_method") == "manual_transcription":
            doc_flags.append("manual_transcription")

        seq = 0
        for head, body in sections:
            if head in SKIP_SECTIONS or not body and len(head) < 8:
                continue
            by_para = meta.get("source_method") == "markdown" and len(body) > 400
            for sub_head, sub_body in (split_paragraphs(head, body) if by_para else split_long(head, body)):
                seq += 1
                chunk_id = f"{doc_id}#{seq:02d}"
                text = f"{sub_head}\n{sub_body}".strip()
                c_yf, c_yt, c_prog, year_source = yf, yt, program, "doc"
                flags = list(doc_flags)
                if chunk_id in chunk_ovr:
                    o = chunk_ovr[chunk_id]
                    v = parse_year(o.get("year_from"))
                    c_yf = c_yf if v == "keep" else v
                    v = parse_year(o.get("year_to"))
                    c_yt = c_yt if v == "keep" else v
                    if (o.get("program") or "").strip():
                        c_prog = [p.strip() for p in o["program"].split("|")]
                    year_source = "chunk_override"
                    flags.append("chunk_override")

                body_core = RE_ARTICLE_PREFIX.sub("", text.split(" > ")[-1]).replace("\n", "").strip()
                if RE_BOILERPLATE.match(body_core):
                    boilerplate.append(chunk_id)
                    continue
                years, suggestion = year_suggestion(text)
                # 文件本身已有學年度、條文裡的學年度又只是會議紀錄 → 不用人看
                noise = (c_yf is not None or c_yt is not None) and suggestion.startswith("學年度非指入學年")
                if years and year_source == "doc" and not noise:
                    flags.append("year_mention")
                if len(text) < 40:
                    flags.append("short")

                # 去重鍵含適用範圍：各學年度明細表的「一、修業年限」內容完全相同，
                # 但分屬不同學年度，不能被當成重複刪掉
                norm = re.sub(r"\s+", "", sub_body or sub_head)
                h = hashlib.sha1(norm.encode()).hexdigest()[:16]
                key = (h, c_yf, c_yt, tuple(c_prog))
                if key in seen:
                    dropped.append((chunk_id, seen[key]))
                    continue
                seen[key] = chunk_id

                sec_path = sub_head
                chunk = Chunk(
                    chunk_id=chunk_id, doc_id=doc_id, title=src["title"], text=text,
                    text_with_header=header(src["title"], meta.get("revision_date"), c_yf, c_yt,
                                            c_prog, src["office"], sec_path) + text,
                    level=src["level"], applicable_year_from=c_yf, applicable_year_to=c_yt,
                    applicable_program=c_prog, responsible_office=src["office"],
                    topic=[t for t, pat in TOPICS.items() if re.search(pat, text)],
                    article_no=article_label(sub_head), section_path=sec_path,
                    forms=find_forms(text), source_url=src["url"],
                    revision_date=meta.get("revision_date"),
                    year_mentions=years, year_source=year_source, text_hash=h,
                )
                chunks.append(chunk)
                review.append({
                    "chunk_id": chunk_id, "doc_id": doc_id, "article_no": chunk.article_no,
                    "year_from": c_yf if c_yf is not None else "", "year_to": c_yt if c_yt is not None else "",
                    "program": "|".join(c_prog), "flags": "|".join(flags), "year_mentions": " ".join(map(str, years)),
                    "suggestion": suggestion, "topic": "|".join(chunk.topic), "forms": "|".join(chunk.forms),
                    "chars": len(text), "preview": text.replace("\n", " ")[:160],
                })

    CHUNKS_JSONL.write_text("".join(c.model_dump_json() + "\n" for c in chunks), "utf-8")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_path = REVIEW_DIR / "chunks_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as fh:     # utf-8-sig：Excel 直接開不亂碼
        w = csv.DictWriter(fh, fieldnames=list(review[0].keys()))
        w.writeheader()
        w.writerows(review)

    # ---- 摘要 ----
    lens = sorted(len(c.text) for c in chunks)
    flags = Counter(f for r in review for f in r["flags"].split("|") if f)
    print(f"chunks: {len(chunks)}（{len(sources)} 份文件）  去重丟棄: {len(dropped)}  制式條文丟棄: {len(boilerplate)}")
    print(f"長度 min/中位/max: {lens[0]} / {lens[len(lens)//2]} / {lens[-1]}")
    print("flags:", dict(flags))
    print("無 topic 的 chunk:", sum(1 for c in chunks if not c.topic))
    for d, keep in dropped:
        print(f"  [dup] {d} == {keep}")
    print(f"\n→ {CHUNKS_JSONL}\n→ {review_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
