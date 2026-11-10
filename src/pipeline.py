"""主入口：answer(question) → 反問學年度，或 檢索 + 生成。

用法:
    python -m src.pipeline "我112入學，外系學分可以抵幾學分？"
    python -m src.pipeline "外系學分可以抵幾學分？" --year 114        # 模擬介面已選學年度
    python -m src.pipeline "碩專班最低畢業學分" --dry                  # 不呼叫 LLM，只看抽槽、檢索、prompt

流程：
    slots.extract()  → 需要學年度卻沒有 → 回傳反問，不檢索、不呼叫 LLM
                     → 否則 retrieve.search(year, program) → generate()
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field

from src import slots as slot_mod
from src.config import GEN_TOP_K

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieve import Retriever

        _retriever = Retriever()
    return _retriever


@dataclass
class Answer:
    status: str                      # "answer" | "ask_year"
    text: str
    year: int | None
    program: str
    year_source: str
    year_topics: list[str]
    sources: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


MAX_CHUNKS = 8          # context 上限：7B 的 num_ctx=8192，合併後的長節不能無限塞


def select_hits(hits: list, k: int) -> list:
    """取前 k 個「不同節」。同一節的多塊會合併成一個來源，只取 k 塊常常只剩 2 個來源。"""
    from src.generate import section_key

    keys, out = [], []
    for h in hits:
        key = section_key(h.chunk)
        if key not in keys:
            if len(keys) == k:
                continue
            keys.append(key)
        if len(out) < MAX_CHUNKS:
            out.append(h)
    return out


def answer(question: str, ui_year: int | None = None, ui_program: str | None = None,
           k: int = GEN_TOP_K, dry: bool = False) -> Answer:
    s = slot_mod.extract(question, ui_year, ui_program)
    base = dict(year=s.year, program=s.program, year_source=s.year_source, year_topics=s.year_topics)
    if s.needs_year:
        return Answer(status="ask_year", text=slot_mod.ask_year_message(s), **base)

    hits = select_hits(get_retriever().search(question, s.year, s.program, k * 2), k)
    from src.generate import group_hits

    # 與 context 的 <source id> 同一套編號：同一節的多塊已合併
    sources = [{"n": i, "chunk_id": g["chunk_ids"][0], "chunk_ids": g["chunk_ids"], "title": g["title"],
                "article_no": g["section"], "office": g["responsible_office"], "url": g["source_url"],
                "parts": g["parts"]} for i, g in enumerate(group_hits(hits), 1)]
    if dry:
        from src.generate import SYSTEM_PROMPT, format_user

        prompt = f"[system]\n{SYSTEM_PROMPT}\n\n[user]\n{format_user(question, s.year, s.program, hits)}"
        return Answer(status="dry", text=prompt, sources=sources, **base)

    from src.generate import generate

    g = generate(question, s.year, s.program, hits)
    usage = {"model": g.model, "input_tokens": g.input_tokens, "output_tokens": g.output_tokens,
             "cache_read_tokens": g.cache_read_tokens, "stop_reason": g.stop_reason, "fell_back": g.fell_back}
    return Answer(status="answer", text=g.text, sources=sources, usage=usage, **base)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--year", type=int)
    ap.add_argument("--program")
    ap.add_argument("-k", type=int, default=GEN_TOP_K)
    ap.add_argument("--dry", action="store_true", help="不呼叫 LLM，印出 prompt")
    args = ap.parse_args()
    a = answer(args.question, args.year, args.program, args.k, args.dry)
    print(f"[{a.status}] 學年度={a.year}（{a.year_source or '無'}） 學制={a.program}  主題={a.year_topics}\n")
    print(a.text)
    if a.sources and a.status != "dry":
        print("\n來源 chunk：")
        for s in a.sources:
            print(f"  [{s['n']}] {','.join(s['chunk_ids'])}  {s['title']}｜{s['article_no']}")
    if a.usage:
        print("\n", a.usage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
