"""Chunk 的資料結構（README 4.3）。

`applicable_year_from/to` 與 `applicable_program` 填錯 = 系統給錯答案，
這幾個欄位一律以人工覆核為準（data/doc_overrides.csv、data/chunk_overrides.csv）。
"""
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["校級", "院級", "系級", "作業層", "非正式"]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str                               # 條文原文
    text_with_header: str                   # 加了上下文前綴的版本（實際拿去 embed）

    # --- 過濾用，最重要 ---
    level: Level
    applicable_year_from: int | None = None  # 適用入學學年度起，None = 不分
    applicable_year_to: int | None = None    # None = 現行有效
    applicable_program: list[str] = Field(default_factory=lambda: ["全部"])
    applicable_group: list[str] = Field(default_factory=lambda: ["全部"])

    # --- 回答用 ---
    responsible_office: str
    topic: list[str] = Field(default_factory=list)
    article_no: str | None = None           # "第三條" / "四、" / "二、碩士論文考試申請"
    section_path: str | None = None         # "第二章 入學 > 第五條"
    forms: list[str] = Field(default_factory=list)

    # --- 溯源用 ---
    source_url: str
    revision_date: str | None = None
    is_informal: bool = False

    # --- 覆核用（不進 Chroma filter，但留著追蹤） ---
    year_mentions: list[int] = Field(default_factory=list)   # 條文中出現的學年度
    year_source: Literal["doc", "chunk_override"] = "doc"     # 學年度從哪來
    text_hash: str = ""
