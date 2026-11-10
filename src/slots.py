"""從問句抽出入學學年度與學制，並判斷「該不該反問學年度」。

用規則不用 LLM：學年度的寫法有限（「112入學」「我是113的」「111學年度」），規則可預期、
可測試，也不花錢。介面（Streamlit 側欄）若已選了學年度/學制，以問句裡寫的為優先。

反問規則（README 5.3：會因學年度而異的主題卻沒給學年度 → 反問，不要猜）：
    只有碩士班需要 —— 碩專班準則不分學年度。
    主題清單刻意收窄：「抵免」本身不算（抵免成績、期限不分學年度），「外系」「幾學分」才算。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 「112入學」「112學年度」「112年入學」「112級」「我是112的」「112的學生」
RE_YEAR = [
    re.compile(r"(?<!\d)(1[0-2]\d|9\d)\s*(?:學年度|學年)"),
    re.compile(r"(?<!\d)(1[0-2]\d|9\d)\s*年?\s*(?:入學|進來|進碩|級)"),
    re.compile(r"我是\s*(1[0-2]\d|9\d)\s*(?:的|入學|學年)"),
    re.compile(r"(?<!\d)(1[0-2]\d|9\d)\s*的(?:碩|研究生|學生|學長|學姊)"),
]
RE_EXEC = re.compile(r"碩專|在職專班|在職碩士")

# 依入學學年度而不同的主題（Phase 1–4 實際看到會變動的項目）
YEAR_DEPENDENT = {
    "畢業學分": r"畢業.{0,4}學分|總學分|幾學分|最低.{0,2}學分|要修多少",
    "外系學分": r"外系|外所|校際選課|管理學院.{0,6}課",
    "必修": r"必修",
    "補修": r"補修|基礎課|基礎科目",
    "英文門檻": r"英檢|英文|多益|托福|TOEIC|TOEFL|IELTS|全英",
    "大學部課程": r"大學部|大四|進階課程|合開",
    "學術倫理時數": r"學術倫理.{0,8}(?:幾|多少)?小時|倫理.{0,4}時數",
    "資訊能力": r"CPE|程式能力|資訊能力",
}


@dataclass
class Slots:
    year: int | None = None
    program: str = "碩士班"
    year_source: str = ""                # "question" / "ui" / ""
    program_source: str = ""
    year_topics: list[str] = field(default_factory=list)   # 問句命中的「依學年度而異」主題

    @property
    def needs_year(self) -> bool:
        return self.year is None and self.program == "碩士班" and bool(self.year_topics)


def extract(question: str, ui_year: int | None = None, ui_program: str | None = None) -> Slots:
    s = Slots()
    for pat in RE_YEAR:
        if m := pat.search(question):
            s.year, s.year_source = int(m.group(1)), "question"
            break
    if s.year is None and ui_year is not None:
        s.year, s.year_source = ui_year, "ui"

    if RE_EXEC.search(question):
        s.program, s.program_source = "碩專班", "question"
    elif ui_program:
        s.program, s.program_source = ui_program, "ui"

    s.year_topics = [t for t, pat in YEAR_DEPENDENT.items() if re.search(pat, question, re.I)]
    return s


ASK_YEAR_TEMPLATE = (
    "「{topics}」的規定會依**入學學年度**而不同（例如承認外系學分：106–113 學年度入學為 6 學分，"
    "114 學年度起為 9 學分）。\n\n請問你是哪一學年度入學的？如果是碩士在職專班也請一併告訴我。"
)


def ask_year_message(s: Slots) -> str:
    return ASK_YEAR_TEMPLATE.format(topics="、".join(s.year_topics))
