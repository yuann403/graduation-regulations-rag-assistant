"""Streamlit 聊天介面。

用法:
    streamlit run app.py

側欄選入學學年度與學制 → 對應 pipeline.answer(ui_year, ui_program)。
問句裡寫的學年度（「我112入學…」）優先於側欄。
系統反問學年度時，直接回「112」或「我是112入學的」，會自動用原本的問題重新回答。
"""
from __future__ import annotations

import re
import time

import requests
import streamlit as st

from src.config import LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL
from src.slots import extract

st.set_page_config(page_title="中興資管碩士班 畢業規定查詢", page_icon="🎓", layout="centered")

YEARS = list(range(115, 100, -1))
EXAMPLES = [
    "外系學分最多可以承認幾學分？",
    "下學期想口試，現在開始要準備什麼？",
    "論文考試申請書要在口試前幾天交？",
    "Turnitin 帳號要跟誰拿？",
    "畢業離校手續要怎麼跑？",
]
RE_BARE_YEAR = re.compile(r"^\s*(1[0-2]\d|9\d)\s*$")
RE_HEADING = re.compile(r"^#{1,4}\s*(.+?)\s*$", re.M)


def tidy(text: str) -> str:
    """模型回的「## 結論」在聊天泡泡裡會變成跟頁面標題一樣大，改成粗體小標。"""
    return RE_HEADING.sub(r"\n\n**\1**\n\n", text)   # 前後空行：小標自成一段，不和下一行黏在一起


# ---------------------------------------------------------------- 資源（只載一次）
@st.cache_resource(show_spinner="載入檢索模型（第一次約 10 秒）…")
def load_pipeline():
    from src import pipeline

    pipeline.get_retriever()          # 預先載入 bge-m3、Chroma、BM25
    return pipeline


@st.cache_data(ttl=30, show_spinner=False)
def ollama_status() -> tuple[bool, str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        names = [m["name"] for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in names:
            return False, f"Ollama 有在跑，但沒有 {OLLAMA_MODEL}（請執行 ollama pull {OLLAMA_MODEL}）"
        return True, f"{OLLAMA_MODEL} 就緒"
    except requests.RequestException:
        return False, "連不到 Ollama，請先開啟 Ollama"


# ---------------------------------------------------------------- 側欄
with st.sidebar:
    st.header("你的身分")
    year_label = st.selectbox("入學學年度", ["未選擇"] + [f"{y} 學年度" for y in YEARS], index=0,
                              help="畢業學分、外系學分、英檢等規定會依入學學年度不同")
    ui_year = None if year_label == "未選擇" else int(year_label.split()[0])
    ui_program = st.radio("學制", ["碩士班", "碩專班"], horizontal=True)

    st.divider()
    if LLM_PROVIDER == "ollama":
        ok, msg = ollama_status()
        (st.success if ok else st.error)(msg, icon="✅" if ok else "⚠️")
    else:
        st.info("LLM：Claude API", icon="☁️")
    show_debug = st.toggle("顯示檢索細節", value=False, help="顯示每個來源的 chunk_id，除錯用")
    if st.button("清除對話", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.divider()
    st.caption("資料來源：資管系系網、教務處法規（2026-11 擷取）。"
               "本系統僅供參考，請以各處室公告與承辦人說明為準。")


# ---------------------------------------------------------------- 狀態
st.session_state.setdefault("messages", [])     # [{role, content, meta}]
st.session_state.setdefault("pending", None)     # 被反問學年度時暫存的原問題


def render_meta(meta: dict) -> None:
    if not meta:
        return
    if meta.get("status") == "answer":
        who = f"{meta['year']} 學年度" if meta.get("year") else "未指定學年度"
        src = {"question": "（從問題判斷）", "ui": "（側欄）", "reply": "（你的回覆）"}.get(meta.get("year_source"), "")
        st.caption(f"依 {who}{src}・{meta['program']} 的規定回答・{meta.get('secs', 0):.0f} 秒")
    if meta.get("sources"):
        with st.expander(f"參考條文（{len(meta['sources'])}）"):
            for s in meta["sources"]:
                art = f"｜{s['article_no']}" if s.get("article_no") else ""
                line = f"**[{s['n']}]** [{s['title']}]({s['url']}){art}　_{s['office']}_"
                st.markdown(line)
                for pt in s.get("parts", []):
                    body = pt["text"] if len(pt["text"]) <= 160 else pt["text"][:160] + "…"
                    label = f"**{pt['label']}**　" if pt["label"] else ""
                    cid = f"　`{pt['chunk_id']}`" if show_debug else ""
                    st.caption(f"{label}{body.replace(chr(10), '　')}{cid}")


def run(question: str, year: int | None, from_reply: bool = False) -> dict:
    pipeline = load_pipeline()
    t0 = time.time()
    try:
        a = pipeline.answer(question, ui_year=year, ui_program=ui_program)
    except RuntimeError as e:          # Ollama 沒開、沒有 key 等
        return {"content": f"⚠️ {e}", "meta": {}}
    year_source = "reply" if from_reply and a.year_source == "ui" else a.year_source
    meta = {"status": a.status, "year": a.year, "program": a.program, "year_source": year_source,
            "sources": a.sources, "secs": time.time() - t0}
    if a.status == "ask_year":
        st.session_state.pending = question
    else:
        st.session_state.pending = None
    return {"content": a.text, "meta": meta}


# ---------------------------------------------------------------- 主畫面
st.title("🎓 碩士班畢業規定查詢")
st.caption("中興大學資訊管理學系碩士班｜畢業學分、外系學分、英檢、論文口試、離校手續")

# chat_input 固定在畫面底部，先取值不影響版面；這樣才能在「已經有問題送出」時就不畫範例按鈕
prompt = st.chat_input("輸入你的問題，例如：我113入學，畢業要修幾學分？")
prompt = prompt or st.session_state.pop("queued", None)

if not st.session_state.messages and not prompt:
    st.info("左側先選**入學學年度**，問題會更準。也可以直接在問題裡寫，例如「我112入學，…」。", icon="💡")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.queued = ex
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(tidy(m["content"]) if m["role"] == "assistant" else m["content"])
        render_meta(m.get("meta", {}))

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 上一輪被反問學年度，這一輪只回了年份 → 用原問題重答
    question, year, from_reply = prompt, ui_year, False
    if st.session_state.pending:
        m = RE_BARE_YEAR.match(prompt)
        given = int(m.group(1)) if m else extract(prompt).year
        if given:
            question, year, from_reply = st.session_state.pending, given, True

    with st.chat_message("assistant"):
        with st.spinner("查詢規章中…（本機模型約 20 秒）"):
            reply = run(question, year, from_reply)
        st.markdown(tidy(reply["content"]))
        render_meta(reply["meta"])
    st.session_state.messages.append({"role": "assistant", **reply})
