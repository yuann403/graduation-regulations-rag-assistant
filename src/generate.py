"""Prompt 組裝 + 呼叫 LLM（預設本機 Ollama qwen2.5:7b；config.LLM_PROVIDER="claude" 改用 Anthropic API）。

System prompt 以 README 5.4 為底，加上 Phase 1–4 實際遇到、需要明講的規則：
    - 系規比校規嚴時以系規為準（校選課辦法大學部 6 學分 vs 系明細表 3 學分）
    - 不同來源講同一件事但數字不同時，兩個都講清楚（系 30 天前 vs 校 20 天前）
    - 同一份文件前後矛盾要指出（114 明細表五、寫 9 學分，六、備註寫 6 學分）
    - 問句帶錯誤前提要先糾正
    - 規章只寫到哪就答到哪，不自行延伸（「配偶不得擔任」不等於「女朋友不得擔任」）

Claude 路徑依 claude-api skill：claude-opus-5-5、adaptive thinking（此模型恆開）、
effort 明設、fallbacks="default"（安全分類器誤拒時由伺服器改用其他模型重跑）。
免責聲明由 generate() 附加，不交給模型（7B 會把「最後一行固定寫：」照抄）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from src.config import ENV_FILE, LLM_EFFORT, LLM_MODEL

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """你是國立中興大學資訊管理學系碩士班的行政法規查詢助理，幫研究生查畢業相關規定。一律使用繁體中文（台灣用語）回答。

## 回答依據
- 只能依據 <context> 裡的條文回答。context 沒有的資訊，明確說「現有規章未涵蓋此項」，並建議洽詢最相關的處室（例如住宿找學務處、學籍找教務處註冊組、系上事務找資管系系辦）；不要用一般常識或其他學校的規定補。
- context 裡只有相近但不是使用者問的規定時（例如問外系學分，context 只有校際選課），要說明 context 沒有直接規定，不要拿相近的條文當答案。
- 每項事實後標註來源編號，例如 [1][3]。
- 規章寫到哪就答到哪。條文只列了「配偶或三等親」，就不要推論到條文沒提到的關係；條文沒寫金額，就不要估金額。
- 表單名稱、表單代碼、期限、處室名稱只能照 context 寫。使用者問到 context 裡沒有的表單（例如一個沒出現過的代碼），直接說現有規章中沒有這份表單。

## 學年度與學制
- 使用者的入學學年度與學制會寫在 <user_profile>。context 中標示的【適用】範圍與使用者不符的條文，不得使用。
- 同一主題各學年度規定不同時，只回答適用於使用者的版本，並主動提一句「此規定依入學學年度而異」。
- 條文自己區分新舊制（例如「105學年度入學者5小時、106學年度以後6小時」）時，挑出適用於使用者的那一句。
- 碩士班與碩士在職專班規定不同，不可互相套用。

## 多個來源說法不同時
- 系級規定比校級規定嚴格時（例如校級上限 6 學分、系級限 3 學分），以系級為準，並說明校級規定是全校上限。
- 系和校各訂期限時（例如系要求 30 天前、校要求 20 天前），兩個都講，並說明哪一個先到、實際上要照哪個辦。
- 同一份文件前後不一致時，指出不一致之處，以該項目本身的規定為準，並建議向主責單位確認。

## 錯誤前提
- 使用者的問題若帶有錯誤前提（例如以為口試後再做原創性比對即可），先明確糾正，再說明正確做法。

## 格式
依序使用以下四個段落，每段只出現一次（沒有內容的段落寫「無」）：
## 結論
## 依據
（每條依據一行，以「- 」開頭：先寫文件名與條次，再用「」引用條文中支持結論的原句，最後標來源編號。
  範例：- 碩士論文考試流程與表單 一、：「論文初稿需經指導教授審閱認可，始可提出論文考試申請」[2]
  不要寫「根據來源1」這種只有編號、沒有內容的句子）
## 需繳交文件
## 辦理處室與期限"""

# 免責聲明由程式附加，不交給模型：本機小模型會把「最後一行固定寫：」整句照抄
DISCLAIMER = "本回答僅供參考，請以各處室公告與承辦人說明為準。"


RE_PART = re.compile(r"（(?:續 )?(\d+)/(\d+)）$")


def section_key(c: dict) -> tuple[str, str]:
    return c["doc_id"], RE_PART.sub("", c.get("section_path") or c.get("article_no") or "").strip()


def part_label(c: dict) -> str:
    """「一、…（續 2/4）」→「第2/4段」；沒被切開的條文回傳空字串。"""
    m = RE_PART.search(c.get("section_path") or "")
    return f"第{m.group(1)}/{m.group(2)}段" if m else ""


def group_hits(hits: list) -> list[dict]:
    """同一份文件、同一節被切成好幾塊時，給模型時合併成一個來源（small-to-big）。

    檢索與評估仍以 chunk 為單位；合併只發生在檢索之後、且只合併有被檢索到的塊。
    合併後的內文標出（第k/n段），介面也逐塊列出，出處仍精確到塊。
    群組順序依最好的名次；群組內依原文順序。
    """
    groups: dict[tuple, dict] = {}
    for h in hits:
        c = h.chunk
        key = section_key(c)
        g = groups.setdefault(key, {"chunk": c, "section": key[1], "parts": []})
        g["parts"].append(c)
    out = []
    for g in groups.values():
        parts = []
        for c in sorted(g["parts"], key=lambda c: c["chunk_id"]):
            body = (c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"]).strip()
            parts.append({"chunk_id": c["chunk_id"], "label": part_label(c), "text": body})
        lines = [g["section"]]
        for pt in parts:
            lines.append(f"（{pt['label']}）\n{pt['text']}" if pt["label"] else pt["text"])
        out.append({**g["chunk"], "section": g["section"], "text": "\n".join(lines), "parts": parts,
                    "chunk_ids": [pt["chunk_id"] for pt in parts]})
    return out


def format_context(hits: list) -> str:
    """把檢索結果排成編號的 context。每塊附上文件名、適用範圍、主責、條次與來源網址。"""
    blocks = []
    for i, c in enumerate(group_hits(hits), 1):
        yf, yt = c["applicable_year_from"], c["applicable_year_to"]
        if yf is None and yt is None:
            scope = "不分入學學年度"
        elif yt is None:
            scope = f"{yf} 學年度(含)以後入學"
        elif yf is None:
            scope = f"{yt} 學年度(含)以前入學"
        else:
            scope = f"{yf}–{yt} 學年度入學" if yf != yt else f"{yf} 學年度入學"
        rev = f"（{c['revision_date']} 修訂）" if c.get("revision_date") and "-" not in c["revision_date"] else ""
        blocks.append(
            f'<source id="{i}">\n'
            f"【文件】{c['title']}{rev}｜{c['level']}\n"
            f"【適用】{scope}之{'、'.join(c['applicable_program'])}\n"
            f"【主責】{c['responsible_office']}\n"
            f"【條次】{c['section']}\n"
            f"【網址】{c['source_url']}\n"
            f"---\n{c['text']}\n</source>"
        )
    return "<context>\n" + "\n\n".join(blocks) + "\n</context>"


def format_user(question: str, year: int | None, program: str, hits: list) -> str:
    profile = f"入學學年度：{year if year is not None else '未提供'}\n學制：{program}"
    return f"{format_context(hits)}\n\n<user_profile>\n{profile}\n</user_profile>\n\n<question>\n{question}\n</question>"


@dataclass
class Generation:
    text: str
    stop_reason: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    fell_back: bool = False
    raw_blocks: list = field(default_factory=list, repr=False)


_client = None


def get_client():
    """延遲建立 client，讓 --dry 等不呼叫 API 的路徑不需要 key。"""
    global _client
    if _client is None:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE)
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(f"找不到 API key：請在 {ENV_FILE} 寫入 ANTHROPIC_API_KEY=...（參考 .env.example）")
        _client = anthropic.Anthropic()
    return _client


def call_claude(system: str, user: str, *, model: str = LLM_MODEL, effort: str = LLM_EFFORT,
                max_tokens: int = 16000, output_format: dict | None = None) -> Generation:
    client = get_client()
    output_config: dict = {"effort": effort}
    if output_format:
        output_config["format"] = output_format
    resp = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        # system prompt 固定不變 → 標 cache_control，評估時 50 題共用同一段前綴
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config=output_config,
    )
    fell_back = any(getattr(it, "type", "") == "fallback_message" for it in (resp.usage.iterations or []))
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        text = f"（模型拒絕回答：{cat}）"
    else:
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return Generation(
        text=text, stop_reason=resp.stop_reason, model=resp.model,
        input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
        cache_read_tokens=resp.usage.cache_read_input_tokens or 0, fell_back=fell_back,
    )


def call_ollama(system: str, user: str, *, model: str | None = None,
                output_format: dict | None = None) -> Generation:
    """本機 Ollama（/api/chat）。不需要 key。"""
    import requests

    from src.config import (OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_REPEAT_PENALTY,
                            OLLAMA_TIMEOUT, OLLAMA_URL)

    body = {
        "model": model or OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0,
                    "num_predict": OLLAMA_NUM_PREDICT, "repeat_penalty": OLLAMA_REPEAT_PENALTY},
    }
    if output_format:
        body["format"] = output_format["schema"]       # Ollama 直接吃 JSON schema
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=OLLAMA_TIMEOUT)
    except requests.ConnectionError as e:
        raise RuntimeError("連不到 Ollama：請確認 Ollama 有在執行（ollama list）") from e
    except requests.Timeout as e:
        raise RuntimeError(f"Ollama 超過 {OLLAMA_TIMEOUT} 秒沒有回應") from e
    r.raise_for_status()
    d = r.json()
    if d.get("prompt_eval_count", 0) >= OLLAMA_NUM_CTX:
        print(f"[warn] prompt 用滿 num_ctx={OLLAMA_NUM_CTX}，內容可能被截斷")
    text = d["message"]["content"].strip()
    if d.get("done_reason") == "length":
        print(f"[warn] 輸出達到上限 {OLLAMA_NUM_PREDICT} tokens（可能陷入重複），已截斷")
    return Generation(
        text=text, stop_reason=d.get("done_reason", ""),
        model=d.get("model", body["model"]),
        input_tokens=d.get("prompt_eval_count", 0), output_tokens=d.get("eval_count", 0),
    )


def call_llm(system: str, user: str, *, provider: str | None = None, **kw) -> Generation:
    from src.config import LLM_PROVIDER

    if (provider or LLM_PROVIDER) == "ollama":
        kw.pop("effort", None)
        kw.pop("max_tokens", None)
        if kw.get("model", "").startswith("claude"):
            kw.pop("model")
        return call_ollama(system, user, **kw)
    return call_claude(system, user, **kw)


def generate(question: str, year: int | None, program: str, hits: list) -> Generation:
    g = call_llm(SYSTEM_PROMPT, format_user(question, year, program, hits))
    if g.stop_reason != "refusal":
        g.text = f"{tidy_citations(g.text).rstrip()}\n\n{DISCLAIMER}"
    return g


def tidy_citations(text: str) -> str:
    """7B 常寫「【依據1】根據【來源1】，…」：統一成 [1]，並讓每條依據自成一行。"""
    text = text.replace("辯理", "辦理")                    # 7B 偶發錯字
    text = re.sub(r"(?:\n\s*無\s*){2,}", "\n\n無\n", text)  # 7B 重複迴圈留下的一長串「無」只留一個
    text = re.sub(r"【?來源\s*(\d+)】?", r"[\1]", text)
    text = re.sub(r"\s*【依據\s*\d+】\s*", "\n- ", text)
    text = re.sub(r"(?:根據|依據)\s*(\[\d+\])\s*[，,]\s*", r"\1 ", text)   # 「根據[1]，您需要…」→「[1] 您需要…」
    return text
