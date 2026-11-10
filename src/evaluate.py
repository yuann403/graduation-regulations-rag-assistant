"""跑評估集 eval/questions.jsonl。

用法:
    python -m src.evaluate                 # 檢索指標（Phase 4）
    python -m src.evaluate --check         # 只驗證題目格式與 gold chunk_id 是否存在，不載模型
    python -m src.evaluate -k 10

目前只算「檢索層」指標；生成層指標（要點涵蓋率、窗口正確率、適當拒答率、幻覺數，
README 6.5）要等 Phase 5 有 pipeline.answer() 之後才能算，這裡先把欄位留好。

檢索指標只算 gold_chunk_ids 非空的題目：
    hit@k      top k 至少有一個 gold 或 acceptable   （README 的 Recall@5）
    recall@k   gold 有幾成進了 top k                 （多跳題要全部撈到才算完整；不含 acceptable）
    mrr        第一個 gold 或 acceptable 的名次倒數
    leak       top k 中違反學年度/學制過濾的 chunk 數（應永遠是 0）
    redundant  top k 中與前面某塊文字完全相同的塊數   （不分學年度時，13 版明細表的同一項會佔滿名額）

acceptable_chunk_ids：內容也能正確回答、但不是「必須撈到」的來源。例：問修業年限，
13 個學年度明細表的「一、修業年限」都寫 4 年，任一版都算答對。
ask_year / refuse / deny_form 類題目沒有 gold chunk，檢索層不評，留給生成層。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.config import CHUNKS_JSONL, ROOT, TOP_K

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"

REQUIRED = {"qid", "question", "type", "source", "slots", "gold_chunk_ids", "gold_answer_points",
            "gold_office", "answerable", "expected_behavior"}
TYPES = {"T1", "T2", "T3", "T4", "T5", "T6"}
BEHAVIORS = {"answer", "ask_year", "correct_premise", "refuse", "deny_form"}


def load_questions() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check(qs: list[dict], chunks: dict[str, dict]) -> list[str]:
    """題目本身的錯：缺欄位、gold 不存在、gold 和題目的學年度/學制矛盾。"""
    from src.retrieve import passes

    errs = []
    seen = set()
    for q in qs:
        qid = q.get("qid", "?")
        if qid in seen:
            errs.append(f"{qid}: qid 重複")
        seen.add(qid)
        if miss := REQUIRED - q.keys():
            errs.append(f"{qid}: 缺欄位 {sorted(miss)}")
            continue
        if q["type"] not in TYPES:
            errs.append(f"{qid}: type={q['type']}")
        if q["expected_behavior"] not in BEHAVIORS:
            errs.append(f"{qid}: expected_behavior={q['expected_behavior']}")
        if q["expected_behavior"] == "answer" and not q["gold_chunk_ids"]:
            errs.append(f"{qid}: 應回答的題目沒有 gold_chunk_ids")
        year, prog = q["slots"].get("year"), q["slots"].get("program")
        for cid in q["gold_chunk_ids"] + q.get("acceptable_chunk_ids", []):
            if cid not in chunks:
                errs.append(f"{qid}: gold {cid} 不存在（chunk_id 可能因重新切塊而改變）")
            elif not passes(chunks[cid], year, prog):
                # gold 本身會被過濾掉 = 題目或 metadata 有一邊錯了
                errs.append(f"{qid}: gold {cid} 不符題目的 year={year} program={prog}，會被過濾掉")
    return errs


def summarize(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        n = len(rs)
        out[g] = {"n": n,
                  "hit": round(sum(r["hit"] for r in rs) / n, 3),
                  "recall": round(sum(r["recall"] for r in rs) / n, 3),
                  "mrr": round(sum(r["mrr"] for r in rs) / n, 3)}
    return out


# ---------------------------------------------------------------- 生成層（Phase 5）
JUDGE_SYSTEM = """你是評估員，負責評分一個「大學行政法規問答系統」的回答。請嚴格依照給你的標準答案要點判斷，不要用你自己的知識補充或放寬。

評分項目：
1. points：逐一判斷每個標準答案要點是否被回答涵蓋。意思相同即可，不必逐字；數字、期限、單位名稱必須正確才算涵蓋。
2. behavior_ok：回答是否做到 expected_behavior：
   - answer：直接回答問題
   - correct_premise：明確指出使用者前提錯誤，並給出正確規定
   - refuse：表明現有規章未涵蓋，並轉介處室；不得編造答案
   - deny_form：表明規章中沒有這份表單，不得編造用途或繳交處
3. office_ok：回答中提到的辦理處室是否與 gold_office 相符（gold_office 出現在回答中、或回答指向的是同一單位即算）。回答完全沒提處室則為 false。
4. hallucination：回答中是否出現「標準答案要點與檢索到的 context 都沒有支持」的具體事實（數字、期限、表單名稱、處室）。一般性建議（如「建議洽詢系辦」）不算。若為 true，在 hallucination_detail 寫出是哪一句。"""

JUDGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "points": {"type": "array", "items": {
                "type": "object",
                "properties": {"point": {"type": "string"}, "covered": {"type": "boolean"}},
                "required": ["point", "covered"], "additionalProperties": False}},
            "behavior_ok": {"type": "boolean"},
            "office_ok": {"type": "boolean"},
            "hallucination": {"type": "boolean"},
            "hallucination_detail": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["points", "behavior_ok", "office_ok", "hallucination", "hallucination_detail", "reason"],
        "additionalProperties": False,
    },
}

PRICE_IN, PRICE_OUT = 4.0, 20.0     # claude-opus-5-5，USD / 1M tokens（快取讀取另計，這裡從寬估）


def resolve_judge(requested: str | None) -> str:
    """claude / ollama / none。要 Claude 但沒有 key → 降為 none（只產出人工覆核檔），不中途崩潰。"""
    import os

    from dotenv import load_dotenv

    from src.config import ENV_FILE, JUDGE_PROVIDER

    mode = requested or JUDGE_PROVIDER
    if mode == "claude":
        load_dotenv(ENV_FILE)
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            print("[judge] 沒有 ANTHROPIC_API_KEY → 不做 LLM 評分，改輸出人工覆核檔（--judge none）")
            return "none"
    if mode == "ollama":
        print("[judge] 用本機 Ollama 當評分員：7B 模型評分不可靠，數字只能當參考，務必人工抽查")
    return mode


def judge(q: dict, ans, context: str, provider: str) -> dict:
    from src.config import JUDGE_EFFORT, JUDGE_MODEL
    from src.generate import call_llm

    user = (
        f"<question>{q['question']}</question>\n"
        f"<user_profile>入學學年度：{q['slots'].get('year')}；學制：{q['slots'].get('program')}</user_profile>\n"
        f"<expected_behavior>{q['expected_behavior']}</expected_behavior>\n"
        f"<gold_answer_points>\n" + "\n".join(f"- {p}" for p in q["gold_answer_points"]) + "\n</gold_answer_points>\n"
        f"<gold_office>{q['gold_office']}</gold_office>\n"
        f"{context}\n"
        f"<system_answer>\n{ans.text}\n</system_answer>"
    )
    g = call_llm(JUDGE_SYSTEM, user, provider=provider, model=JUDGE_MODEL, effort=JUDGE_EFFORT,
                 output_format=JUDGE_SCHEMA)
    empty = {"points": [], "behavior_ok": False, "office_ok": False, "hallucination": False,
             "hallucination_detail": ""}
    if g.stop_reason == "refusal":
        out = {**empty, "reason": g.text}
    else:
        try:
            out = json.loads(g.text)
        except json.JSONDecodeError:            # 本機模型偶爾吐出不合法的 JSON
            out = {**empty, "reason": f"評分員輸出無法解析：{g.text[:120]}"}
    out["_usage"] = {"input_tokens": g.input_tokens, "output_tokens": g.output_tokens}
    return out


def run_generation(qs: list[dict], tag: str, judge_req: str | None = None) -> int:
    """端到端評估：每題跑 pipeline.answer()，ask_year 題直接判斷，其餘交給評分員（或留給人工）。"""
    from src.config import LLM_MODEL, LLM_PROVIDER, OLLAMA_MODEL
    from src.generate import format_context
    from src.pipeline import answer, get_retriever

    judge_mode = resolve_judge(judge_req)
    gen_model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else LLM_MODEL
    print(f"[gen] 生成：{LLM_PROVIDER} / {gen_model}　評分：{judge_mode}")
    rows, tok_in, tok_out = [], 0, 0
    claude_in = claude_out = 0           # 只有 Claude 的 token 要算錢
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        year, prog = q["slots"].get("year"), q["slots"].get("program")
        try:
            a = answer(q["question"], ui_year=year, ui_program=prog)
        except RuntimeError as e:
            # 單題失敗（逾時等）不讓整輪 50 題白跑：記下錯誤，繼續下一題
            print(f"[{i:2d}/{len(qs)}] {q['qid']} 失敗：{e}")
            rows.append({"qid": q["qid"], "type": q["type"], "source": q["source"], "question": q["question"],
                         "expected_behavior": q["expected_behavior"], "status": "error",
                         "year": year, "program": prog, "answer": f"（錯誤：{e}）",
                         "sources": [], "usage": {}, "gold_answer_points": q["gold_answer_points"],
                         "gold_office": q["gold_office"], "behavior_ok": False, "points": [],
                         "point_coverage": None, "office_ok": None, "hallucination": None,
                         "judge_reason": f"執行錯誤：{e}"})
            continue
        row = {"qid": q["qid"], "type": q["type"], "source": q["source"], "question": q["question"],
               "expected_behavior": q["expected_behavior"], "status": a.status,
               "year": a.year, "program": a.program, "answer": a.text,
               "sources": [cid for s in a.sources for cid in s["chunk_ids"]], "usage": a.usage}
        tok_in += a.usage.get("input_tokens", 0)
        tok_out += a.usage.get("output_tokens", 0)
        if LLM_PROVIDER == "claude":
            claude_in += a.usage.get("input_tokens", 0)
            claude_out += a.usage.get("output_tokens", 0)
        row["gold_answer_points"] = q["gold_answer_points"]
        row["gold_office"] = q["gold_office"]

        if q["expected_behavior"] == "ask_year" or a.status == "ask_year":
            # 反問與否是規則判斷，不需要 LLM 評
            ok = (q["expected_behavior"] == "ask_year") == (a.status == "ask_year")
            row.update(behavior_ok=ok, points=[], point_coverage=None, office_ok=None,
                       hallucination=False, judge_reason="反問學年度" if a.status == "ask_year" else "未反問")
        elif judge_mode == "none":
            row.update(behavior_ok=None, points=[], point_coverage=None, office_ok=None,
                       hallucination=None, judge_reason="未評分（人工覆核）")
        else:
            hits = get_retriever().search(q["question"], a.year, a.program)
            j = judge(q, a, format_context(hits), judge_mode)
            tok_in += j["_usage"]["input_tokens"]
            tok_out += j["_usage"]["output_tokens"]
            if judge_mode == "claude":
                claude_in += j["_usage"]["input_tokens"]
                claude_out += j["_usage"]["output_tokens"]
            pts = j["points"]
            row.update(behavior_ok=j["behavior_ok"], points=pts,
                       point_coverage=round(sum(p["covered"] for p in pts) / len(pts), 3) if pts else None,
                       office_ok=j["office_ok"], hallucination=j["hallucination"],
                       hallucination_detail=j["hallucination_detail"], judge_reason=j["reason"])
        rows.append(row)
        cov = "-" if row["point_coverage"] is None else f"{row['point_coverage']:.2f}"
        mark = lambda v: "-" if v is None else ("✓" if v else "✗")  # noqa: E731
        print(f"[{i:2d}/{len(qs)}] {q['qid']} {q['expected_behavior']:15s} status={a.status:8s} "
              f"behavior={mark(row['behavior_ok'])} 要點={cov} 處室={mark(row['office_ok'])} "
              f"{'幻覺!' if row['hallucination'] else ''} {time.time() - t0:.0f}s", flush=True)

    # ---- 指標（README 6.5）；未評分（None）的題目不列入分母 ----
    answerable = [r for r in rows if r["expected_behavior"] in ("answer", "correct_premise")]
    covs = [r["point_coverage"] for r in answerable if r["point_coverage"] is not None]
    offices = [r["office_ok"] for r in answerable if r["office_ok"] is not None]
    refusal = [r for r in rows if r["expected_behavior"] in ("refuse", "deny_form") and r["behavior_ok"] is not None]
    judged = [r for r in rows if r["behavior_ok"] is not None]
    by_beh = defaultdict(list)
    for r in judged:
        by_beh[r["expected_behavior"]].append(r["behavior_ok"])
    gen = {
        "n": len(rows),
        "生成": f"{LLM_PROVIDER} / {gen_model}",
        "評分": judge_mode,
        "已評分題數": len(judged),
        "要點涵蓋率": round(sum(covs) / len(covs), 3) if covs else None,
        "窗口正確率": round(sum(offices) / len(offices), 3) if offices else None,
        "適當拒答率": round(sum(r["behavior_ok"] for r in refusal) / len(refusal), 3) if refusal else None,
        "幻覺數": sum(bool(r["hallucination"]) for r in rows),
        "行為正確率": round(sum(r["behavior_ok"] for r in judged) / len(judged), 3) if judged else None,
        "行為正確率_by": {b: f"{sum(v)}/{len(v)}" for b, v in sorted(by_beh.items())},
        "tokens": {"input": tok_in, "output": tok_out},
        "Claude花費_USD": round(claude_in / 1e6 * PRICE_IN + claude_out / 1e6 * PRICE_OUT, 2),
        "秒": round(time.time() - t0),
        "秒_每題": round((time.time() - t0) / max(len(rows), 1), 1),
    }
    result = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "retrieval": None,
              "generation": gen, "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-gen{'-' + tag if tag else ''}"
    out = RESULTS / f"{stem}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    review = RESULTS / f"{stem}-review.md"
    review.write_text(review_markdown(rows, gen), "utf-8")

    print("\n生成層評估")
    for k, v in gen.items():
        print(f"  {k}: {v}")
    bad = [r for r in judged if not r["behavior_ok"] or r["hallucination"]
           or (r["point_coverage"] is not None and r["point_coverage"] < 0.5)]
    if judged:
        print(f"\n  需要看的題目（{len(bad)}）:")
        for r in bad:
            why = r.get("hallucination_detail") if r["hallucination"] else r["judge_reason"]
            print(f"    {r['qid']} {r['question'][:26]}  → {(why or '')[:80]}")
    print(f"\n→ {out.relative_to(ROOT)}\n→ {review.relative_to(ROOT)}（逐題：問題／標準要點／系統回答，人工覆核用）")
    return 0


def review_markdown(rows: list[dict], gen: dict) -> str:
    """人工覆核用：每題並列標準要點與系統回答，勾選框讓人逐點打勾。"""
    out = [f"# 生成層人工覆核（{gen['生成']}，評分：{gen['評分']}）", "",
           "每題對照「標準要點」逐點打勾；`處室`、`幻覺`、`行為` 同樣勾選。", ""]
    for r in rows:
        out += [f"## {r['qid']}　{r['question']}", "",
                f"- 預期行為：`{r['expected_behavior']}`　系統：`{r['status']}`　"
                f"學年度 {r['year']}／{r['program']}",
                f"- 檢索到：{', '.join(r['sources']) or '（無）'}", "", "**標準要點**", ""]
        pts = {p["point"]: p["covered"] for p in r.get("points") or []}
        for p in r["gold_answer_points"]:
            box = "x" if pts.get(p) else " "
            out.append(f"- [{box}] {p}")
        out += [f"- [{'x' if r['office_ok'] else ' '}] 處室：{r['gold_office']}",
                f"- [{'x' if r['behavior_ok'] else ' '}] 行為正確", f"- [ ] 無幻覺", "",
                "**系統回答**", "", "> " + r["answer"].replace("\n", "\n> "), ""]
        if r.get("judge_reason") and r["judge_reason"] != "未評分（人工覆核）":
            out += [f"*評分理由：{r['judge_reason']}*", ""]
        out.append("---\n")
    return "\n".join(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=TOP_K)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tag", default="", help="寫進結果檔名，方便比較（例：baseline、rerank）")
    ap.add_argument("--gen", action="store_true", help="端到端：跑 pipeline.answer() + 評分")
    ap.add_argument("--judge", choices=["claude", "ollama", "none"],
                    help="評分員；預設 config.JUDGE_PROVIDER，要 Claude 但沒 key 會自動降為 none")
    ap.add_argument("--only", default="", help="只跑這些 qid，逗號分隔（例：Q01,Q36,Q42），先小量試")
    args = ap.parse_args()

    qs = load_questions()
    with CHUNKS_JSONL.open(encoding="utf-8") as fh:
        chunks = {c["chunk_id"]: c for c in map(json.loads, fh)}

    errs = check(qs, chunks)
    counts = defaultdict(lambda: defaultdict(int))
    for q in qs:
        counts["type"][q["type"]] += 1
        counts["source"][q["source"]] += 1
        counts["behavior"][q["expected_behavior"]] += 1
    print(f"題目 {len(qs)} 題")
    for k, v in counts.items():
        print(f"  {k:9s}", dict(sorted(v.items())))
    if errs:
        print(f"\n題目檢查：{len(errs)} 個問題")
        for e in errs:
            print("  ✗", e)
        return 1
    print("題目檢查：OK（gold chunk 全部存在，且都通過題目自己的學年度/學制過濾）")
    if args.check:
        return 0
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        qs = [q for q in qs if q["qid"] in keep]
    if args.gen:
        return run_generation(qs, args.tag, args.judge)

    from src.retrieve import Retriever, passes

    r = Retriever()
    rows = []
    t0 = time.time()
    for q in qs:
        gold = q["gold_chunk_ids"]
        if not gold:
            continue
        year, prog = q["slots"].get("year"), q["slots"].get("program")
        hits = r.search(q["question"], year, prog, args.k)
        ids = [h.chunk_id for h in hits]
        ok = gold + q.get("acceptable_chunk_ids", [])
        ranks = [ids.index(g) + 1 for g in ok if g in ids]
        hashes = [h.chunk["text_hash"] for h in hits]
        rows.append({
            "qid": q["qid"], "type": q["type"], "source": q["source"],
            "question": q["question"], "year": year, "program": prog,
            "gold": gold, "retrieved": ids,
            "hit": int(bool(ranks)),
            "recall": round(sum(g in ids for g in gold) / len(gold), 3),
            "mrr": round(1 / min(ranks), 3) if ranks else 0.0,
            "leak": [h.chunk_id for h in hits if not passes(h.chunk, year, prog)],
            "redundant": len(hashes) - len(set(hashes)),
            "dense_bm25_ranks": {h.chunk_id: [h.dense_rank, h.bm25_rank] for h in hits},
        })
    secs = time.time() - t0

    n = len(rows)
    overall = {"n": n,
               f"hit@{args.k}": round(sum(x["hit"] for x in rows) / n, 3),
               f"recall@{args.k}": round(sum(x["recall"] for x in rows) / n, 3),
               "mrr": round(sum(x["mrr"] for x in rows) / n, 3),
               "leaks": sum(len(x["leak"]) for x in rows),
               "redundant_slots": sum(x["redundant"] for x in rows),
               "secs_per_query": round(secs / n, 2)}
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag, "k": args.k,
        "retrieval": {"overall": overall, "by_type": summarize(rows, "type"),
                      "by_source": summarize(rows, "source")},
        "generation": None,     # Phase 5：要點涵蓋率、窗口正確率、適當拒答率、幻覺數
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}{'-' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")

    print(f"\n檢索評估（{n} 題有 gold，k={args.k}）")
    print("  " + "  ".join(f"{k}={v}" for k, v in overall.items()))
    print("\n  依題型:")
    for g, s in result["retrieval"]["by_type"].items():
        print(f"    {g}  n={s['n']:2d}  hit={s['hit']:.2f}  recall={s['recall']:.2f}  mrr={s['mrr']:.2f}")
    print("\n  依來源:")
    for g, s in result["retrieval"]["by_source"].items():
        print(f"    {g:16s} n={s['n']:2d}  hit={s['hit']:.2f}  recall={s['recall']:.2f}  mrr={s['mrr']:.2f}")
    miss = [x for x in rows if x["recall"] < 1]
    print(f"\n  沒撈全的題目（{len(miss)}）:")
    for x in miss:
        lost = [g for g in x["gold"] if g not in x["retrieved"]]
        print(f"    {x['qid']} recall={x['recall']:.2f}  {x['question'][:28]}  缺 {', '.join(lost)}")
    print(f"\n→ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
