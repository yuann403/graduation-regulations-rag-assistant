"""Hybrid 檢索：metadata filter → dense (top 20) ＋ BM25 (top 20) → RRF 融合 → top k。

用法:
    python -m src.retrieve "外系學分可以抵幾學分" --year 114
    python -m src.retrieve "碩專班最低畢業學分" --year 113 --program 碩專班
    python -m src.retrieve --smoke              # 跑 eval/smoke_queries.jsonl，算 hit@k

學年度與學制是「硬過濾」：不符的 chunk 在 dense 與 BM25 兩邊都不會出現，
而不是撈出來再靠 LLM 判斷。year=None 時不做學年度過濾（是否該反問使用者，
由之後的 slots / pipeline 決定，不是檢索層的事）。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.config import (BM25_PKL, CHROMA_DIR, CHUNKS_JSONL, COLLECTION, EMBED_DEVICE, EMBED_MODEL,
                        N_CANDIDATES, ROOT, RRF_K, TOP_K, YEAR_MAX, YEAR_MIN)
from src.index import tokenize

PROGRAM_FLAG = {"碩士班": "prog_master", "碩專班": "prog_exec"}


@dataclass
class Hit:
    chunk_id: str
    score: float                       # RRF 分數
    dense_rank: int | None = None      # 1-based；None = 沒進該路的候選
    bm25_rank: int | None = None
    chunk: dict = field(default_factory=dict, repr=False)

    def label(self) -> str:
        c = self.chunk
        yf, yt = c["applicable_year_from"], c["applicable_year_to"]
        yr = "不分" if yf is None and yt is None else f"{yf or ''}–{yt or ''}"
        return f"{self.chunk_id:34s} [{yr:>9s}] {c['title'][:22]}｜{(c['article_no'] or '')[:12]}"


def passes(c: dict, year: int | None, program: str | None) -> bool:
    """與 Chroma where 等價的 Python 版，BM25 那一路用。"""
    if year is not None:
        yf = c["applicable_year_from"] if c["applicable_year_from"] is not None else YEAR_MIN
        yt = c["applicable_year_to"] if c["applicable_year_to"] is not None else YEAR_MAX
        if not (yf <= year <= yt):
            return False
    if program:
        progs = set(c["applicable_program"])
        if program not in progs and "全部" not in progs:
            return False
    return True


def build_where(year: int | None, program: str | None) -> dict | None:
    conds: list[dict] = []
    if year is not None:
        conds += [{"year_from": {"$lte": year}}, {"year_to": {"$gte": year}}]
    if program in PROGRAM_FLAG:
        conds.append({"$or": [{PROGRAM_FLAG[program]: True}, {"prog_all": True}]})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class Retriever:
    def __init__(self) -> None:
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        with CHUNKS_JSONL.open(encoding="utf-8") as fh:
            self.chunks = {c["chunk_id"]: c for c in map(json.loads, fh)}
        with BM25_PKL.open("rb") as fh:
            data = pickle.load(fh)
        self.bm25_ids: list[str] = data["ids"]
        self.bm25 = BM25Okapi(data["corpus"])
        self.col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
        self.model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    def dense(self, query: str, year: int | None, program: str | None, n: int) -> list[str]:
        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=emb, n_results=n, where=build_where(year, program))
        return res["ids"][0]

    def sparse(self, query: str, year: int | None, program: str | None, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            cid = self.bm25_ids[i]
            if passes(self.chunks[cid], year, program):
                out.append(cid)
                if len(out) == n:
                    break
        return out

    def search(self, query: str, year: int | None = None, program: str | None = "碩士班",
               k: int = TOP_K, n_candidates: int = N_CANDIDATES) -> list[Hit]:
        d = self.dense(query, year, program, n_candidates)
        s = self.sparse(query, year, program, n_candidates)
        hits: dict[str, Hit] = {}
        for rank, cid in enumerate(d, 1):
            h = hits.setdefault(cid, Hit(cid, 0.0, chunk=self.chunks[cid]))
            h.dense_rank, h.score = rank, h.score + 1 / (RRF_K + rank)
        for rank, cid in enumerate(s, 1):
            h = hits.setdefault(cid, Hit(cid, 0.0, chunk=self.chunks[cid]))
            h.bm25_rank, h.score = rank, h.score + 1 / (RRF_K + rank)
        return sorted(hits.values(), key=lambda h: -h.score)[:k]


def print_hits(hits: list[Hit]) -> None:
    for i, h in enumerate(hits, 1):
        dr = h.dense_rank or "-"
        br = h.bm25_rank or "-"
        print(f"  {i}. {h.label()}  (dense {dr}, bm25 {br})")


def smoke(r: Retriever, k: int) -> int:
    """跑手動設計的 query，看 gold chunk 有沒有進 top k（dense only / BM25 only / hybrid 三種都算）。"""
    path = ROOT / "eval" / "smoke_queries.jsonl"
    qs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    tally = {"dense": 0, "bm25": 0, "hybrid": 0}
    for q in qs:
        gold = set(q["gold"])
        year, prog = q.get("year"), q.get("program", "碩士班")
        hyb = r.search(q["q"], year, prog, k)
        res = {"dense": r.dense(q["q"], year, prog, k), "bm25": r.sparse(q["q"], year, prog, k),
               "hybrid": [h.chunk_id for h in hyb]}
        mark = {m: bool(gold & set(ids)) for m, ids in res.items()}
        for m in tally:
            tally[m] += mark[m]
        flag = "".join("✓" if mark[m] else "✗" for m in ("dense", "bm25", "hybrid"))
        print(f"\n[{flag}] {q['id']} ({year}, {prog}) {q['q']}\n  gold: {', '.join(sorted(gold))}")
        print_hits(hyb)
        # 過濾正確性：top k 裡不該有不符學年度/學制的 chunk
        leaked = [h.chunk_id for h in hyb if not passes(h.chunk, year, prog)]
        if leaked:
            print("  !! 過濾失效:", leaked)
    n = len(qs)
    print(f"\nhit@{k}  dense {tally['dense']}/{n}   bm25 {tally['bm25']}/{n}   hybrid {tally['hybrid']}/{n}"
          "   （標記順序：dense / bm25 / hybrid）")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--year", type=int)
    ap.add_argument("--program", default="碩士班")
    ap.add_argument("-k", type=int, default=TOP_K)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    r = Retriever()
    if args.smoke:
        return smoke(r, args.k)
    if not args.query:
        ap.error("需要 query 或 --smoke")
    print_hits(r.search(args.query, args.year, args.program, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
