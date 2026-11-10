"""chunks.jsonl → ChromaDB（dense）＋ BM25（sparse）。

用法:
    python -m src.index

每次都整個重建（360 個 chunk，CPU 約幾分鐘）。chunks.jsonl 改了就重跑。

存進 Chroma 的 metadata：
    year_from / year_to   int；不分學年度存 0 / 9999（Chroma 不收 None）
    prog_master / prog_exec / prog_all   bool；學制用三個布林欄位，
        因為 Chroma 的 metadata 不收 list，用布林比逗號字串好過濾
    其餘欄位（doc_id、title、office、article_no、level…）只是帶著走，回答時用
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import time

import jieba

from src.config import (BM25_PKL, CHROMA_DIR, CHUNKS_JSONL, COLLECTION, EMBED_DEVICE, EMBED_MODEL,
                        JIEBA_DICT, YEAR_MAX, YEAR_MIN)

jieba.setLogLevel(60)
_jieba_ready = False
RE_KEEP = re.compile(r"[一-鿿A-Za-z0-9]")
STOP = set("的 之 及 與 或 並 於 為 以 得 應 須 者 等 其 該 本 經 後 前 由 依 如 則 而 但 且 亦 所 在 是 有 將 對 中".split())


def tokenize(text: str) -> list[str]:
    """BM25 用的斷詞：jieba 搜尋模式 + 自訂詞典，去標點、去單字虛詞，英文轉小寫。"""
    global _jieba_ready
    if not _jieba_ready:
        if JIEBA_DICT.exists():
            jieba.load_userdict(str(JIEBA_DICT))
        _jieba_ready = True
    toks = []
    for t in jieba.lcut_for_search(text):
        t = t.strip().lower()
        if t and RE_KEEP.search(t) and t not in STOP:
            toks.append(t)
    return toks


def load_chunks() -> list[dict]:
    with CHUNKS_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_metadata(c: dict) -> dict:
    prog = set(c["applicable_program"])
    yf, yt = c["applicable_year_from"], c["applicable_year_to"]
    return {
        "doc_id": c["doc_id"],
        "title": c["title"],
        "level": c["level"],
        "office": c["responsible_office"],
        "article_no": c["article_no"] or "",
        "source_url": c["source_url"],
        "revision_date": c["revision_date"] or "",
        "year_from": yf if yf is not None else YEAR_MIN,
        "year_to": yt if yt is not None else YEAR_MAX,
        "prog_master": "碩士班" in prog,
        "prog_exec": "碩專班" in prog,
        "prog_all": "全部" in prog,
        "topic": ",".join(c["topic"]),
        "forms": ",".join(c["forms"]),
    }


def build_dense(chunks: list[dict]) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    print(f"模型載入 {time.time() - t0:.0f}s")
    t0 = time.time()
    # embed 加了前綴的版本：使用者問「112入學…」時，前綴裡的適用範圍也能幫忙命中
    emb = model.encode([c["text_with_header"] for c in chunks], batch_size=8,
                       normalize_embeddings=True, show_progress_bar=True)
    print(f"embedding {len(chunks)} 筆 {time.time() - t0:.0f}s")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001  第一次建沒有舊的
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["chunk_id"] for c in chunks],
            embeddings=emb.tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=[to_metadata(c) for c in chunks])
    print(f"Chroma collection '{COLLECTION}': {col.count()} 筆 → {CHROMA_DIR}")


def build_sparse(chunks: list[dict]) -> None:
    # BM25 不用 text_with_header：每塊都有的「【文件】【適用】【主責】」會變成高頻雜訊；
    # 只加文件標題，讓「選課辦法」「學位考試細則」這類文件名能被搜到
    corpus = [tokenize(f"{c['title']}\n{c['text']}") for c in chunks]
    with BM25_PKL.open("wb") as fh:
        pickle.dump({"ids": [c["chunk_id"] for c in chunks], "corpus": corpus}, fh)
    avg = sum(map(len, corpus)) / len(corpus)
    print(f"BM25: {len(corpus)} 筆、平均 {avg:.0f} tokens → {BM25_PKL}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = load_chunks()
    build_sparse(chunks)
    build_dense(chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
