"""路徑與參數集中管理。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SOURCES_CSV = DATA / "sources.csv"
RAW_DIR = DATA / "raw"
MANUAL_DIR = DATA / "manual"            # 掃描檔的人工轉錄稿，優先於 raw
PROCESSED_DIR = DATA / "processed"      # ingest 產出：每份文件一個正規化 Markdown
CHUNKS_JSONL = DATA / "chunks.jsonl"
REVIEW_DIR = DATA / "review"

# 人工覆核結果：doc 層級的 metadata 修正，與 chunk 層級的學年度覆寫
DOC_OVERRIDES_CSV = DATA / "doc_overrides.csv"
CHUNK_OVERRIDES_CSV = DATA / "chunk_overrides.csv"

# chunk 長度：單條超過這個字數才二次切
MAX_CHARS = 800

# ---- Phase 3：索引與檢索 ----
CHROMA_DIR = DATA / "chroma"
BM25_PKL = DATA / "bm25.pkl"
JIEBA_DICT = DATA / "jieba_userdict.txt"
COLLECTION = "nchu_mis"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DEVICE = "cpu"            # 有 CUDA 版 torch 時改 "cuda"

# Chroma 不收 None：不分學年度存成 0 / 9999，filter 就只是 from <= y <= to
YEAR_MIN, YEAR_MAX = 0, 9999

N_CANDIDATES = 20               # dense / BM25 各取幾筆進 RRF
RRF_K = 60
TOP_K = 5

# ---- Phase 5：生成層 ----
# API key 放在專案根目錄的 .env（ANTHROPIC_API_KEY=...），不要寫進程式或 commit
ENV_FILE = ROOT / ".env"
# "ollama"：本機開源模型（免費、不需 key）；"claude"：Anthropic API（需 .env 的 key）
LLM_PROVIDER = "ollama"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_NUM_CTX = 8192           # 預設 4096 不夠放 5 段條文，會被默默截掉
OLLAMA_NUM_PREDICT = 1024       # 輸出上限：正常回答 200–550 tokens；7B 偶爾無限重複「無」，沒上限會卡到逾時
OLLAMA_REPEAT_PENALTY = 1.1     # qwen2.5 預設 1.05；Q38 在 1.05 下陷入重複迴圈，1.1 正常結束
OLLAMA_TIMEOUT = 300            # 秒；有輸出上限後，一題不該超過 2 分鐘
# 評分員：本機 7B 當評分員不可靠，預設仍用 Claude；沒有 key 就只能人工看
JUDGE_PROVIDER = "claude"
LLM_MODEL = "claude-opus-5-5"
LLM_EFFORT = "medium"           # Opus 5.5 的預設也是 medium；low 較省、high 較仔細
JUDGE_MODEL = "claude-opus-5-5"
JUDGE_EFFORT = "medium"
GEN_TOP_K = TOP_K               # 給 LLM 看幾個 chunk
