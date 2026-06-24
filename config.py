"""配置 — 所有可调参数集中在这里"""
from pathlib import Path

# 模型
LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"

# 评论分块
MIN_CHUNK_LENGTH = 5        # 最短观点长度（过滤掉"好""不错"这种无意义词）
MAX_CHUNKS_PER_REVIEW = 10  # 一条评论最多切几条观点

# 检索
SEARCH_K = 5                # 返回 Top-K

# 路径
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "reviews.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
