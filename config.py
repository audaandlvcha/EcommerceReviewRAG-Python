"""配置 — 所有可调参数集中在这里"""
import os
from pathlib import Path
from langchain_openai import ChatOpenAI

# ── DeepSeek API（LLM） ──
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ── 本地 Embedding（替代 Ollama，无需额外服务） ──
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# 评论分块
MIN_CHUNK_LENGTH = 5        # 最短观点长度（过滤掉"好""不错"这种无意义词）
MAX_CHUNKS_PER_REVIEW = 10  # 一条评论最多切几条观点

# 检索
SEARCH_K = 5                # 返回 Top-K

# 路径
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "reviews.json"
CHROMA_DIR = BASE_DIR / "chroma_db"


def get_llm(temperature: float = 0.3):
    """统一 LLM 工厂 — 所有模块用这一个函数拿 DeepSeek ChatOpenAI"""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


# ── Embedding 单例（只加载一次，避免每次搜索都重新加载模型） ──
_embeddings = None

def get_embeddings():
    """全局单例 Embedding — 所有模块共用，避免 hyde_retriever 每次搜索重新加载 100MB 模型"""
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings
