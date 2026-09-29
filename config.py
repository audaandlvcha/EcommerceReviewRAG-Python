"""配置 — 所有可调参数集中在这里"""
import os
import threading
from pathlib import Path
from langchain_openai import ChatOpenAI

# ── DeepSeek API（LLM） ──
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ── 本地 Embedding（HuggingFace，无需额外服务） ──
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
    """统一 LLM 工厂 — 所有模块用这一个函数拿 DeepSeek ChatOpenAI

    注意：每次调用时重新读取环境变量，支持在导入之后才加载 .env 的场景
    （如 python web.py 直启时 __main__ 中才 load_dotenv）
    """
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


# ── Embedding 单例（线程安全，只加载一次） ──
_embeddings = None
_embeddings_lock = threading.Lock()

def get_embeddings():
    """全局单例 Embedding — 所有模块共用，线程安全

    HF_HUB_OFFLINE=1 时自动传 local_files_only=True，
    适配新版 sentence-transformers（不读取 HF_HUB_OFFLINE 环境变量）
    """
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                model_kwargs = {}
                if os.environ.get("HF_HUB_OFFLINE") == "1":
                    model_kwargs["local_files_only"] = True
                _embeddings = HuggingFaceEmbeddings(
                    model_name=EMBED_MODEL,
                    model_kwargs=model_kwargs
                )
    return _embeddings
