"""
Langfuse 全链路追踪（v3.x+ API）

面试怎么讲：
  "两个项目统一了可观测性方案——Langfuse 追踪每次检索的延迟、token 消耗、
  重排前后对比。不做追踪的 RAG 是黑盒，上线后检索质量退化了你都不知道。"

集成方式：Langfuse Cloud 免费版
  1. 去 https://cloud.langfuse.com 注册
  2. Settings → API Keys → 创建
  3. 设环境变量：LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY

依赖：pip install langfuse
"""
import os
from pathlib import Path

try:
    # langfuse v3+ 的正确导入路径
    from langfuse.callback.langchain import CallbackHandler
    _HAS_LANGFUSE = True
except ImportError:
    try:
        # langfuse v2 的旧路径兼容
        from langfuse.langchain import CallbackHandler
        _HAS_LANGFUSE = True
    except ImportError:
        _HAS_LANGFUSE = False
        CallbackHandler = None  # 类型占位，无 langfuse 时优雅降级


def get_langfuse_callback(session_id: str = None, tags: list[str] = None):
    """
    获取 Langfuse callback — 未安装/未配置时静默降级，不影响系统运行

    langfuse v4.x: CallbackHandler 从环境变量自动读取
    LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY
    """
    if not _HAS_LANGFUSE:
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return None

    try:
        # langfuse v4: public_key 必传，secret_key 从环境变量自动读取
        kwargs = {"public_key": public_key}
        if session_id:
            kwargs["session_id"] = session_id
        if tags:
            kwargs["tags"] = tags
        return CallbackHandler(**kwargs)
    except Exception:
        return None


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    cb = get_langfuse_callback()
    if cb:
        print("[Langfuse] 追踪已启用")
    else:
        print("[Langfuse] 追踪未配置（不影响系统运行）")
