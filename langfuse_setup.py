"""
Langfuse 全链路追踪（v4.x API）

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
from langfuse.langchain import CallbackHandler

# 启动时自动从 .env 加载环境变量
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
    print(f"[Langfuse] 已加载 {_ENV_FILE}")


def get_langfuse_callback(session_id: str = None, tags: list[str] = None) -> CallbackHandler | None:
    """
    获取 Langfuse callback，如果环境变量没设则优雅降级

    Langfuse 4.x 通过环境变量读取配置：
      LANGFUSE_PUBLIC_KEY  /  LANGFUSE_SECRET_KEY  /  LANGFUSE_HOST

    参数：
        session_id: 会话 ID
        tags:       标签列表

    返回：
        CallbackHandler 或 None（未配置时）
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        print("[Langfuse] 未配置 API key，追踪已跳过")
        print("  设环境变量后启用: LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY")
        return None

    # Langfuse 4.x: CallbackHandler 只接受 public_key，secret_key 从环境变量读
    return CallbackHandler(public_key=public_key)


if __name__ == "__main__":
    cb = get_langfuse_callback()
    if cb:
        print("[Langfuse] 追踪已启用")
    else:
        print("[Langfuse] 追踪未配置（不影响系统运行）")
