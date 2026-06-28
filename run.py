"""启动脚本 — 加载 .env + 修复 SSL 证书 + 启动 RAG 服务"""
import os
import sys
from pathlib import Path

# ── 第一步：加载 .env（必须在其他 import 前） ──
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── 第二步：HuggingFace 离线模式（避免每次启动联网验证，中国网络 HF 不稳定） ──
# 首次使用或换模型时临时注释掉这行
os.environ["HF_HUB_OFFLINE"] = "1"

# ── 第三步：修复 SSL 证书问题 ──
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# 现在安全地 import 其他模块
import uvicorn

if __name__ == "__main__":
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=False)
