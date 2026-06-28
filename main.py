"""入口 —— python main.py 启动 Web 服务（开发模式，带热重载）"""
import os
import sys
from pathlib import Path

# ── 第一步：加载 .env ──
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── 第二步：HuggingFace 离线模式 ──
os.environ["HF_HUB_OFFLINE"] = "1"

# ── 第三步：修复 SSL 证书（与 run.py 保持一致，删掉不存在的路径） ──
if "SSL_CERT_FILE" in os.environ:
    if not os.path.exists(os.environ["SSL_CERT_FILE"]):
        del os.environ["SSL_CERT_FILE"]

import uvicorn

if __name__ == "__main__":
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=True)
