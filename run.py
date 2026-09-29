"""启动脚本 — 加载 .env + 修复 SSL 证书 + 启动 RAG 服务"""
import os
import sys
from pathlib import Path

# ── 第一步：加载 .env（必须在其他 import 前） ──
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── 第二步：HuggingFace 离线模式（多环境变量确保完全离线） ──
# 首次使用或换模型时临时注释掉这些行
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ── 第三步：修复 SSL 证书问题 ──
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# 现在安全地 import 其他模块
import uvicorn

if __name__ == "__main__":
    import os as _os
    _port = int(_os.environ.get("PORT", "8000"))
    uvicorn.run("web:app", host="0.0.0.0", port=_port, reload=False)
