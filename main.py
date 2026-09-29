"""入口 —— python main.py 启动 Web 服务（开发模式，带热重载）"""
import os
import sys
from pathlib import Path

# ── 第一步：加载 .env ──
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── 第二步：HuggingFace 离线模式 ──
# 多环境变量确保 sentence_transformers / transformers / datasets 完全离线
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ── 第三步：修复 SSL 证书（与 run.py 保持一致） ──
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import uvicorn

if __name__ == "__main__":
    import os as _os
    _port = int(_os.environ.get("PORT", "8000"))
    uvicorn.run("web:app", host="0.0.0.0", port=_port, reload=True)
