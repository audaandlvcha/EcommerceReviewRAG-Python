"""
FastAPI —— 搜索 API + SSE 流式 + 多轮对话

端点：
  POST /api/search              REST 检索（含 session 支持）
  GET  /api/search/stream       SSE 流式（含 session、对话上下文）
  GET  /                        前端页面（static/index.html）

启动：python main.py / python run.py / python web.py 均可
"""
# ── 必须在所有 import 之前加载 .env + 设离线模式 ──
import os as _os
from pathlib import Path as _Path
from dotenv import load_dotenv
load_dotenv(_Path(__file__).parent / ".env")

for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    if not _os.environ.get(_key):
        _os.environ[_key] = "1"

if "SSL_CERT_FILE" in _os.environ:
    del _os.environ["SSL_CERT_FILE"]
# ──────────────────────────────────────────

import json
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.documents import Document
from config import SEARCH_K

app = FastAPI(title="电商评论 RAG Plus")

_vectorstore = None
_pipeline = None
_bm25_chunks = None
_router = None
_conversation = None
_langfuse_cb = None
_init_lock = asyncio.Lock()


async def get_pipeline():
    global _pipeline, _vectorstore, _bm25_chunks, _router, _conversation, _langfuse_cb
    if _pipeline is not None:
        return _pipeline

    async with _init_lock:
        # 双重检查：拿到锁后再确认一次（防并发创建两套）
        if _pipeline is not None:
            return _pipeline

        from retrieval.store import load_vectorstore
        from pipeline.orchestrator import RAGPipeline
        from retrieval.reranker import Reranker
        from pipeline.router import IntentRouter
        from conversation.manager import ConversationManager
        from langfuse_setup import get_langfuse_callback

        _vectorstore = load_vectorstore()
        raw = _vectorstore.get(include=["metadatas", "documents"])
        _bm25_chunks = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw['documents'], raw['metadatas'])
        ]
        _router = IntentRouter()
        _conversation = ConversationManager()
        _langfuse_cb = get_langfuse_callback()

        _pipeline = RAGPipeline(
            _vectorstore, _bm25_chunks,
            reranker=Reranker(),
            router=_router,
            langfuse_callback=_langfuse_cb
        )
        return _pipeline


@app.post("/api/search")
async def search(request: Request):
    body = await request.json()
    question = body.get("query", "").strip()
    session_id = body.get("session_id", str(uuid.uuid4()))
    if not question:
        return {"error": "query 不能为空"}

    pipeline = await get_pipeline()
    conversation = _conversation

    # 防御：确保 conversation 已初始化
    if conversation is None:
        from conversation.manager import ConversationManager
        conversation = ConversationManager()

    # 多轮对话：检测追问 → 增强 query
    enriched = conversation.enrich_query(question, session_id)

    # 意图路由 + 检索
    answer, docs = pipeline.query(enriched, auto_route=True)

    # 记录对话历史
    conversation.add_message(session_id, "user", question)
    conversation.add_message(session_id, "assistant", answer)

    return {
        "query": question,
        "enriched_query": enriched if enriched != question else None,
        "answer": answer,
        "session_id": session_id,
        "sources": [
            {
                "content": doc.page_content,
                "sentiment": doc.metadata.get("sentiment", ""),
                "rating": doc.metadata.get("rating", 0),
            }
            for doc in docs
        ],
    }


@app.get("/api/search/stream")
async def search_stream(query: str = "", session_id: str = ""):
    if not query.strip():
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'data': 'query 不能为空'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def generate():
        try:
            pipeline = await get_pipeline()
            conversation = _conversation
            if conversation is None:
                from conversation.manager import ConversationManager
                conversation = ConversationManager()

            sid = session_id or str(uuid.uuid4())

            # 多轮对话：检测追问 → 增强 query
            enriched = conversation.enrich_query(query, sid)

            # 先发状态
            yield f"data: {json.dumps({'type': 'status', 'data': '检索中...'})}\n\n"

            # 意图路由 + 检索 + 真正流式 LLM（用线程防阻塞事件循环）
            final_docs, token_stream = await asyncio.to_thread(
                pipeline.stream_query, enriched, auto_route=True
            )

            # 记录对话历史（先记用户问题）
            conversation.add_message(sid, "user", query)

            # 发 session_id + enriched_query
            yield f"data: {json.dumps({'type': 'meta', 'session_id': sid, 'enriched_query': enriched if enriched != query else None}, ensure_ascii=False)}\n\n"

            # 发引用来源
            sources = [
                {
                    "content": doc.page_content,
                    "sentiment": doc.metadata.get("sentiment", ""),
                    "rating": doc.metadata.get("rating", 0),
                }
                for doc in final_docs
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

            # 真正流式：逐 token 发送
            answer = ""
            for token in token_stream:
                answer += token
                yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

            # 记录完整回答
            conversation.add_message(sid, "assistant", answer)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': f'检索异常: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path(__file__).parent / "static" / "index.html"
    return index_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    import os as _os
    from pathlib import Path as _Path

    # 与 main.py / run.py 保持一致：加载 .env + 离线模式
    from dotenv import load_dotenv
    load_dotenv(_Path(__file__).parent / ".env")

    for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if not _os.environ.get(_key):
            _os.environ[_key] = "1"

    if "SSL_CERT_FILE" in _os.environ:
        del _os.environ["SSL_CERT_FILE"]

    _port = int(_os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=_port)
