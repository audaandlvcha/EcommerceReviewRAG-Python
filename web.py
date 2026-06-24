"""
FastAPI —— 搜索 API + SSE 流式 + 多轮对话

端点：
  POST /api/search              REST 检索（含 session 支持）
  GET  /api/search/stream       SSE 流式（含 session、对话上下文）
  GET  /                        前端页面（static/index.html）
"""
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


def get_pipeline():
    global _pipeline, _vectorstore, _bm25_chunks, _router, _conversation, _langfuse_cb
    if _pipeline is None:
        from vector_store import load_vectorstore
        from rag_pipeline import RAGPipeline
        from reranker import Reranker
        from intent_router import IntentRouter
        from conversation_manager import ConversationManager
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

    pipeline = get_pipeline()
    conversation = _conversation

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
        return

    async def generate():
        pipeline = get_pipeline()
        conversation = _conversation

        sid = session_id or str(uuid.uuid4())

        # 立即返回状态，防止浏览器超时断开
        yield f"data: {json.dumps({'type': 'status', 'data': '检索中...'})}\n\n"

        # 多轮对话 + 意图路由 + 检索（用线程防止阻塞事件循环）
        enriched = conversation.enrich_query(query, sid)
        answer, docs = await asyncio.to_thread(
            pipeline.query, enriched, True, True, True, SEARCH_K
        )

        # 记录对话历史
        conversation.add_message(sid, "user", query)
        conversation.add_message(sid, "assistant", answer)

        # 先发 session_id + 引用
        yield f"data: {json.dumps({'type': 'meta', 'session_id': sid, 'enriched_query': enriched if enriched != query else None}, ensure_ascii=False)}\n\n"

        sources = [
            {
                "content": doc.page_content,
                "sentiment": doc.metadata.get("sentiment", ""),
                "rating": doc.metadata.get("rating", 0),
            }
            for doc in docs
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

        for char in answer:
            yield f"data: {json.dumps({'type': 'token', 'data': char}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path(__file__).parent / "static" / "index.html"
    return index_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
