"""
Recall@5 评测 — 对比三种检索配置

配置:
  A: 纯向量 (dense only)
  B: 向量 + BM25 (hybrid, 无 reranker)
  C: 混合 RRF + Reranker (完整链路)

评测: LLM-as-judge 判断每条检索结果是否相关, 统计 Recall@5

用法: python eval_recall.py
"""

import os, sys, time
from pathlib import Path
from types import ModuleType

_m = ModuleType("langchain_community.chat_models.vertexai")
_m.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models", ModuleType("langchain_community.chat_models"))
sys.modules["langchain_community.chat_models.vertexai"] = _m

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from langchain_core.documents import Document
from config import SEARCH_K, get_llm

# ══════════════════════════════════════════════════════
# 测试集: 20 条 query
# ══════════════════════════════════════════════════════

TEST_QUERIES = [
    "续航怎么样",
    "屏幕清晰吗",
    "拍照好不好",
    "信号稳定吗",
    "外观好看吗",
    "值得买吗",
    "有什么缺点",
    "充电快吗",
    "音质怎么样",
    "系统流畅吗",
    "像素高吗",
    "电池耐用吗",
    "手感如何",
    "发热严重吗",
    "适合送父母吗",
    "颜色有几种",
    "会不会卡顿",
    "防水吗",
    "售后好不好",
    "玩游戏性能怎么样",
]


def judge_relevance(query: str, docs: list, llm) -> list[bool]:
    """LLM-as-judge: 判断每条 document 是否与 query 相关"""
    results = []
    for doc in docs:
        prompt = (
            f"判断以下评论片段是否与用户问题相关。只回复 YES 或 NO。\n\n"
            f"用户问题: {query}\n"
            f"评论片段: {doc.page_content[:200]}\n\n"
            f"相关? "
        )
        try:
            resp = llm.invoke(prompt)
            text = resp.content.strip().upper() if hasattr(resp, 'content') else str(resp).strip().upper()
            results.append("YES" in text)
        except Exception:
            results.append(False)
    return results


def run_eval():
    print("[STEP] 加载向量库...")
    from vector_store import load_vectorstore
    vs = load_vectorstore()
    raw = vs.get(include=["metadatas", "documents"])
    bm25_chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    print(f"[OK] {len(bm25_chunks)} 个文档\n")

    from hybrid_retriever import hybrid_retrieve
    from reranker import Reranker
    reranker = Reranker()

    judge_llm = get_llm(temperature=0)

    # 汇总
    totals = {"dense": 0, "hybrid": 0, "full": 0}
    latencies = {"dense": [], "hybrid": [], "full": []}

    for i, query in enumerate(TEST_QUERIES, 1):
        # ── 配置A: 纯向量 ──
        t0 = time.time()
        dense_docs = vs.similarity_search(query, k=SEARCH_K)
        latencies["dense"].append(time.time() - t0)

        # ── 配置B: 向量+BM25 (无reranker) ──
        t0 = time.time()
        hybrid_docs = hybrid_retrieve(query, vs, chunks=bm25_chunks, k=SEARCH_K)
        hybrid_docs = hybrid_docs[:SEARCH_K]  # 取 top-5
        latencies["hybrid"].append(time.time() - t0)

        # ── 配置C: 完整链路 (hybrid + reranker) ──
        t0 = time.time()
        all_candidates = hybrid_retrieve(query, vs, chunks=bm25_chunks, k=SEARCH_K * 4)
        reranked_docs = reranker.rerank(query, all_candidates, top_k=SEARCH_K)
        latencies["full"].append(time.time() - t0)

        # LLM 评分
        dense_rel = judge_relevance(query, dense_docs, judge_llm)
        hybrid_rel = judge_relevance(query, hybrid_docs, judge_llm)
        full_rel = judge_relevance(query, reranked_docs, judge_llm)

        d_hits = sum(dense_rel)
        h_hits = sum(hybrid_rel)
        f_hits = sum(full_rel)
        totals["dense"] += d_hits
        totals["hybrid"] += h_hits
        totals["full"] += f_hits

        print(f"  [{i:2d}] {query:14s}  dense={d_hits}/5  hybrid={h_hits}/5  full={f_hits}/5  "
              f"  d{latencies['dense'][-1]:.1f}s  h{latencies['hybrid'][-1]:.1f}s  f{latencies['full'][-1]:.1f}s")

        # 打印 full 的前3条供人工抽查
        if i <= 3:
            for j, doc in enumerate(reranked_docs[:3]):
                print(f"       [{j+1}] {doc.page_content[:60]}...")

    # ════════════════════════════════════════
    # 报告
    # ════════════════════════════════════════
    n = len(TEST_QUERIES)
    max_hits = n * 5  # 20 queries * 5 docs each

    print("\n" + "=" * 65)
    print("[REPORT] Recall@5 对比报告")
    print("=" * 65)

    for name, label in [("dense", "A: 纯向量"), ("hybrid", "B: 向量+BM25"), ("full", "C: 混合+RRF+Reranker")]:
        hits = totals[name]
        recall = hits / max_hits * 100
        avg_lat = sum(latencies[name]) / len(latencies[name])
        print(f"\n  {label}")
        print(f"    Recall@5: {hits}/{max_hits} = {recall:.1f}%")
        print(f"    平均延迟: {avg_lat:.2f}s")

    # 提升幅度
    d_recall = totals["dense"] / max_hits * 100
    h_recall = totals["hybrid"] / max_hits * 100
    f_recall = totals["full"] / max_hits * 100
    print(f"\n  +-- [RESUME] 简历可用数据 ────────────────")
    print(f"  |  Recall@5: {d_recall:.0f}% (纯向量) -> {h_recall:.0f}% (+BM25) -> {f_recall:.0f}% (+Reranker)")
    print(f"  |  检索延迟: {sum(latencies['full'])/len(latencies['full']):.1f}s (完整链路)")
    print(f"  +───────────────────────────────────────\n")


if __name__ == "__main__":
    run_eval()
