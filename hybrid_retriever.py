"""
混合检索器 —— BM25 关键词 + 向量语义 → RRF 融合

原理：两路检索各取所长 → RRF 按排名融合 → 返回综合得分最高的文档

BM25 擅长：专有名词（"A17芯片""徕卡镜头"）、精确关键词匹配
向量擅长：同义词（"续航"≈"电池"）、语义相似（"拍照"≈"摄像"）

RRF (Reciprocal Rank Fusion): score = 1 / (k + rank)
  k=60 是经验值，让排名靠前的文档得分差距适中

参考：production-rag 的 hybrid.py（学 RRF 公式，用自己的 LangChain BM25Retriever）

面试怎么讲：
  "纯向量检索对专有名词不敏感——'骁龙8Gen3'这种词 embedding 没见过就是没见过。
  BM25 做关键词匹配正好补这个短板。两路检索用 RRF 融合——不在乎绝对分数，
  只在乎相对排名。语义路排第 1 和关键词路排第 3 的文档，最终得分可能差不多。"
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from config import SEARCH_K


def hybrid_retrieve(
    question: str,
    vectorstore,
    bm25_retriever=None,
    chunks=None,
    semantic_weight: float = 0.5,
    k_rrf: int = 60,
    k: int = SEARCH_K
) -> list:
    """
    BM25 + 向量混合检索，RRF 融合

    参数：
        question:        用户查询
        vectorstore:     ChromaDB 实例
        bm25_retriever:  BM25Retriever 实例（None=自动从 chunks 构建）
        chunks:          Document 列表（用于构建 BM25Retriever）
        semantic_weight: 向量路权重（0~1），BM25 权重 = 1 - semantic_weight
        k_rrf:           RRF 常数
        k:               返回 Top-K
    """
    # 第1步：准备 BM25 检索器
    if bm25_retriever is None:
        if chunks is None:
            raw = vectorstore.get(include=["metadatas", "documents"])
            chunks = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(raw['documents'], raw['metadatas'])
            ]
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = k * 5

    # 第2步：两路分别检索（多召回给融合留空间）
    retrieve_k = max(k * 5, 20)

    semantic_docs = vectorstore.similarity_search(question, k=retrieve_k)
    bm25_docs = bm25_retriever.invoke(question)[:retrieve_k]

    # 第3步：RRF 融合
    fused_scores = {}

    for rank, doc in enumerate(semantic_docs, 1):
        key = doc.page_content
        rrf_score = 1.0 / (k_rrf + rank)
        if key not in fused_scores:
            fused_scores[key] = {"doc": doc, "rrf_score": 0.0}
        fused_scores[key]["rrf_score"] += semantic_weight * rrf_score

    for rank, doc in enumerate(bm25_docs, 1):
        key = doc.page_content
        rrf_score = 1.0 / (k_rrf + rank)
        if key not in fused_scores:
            fused_scores[key] = {"doc": doc, "rrf_score": 0.0}
        fused_scores[key]["rrf_score"] += (1 - semantic_weight) * rrf_score

    # 第4步：按 RRF 得分排序
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True
    )

    print(f"[Hybrid] 向量路 {len(semantic_docs)} + BM25路 {len(bm25_docs)} → 融合后 {len(sorted_results)} 条")

    return [item[1]["doc"] for item in sorted_results[:k]]


if __name__ == "__main__":
    from vector_store import load_vectorstore
    from langchain_core.documents import Document

    vs = load_vectorstore()

    # 从 ChromaDB 直接拿全量文档 + metadata（不重新分块，不重新 embedding）
    raw = vs.get(include=["metadatas", "documents"])
    chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw['documents'], raw['metadatas'])
    ]

    results = hybrid_retrieve("外观好看吗", vs, chunks=chunks)
    print("\n--- 混合检索结果 ---")
    for i, doc in enumerate(results, 1):
        s = "👍" if doc.metadata['sentiment'] == 'positive' else "👎"
        print(f"  {i}. {s} {doc.page_content[:100]}")
