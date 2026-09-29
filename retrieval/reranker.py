"""
Cross-Encoder 精排器

原理：粗排（向量/BM25）返回 Top-20 → Cross-Encoder 逐条重新打分 → 返回 Top-5
和 embedding 的区别：Cross-Encoder 把 query 和 doc 拼在一起过 transformer，
                     逐字做 cross-attention，精度远高于向量距离

模型：BAAI/bge-reranker-base（中文 Cross-Encoder，~1GB，CPU 可跑）
"""
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-Encoder 重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """中文 Cross-Encoder 精排（BGE Reranker），替代英文 ms-marco 避免中文场景 Recall 倒挂"""
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list, top_k: int = 5) -> list:
        """对候选文档重新打分，返回 Top-K"""
        if not docs or len(docs) <= top_k:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        print(f"[Reranker] 粗排 {len(docs)} 条 → 精排 Top-{top_k}:")
        for i, (doc, score) in enumerate(scored[:top_k], 1):
            print(f"  {i}. [{score:.4f}] {doc.page_content[:60]}")

        return [doc for doc, _ in scored[:top_k]]


if __name__ == "__main__":
    from retrieval.store import load_vectorstore
    from retrieval.hybrid import hybrid_retrieve
    from langchain_core.documents import Document

    vs = load_vectorstore()
    raw = vs.get(include=["metadatas", "documents"])
    chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw['documents'], raw['metadatas'])
    ]

    candidates = hybrid_retrieve("外观好看吗", vs, chunks=chunks, k=20)
    print(f"\n粗排召回: {len(candidates)} 条")

    reranker = Reranker()
    results = reranker.rerank("外观好看吗", candidates, top_k=5)

    print(f"\n--- Cross-Encoder 精排结果 ---")
    for i, doc in enumerate(results, 1):
        s = "👍" if doc.metadata.get('sentiment') == 'positive' else "👎"
        print(f"  {i}. {s} {doc.page_content[:100]}")
