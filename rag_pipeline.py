"""
RAG 全链路 —— 把 Day 1+2+3+4 的所有组件串成一条完整的问答管道

调用链（新）：
  question → IntentRouter(复杂度) → Conversation(追问检测)
           → MultiQuery(可选) → HyDE(可选) → Hybrid → CrossEncoder → LLM
           → [可选] Langfuse 全链路追踪

面试画架构图就是画这个文件。
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config import LLM_MODEL, SEARCH_K

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是电商客服，基于用户真实评论来回答买家的问题。

规则：
1. 引用评论中的真实观点，正面和负面都要提
2. 如果多条评论观点一致，可以综合（如"多数用户认为..."）
3. 如果得不到足够信息，直接说"这方面的用户反馈比较少"
4. 回答简洁（100-150字），用自然的口语

{conversation_context}
以下是相关用户评论：
{context}"""),
    ("human", "{question}")
])


class RAGPipeline:
    """RAG 全链路管道（加强版：意图路由 + 多轮对话 + Langfuse）"""

    def __init__(self, vectorstore, bm25_chunks, llm=None, reranker=None,
                 router=None, langfuse_callback=None):
        self.vectorstore = vectorstore
        self.bm25_chunks = bm25_chunks
        self.llm = llm or ChatOllama(model=LLM_MODEL, temperature=0.3)
        self.reranker = reranker
        self.router = router
        self.langfuse_callback = langfuse_callback

    def query(self, question: str, use_multi_query: bool = True,
              use_hyde: bool = True, auto_route: bool = False,
              top_k: int = SEARCH_K,
              conversation_context: str = "") -> tuple[str, list]:
        """
        完整 RAG 流程：

        - auto_route=True 时自动判断复杂度，忽略 use_multi_query/use_hyde 设置
        - conversation_context 非空时拼入 prompt（来自 ConversationManager）

        返回：(答案文本, 引用文档列表)
        """
        # ── 第0步：意图路由（新）──────────────────────
        if auto_route and self.router is not None:
            strategy = self.router.get_strategy(question)
            use_multi_query = strategy["use_multi_query"]
            use_hyde = strategy["use_hyde"]

        # ── 第1步：多角度扩展 ──────────────────────────
        search_queries = [question]
        if use_multi_query:
            from multi_query import generate_multi_queries
            mq_queries = generate_multi_queries(question, self.llm)
            search_queries.extend(mq_queries)
            print(f"[Pipeline] MultiQuery: {len(search_queries)} queries")

        # ── 第2步：HyDE 增强 ────────────────────────────
        if use_hyde:
            from hyde_retriever import hyde_retrieve
            hyde_docs = hyde_retrieve(question, self.vectorstore, self.llm, k=top_k * 4)
        else:
            hyde_docs = []

        # ── 第3步：混合检索 ─────────────────────────────
        from hybrid_retriever import hybrid_retrieve
        all_candidates = []
        for q in search_queries:
            docs = hybrid_retrieve(q, self.vectorstore, chunks=self.bm25_chunks, k=top_k * 4)
            all_candidates.extend(docs)

        all_candidates.extend(hyde_docs)
        seen = set()
        unique_candidates = []
        for doc in all_candidates:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_candidates.append(doc)

        print(f"[Pipeline] 去重后候选: {len(unique_candidates)} 条")

        # ── 第4步：Cross-Encoder 精排 ───────────────────
        if self.reranker is not None and len(unique_candidates) > top_k:
            final_docs = self.reranker.rerank(question, unique_candidates, top_k=top_k)
        else:
            final_docs = unique_candidates[:top_k]

        # ── 第5步：LLM 汇总 ─────────────────────────────
        context = "\n\n".join(
            f"[{i+1}] {doc.page_content}" for i, doc in enumerate(final_docs)
        )
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()

        answer = chain.invoke(
            {"context": context, "question": question,
             "conversation_context": conversation_context}
        )
        print(f"[Pipeline] LLM 答案: {answer[:80]}..." if answer else "[Pipeline] ⚠️ LLM 返回空答案")

        return answer, final_docs


if __name__ == "__main__":
    from vector_store import load_vectorstore
    from langchain_core.documents import Document
    from reranker import Reranker
    from intent_router import IntentRouter

    vs = load_vectorstore()
    raw = vs.get(include=["metadatas", "documents"])
    chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw['documents'], raw['metadatas'])
    ]

    router = IntentRouter()
    pipeline = RAGPipeline(vs, chunks, reranker=Reranker(), router=router)

    # 测试意图路由
    print("=" * 50)
    queries = ["外观", "续航怎么样", "值不值得买"]
    for q in queries:
        answer, docs = pipeline.query(q, auto_route=True)
        print(f"\n📱 [{q}] {answer[:100]}...")
