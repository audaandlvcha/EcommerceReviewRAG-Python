"""
RAG 全链路（Agentic RAG v2.1）—— 自适应检索管道

调用链（新）：
  question → QueryAnalyzer(6种意图识别 + 子问题拆解)
           → 子问题并行检索（每路独立：HyDE → Hybrid → Reranker）
           → 结果合并去重
           → 自适应合成（6种回答风格，按意图动态选择）

面试画架构图就是画这个文件。
"""

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import SEARCH_K, get_llm


class RAGPipeline:
    """RAG 全链路管道（Agentic RAG v2.1：意图自适应 + 子问题分解 + 动态合成）"""

    def __init__(self, vectorstore, bm25_chunks, llm=None, reranker=None,
                 router=None, langfuse_callback=None, query_analyzer=None):
        self.vectorstore = vectorstore
        self.bm25_chunks = bm25_chunks
        self.llm = llm or get_llm(temperature=0.3)
        self.reranker = reranker
        self.router = router  # 保留旧 IntentRouter 兼容
        self.langfuse_callback = langfuse_callback
        self.query_analyzer = query_analyzer  # 新 QueryAnalyzer

    # ═══════════════════════════════════════════════════
    # 检索核心：一轮检索（HyDE + Hybrid + Reranker）
    # ═══════════════════════════════════════════════════

    def _retrieve_one(self, query: str, top_k: int = SEARCH_K) -> list:
        """单轮检索：HyDE → Hybrid → Reranker → 返回 top_k 文档"""
        all_candidates = []

        # HyDE 增强
        from retrieval.hyde import hyde_retrieve
        try:
            hyde_docs = hyde_retrieve(query, self.vectorstore, self.llm, k=top_k * 4)
            all_candidates.extend(hyde_docs)
        except Exception:
            pass

        # 混合检索
        from retrieval.hybrid import hybrid_retrieve
        from langchain_community.retrievers import BM25Retriever
        bm25_retriever = BM25Retriever.from_documents(self.bm25_chunks)
        bm25_retriever.k = top_k * 5

        docs = hybrid_retrieve(
            query, self.vectorstore,
            bm25_retriever=bm25_retriever,
            chunks=self.bm25_chunks,
            k=top_k * 4,
        )
        all_candidates.extend(docs)

        # 去重
        seen = set()
        unique_candidates = []
        for doc in all_candidates:
            key = (doc.page_content, doc.metadata.get("review_id", ""))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(doc)

        # Cross-Encoder 精排
        if self.reranker is not None and len(unique_candidates) > top_k:
            return self.reranker.rerank(query, unique_candidates, top_k=top_k)
        return unique_candidates[:top_k]

    # ═══════════════════════════════════════════════════
    # 主查询入口
    # ═══════════════════════════════════════════════════

    def query(self, question: str, use_multi_query: bool = True,
              use_hyde: bool = True, auto_route: bool = False,
              top_k: int = SEARCH_K,
              conversation_context: str = "") -> tuple[str, list]:
        """
        Agentic RAG 自适应查询

        - auto_route=True 时：
          1. QueryAnalyzer 识别 6 种意图
          2. comparison/decision_advice → 拆子问题并行检索
          3. 其他类型 → 单轮检索
          4. 根据意图类型自适应合成回答
        """

        # ── 第0步：QueryAnalyzer 意图分析 ────────────────
        query_type = "general"
        sub_queries = []
        answer_style = None

        if auto_route and self.query_analyzer is not None:
            analysis = self.query_analyzer.analyze(question)
            query_type = analysis["query_type"]
            sub_queries = analysis["sub_queries"]
            answer_style = analysis["answer_style"]

        # ── 第1步：检索（自适应） ──────────────────────────
        if sub_queries:
            # 复杂查询：每个子问题独立检索 → 合并去重
            print(f"[Pipeline] Agentic: 拆成 {len(sub_queries)} 个子问题并行检索")
            all_docs = []
            for sq in sub_queries:
                sq_docs = self._retrieve_one(sq, top_k=top_k)
                all_docs.extend(sq_docs)

            # 跨子问题去重
            seen = set()
            final_docs = []
            for doc in all_docs:
                key = (doc.page_content, doc.metadata.get("review_id", ""))
                if key not in seen:
                    seen.add(key)
                    final_docs.append(doc)

            # 用原问题精排
            if self.reranker is not None and len(final_docs) > top_k:
                final_docs = self.reranker.rerank(question, final_docs, top_k=top_k)
            else:
                final_docs = final_docs[:top_k]
            print(f"[Pipeline] 合并后候选: {len(all_docs)} → 去重精排: {len(final_docs)} 条")

        elif auto_route and self.router is not None:
            # 兼容旧 IntentRouter（无 QueryAnalyzer 时）
            strategy = self.router.get_strategy(question)
            final_docs = self._retrieve_with_strategy(
                question, strategy, top_k
            )

        else:
            # 单轮检索
            final_docs = self._retrieve_one(question, top_k=top_k)

        # ── 第2步：自适应合成 ──────────────────────────
        context = "\n\n".join(
            f"[{i+1}] {doc.page_content}" for i, doc in enumerate(final_docs)
        )

        # 用 QueryAnalyzer 的动态风格模板，fallback 通用模板
        if answer_style:
            system_prompt = answer_style.replace("{context}", context)
        else:
            system_prompt = f"""你是电商客服，基于用户真实评论来回答买家的问题。

规则：
1. 引用评论中的真实观点，正面和负面都要提
2. 如果多条评论观点一致，可以综合（如"多数用户认为..."）
3. 如果得不到足够信息，直接说"这方面的用户反馈比较少"
4. 回答简洁（100-150字），用自然的口语

{conversation_context}
以下是相关用户评论：
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()

        invoke_kwargs = {"question": question}
        if self.langfuse_callback is not None:
            invoke_kwargs["callbacks"] = [self.langfuse_callback]

        try:
            answer = chain.invoke(invoke_kwargs)
            prefix = f"[Pipeline] ({query_type}) "
            print(f"{prefix}LLM 答案: {answer[:80]}..." if answer else f"{prefix}LLM 返回空答案")
        except Exception as e:
            print(f"[Pipeline] LLM 调用失败: {e}")
            answer = f"抱歉，AI 服务暂时不可用（{str(e)[:100]}），请稍后重试。"

        return answer, final_docs

    # ═══════════════════════════════════════════════════
    # 兼容旧策略的检索方法
    # ═══════════════════════════════════════════════════

    def _retrieve_with_strategy(self, question: str, strategy: dict, top_k: int) -> list:
        """使用旧 IntentRouter 策略的检索（向后兼容）"""
        use_multi_query = strategy.get("use_multi_query", False)
        use_hyde = strategy.get("use_hyde", True)

        all_candidates = []

        if use_hyde:
            from retrieval.hyde import hyde_retrieve
            try:
                hyde_docs = hyde_retrieve(question, self.vectorstore, self.llm, k=top_k * 4)
                all_candidates.extend(hyde_docs)
            except Exception:
                pass

        from retrieval.hybrid import hybrid_retrieve
        from langchain_community.retrievers import BM25Retriever
        bm25_retriever = BM25Retriever.from_documents(self.bm25_chunks)
        bm25_retriever.k = top_k * 5

        search_queries = [question]
        if use_multi_query:
            from retrieval.multi_query import generate_multi_queries
            mq_queries = generate_multi_queries(question, self.llm)
            search_queries.extend(mq_queries)

        for q in search_queries:
            docs = hybrid_retrieve(q, self.vectorstore, bm25_retriever=bm25_retriever,
                                   chunks=self.bm25_chunks, k=top_k * 4)
            all_candidates.extend(docs)

        seen = set()
        unique_candidates = []
        for doc in all_candidates:
            key = (doc.page_content, doc.metadata.get("review_id", ""))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(doc)

        if self.reranker is not None and len(unique_candidates) > top_k:
            return self.reranker.rerank(question, unique_candidates, top_k=top_k)
        return unique_candidates[:top_k]



    def stream_query(self, question: str, use_multi_query: bool = True,
                     use_hyde: bool = True, auto_route: bool = False,
                     top_k: int = SEARCH_K,
                     conversation_context: str = ""):
        """
        流式 RAG 流程（Agentic v2.1）— 检索复用 query()，LLM 阶段逐 token 产出

        先 yield ("docs", final_docs)，再逐个 yield ("token", char)
        """
        # 复用 query() 的检索逻辑，拿到 final_docs
        # （流式版本的检索和 query() 一样，区别只在 LLM 输出方式）
        answer, final_docs = self.query(
            question, use_multi_query=use_multi_query,
            use_hyde=use_hyde, auto_route=auto_route,
            top_k=top_k, conversation_context=conversation_context,
        )

        # 对已生成的 answer 逐字拆分（模拟流式）
        def _char_stream(text: str):
            for char in text:
                yield char

        return final_docs, _char_stream(answer)


if __name__ == "__main__":
    import os
    for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if not os.environ.get(_key):
            os.environ[_key] = "1"

    from retrieval.store import load_vectorstore
    from langchain_core.documents import Document
    from retrieval.reranker import Reranker
    from retrieval.query_decomposer import QueryAnalyzer

    vs = load_vectorstore()
    raw = vs.get(include=["metadatas", "documents"])
    chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw['documents'], raw['metadatas'])
    ]

    analyzer = QueryAnalyzer()
    pipeline = RAGPipeline(vs, chunks, reranker=Reranker(), query_analyzer=analyzer)

    # 测试 Agentic RAG
    print("=" * 60)
    queries = [
        "续航怎么样",
        "有什么缺点",
        "值不值得买",
        "Phone X1和Phone X1 Pro有什么区别",
        "像素多少",
    ]
    for q in queries:
        answer, docs = pipeline.query(q, auto_route=True)
        print(f"\n[{q}]")
        print(f"  {answer[:150]}...")
        print(f"  检索: {len(docs)} 条")
