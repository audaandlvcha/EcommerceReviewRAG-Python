"""
多查询检索器 —— Multi-Query Retrieval

原理：用户一个问题 → LLM 从 3 个角度重写 → 每个角度单独检索 → 合并去重

和 HyDE 的区别：
  - HyDE：生成假答案 → 用假答案的"语义"去匹配（一次检索）
  - MultiQuery：生成多个查询角度 → 用不同"角度"去覆盖（多次检索）
  - 两者互补：MultiQuery 生成 3 个角度 → 每个角度各用 HyDE 增强 → 效果最好

面试怎么讲：
  "用户问'续航怎么样'可能想了解电池容量、充电速度、实际使用时长三个维度。
  一次检索容易漏掉某个维度。我用 LLM 把问题重写成三个不同角度的查询，分别检索
  后合并去重——召回率比单次检索高很多。"
"""

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import SEARCH_K, get_llm

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是电商评论搜索助手。用户问了一个关于手机的问题，你需要
从 3 个不同角度各生成一个搜索查询。

规则（必须严格遵守）：
1. 角度1：产品参数 —— 用户问题里那个功能/部件的配置、规格、参数
2. 角度2：使用体验 —— 用户问题里那个功能在日常使用中好不好用
3. 角度3：优缺点 —— 用户问题里那个功能有什么优点和缺点

关键约束：
- 每个查询都必须包含用户问题中的关键词，围绕它展开，不要换成别的词
- 用户问"定位"你就写定位/GPS相关，绝对不能写成屏幕/续航/拍照
- 用户问"屏幕"你就写屏幕相关，绝对不能写成别的
- 只输出 3 行，每行一个查询，不要编号、不要"角度：""查询："等任何前缀"""),
    ("human", "{question}")
])


def generate_multi_queries(question: str, llm=None, n: int = 3) -> list[str]:
    """LLM 把问题重写成 N 个不同角度的查询"""
    if llm is None:
        llm = get_llm(temperature=0.3)

    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    queries = [q.strip() for q in result.split("\n") if q.strip()]
    queries = queries[:n]

    while len(queries) < n:
        queries.append(question)

    return queries


def multi_query_retrieve(question: str, vectorstore, llm=None, n: int = 3, k: int = SEARCH_K) -> list:
    """
    多查询检索完整流程：
    1. 生成 N 个查询角度
    2. 原始问题 + N 个角度都去检索
    3. 合并结果，按 page_content 去重
    4. 返回去重后的文档列表
    """
    if llm is None:
        llm = get_llm(temperature=0.3)

    queries = generate_multi_queries(question, llm, n)
    all_queries = [question] + queries
    print(f"[MultiQuery] 查询列表: {all_queries}")

    all_docs = []
    for q in all_queries:
        docs = vectorstore.similarity_search(q, k=k)
        all_docs.extend(docs)

    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    print(f"[MultiQuery] 去重前 {len(all_docs)} 条 → 去重后 {len(unique_docs)} 条")
    return unique_docs[:k]


if __name__ == "__main__":
    from retrieval.store import load_vectorstore
    vs = load_vectorstore()
    results = multi_query_retrieve("外观好看吗", vs)
    print("\n--- MultiQuery 检索结果 ---")
    for i, doc in enumerate(results, 1):
        s = "👍" if doc.metadata['sentiment'] == 'positive' else "👎"
        print(f"  {i}. {s} {doc.page_content[:100]}")
