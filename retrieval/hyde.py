"""
HyDE 检索器 —— Hypothetical Document Embeddings

原理：用户问得短且模糊 → LLM 生成一段假想标准评论 →
     用假评论的向量去检索（而不是用原始问题的向量）

面试怎么讲：
  "用户问'续航怎么样'只有四个字，向量化后信息量太少。我用 LLM 先回答一遍——
  生成一段包含'电池容量''充电速度''能用多久'的假想评论，再拿这段假评论的向量
  去搜。本质上是用 LLM 把用户口语转成规范文本，缩小 query 和文档之间的语义鸿沟。"
"""

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import SEARCH_K, get_llm, get_embeddings

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是电商评论生成器。用户问了一个关于手机的疑问，你需要
写一段50-100字的假想评论。严格遵守以下规则：

1. 必须紧紧扣住用户问题中的关键词来写，不要偏题
2. 用简短直白的语言，像普通买家随手写的评价（20-40字即可）
   好的写法："外观挺好看的，颜色正手感好"
   坏的写法："薄边框配上曲面屏握感特别舒服雾光银金属光泽呼吸灯效果科幻片"
   → 不要堆砌形容词，不要用华丽辞藻，简单直接最好
3. 如果用户问题中的词有歧义（如"定位"可能是GPS也可能是市场定位），
   默认按手机硬件功能理解（GPS、导航），不要按商业概念理解
4. 只输出假想评论文本，不要加"用户问""我写"等任何前缀"""),
    ("human", "{question}")
])


def hyde_retrieve(question: str, vectorstore, llm=None, k: int = SEARCH_K) -> list:
    """
    HyDE 检索完整流程：
    1. LLM 生成假想评论
    2. 假想评论向量化
    3. 用假想评论的向量去 ChromaDB 搜
    4. 返回 Top-K 真实评论
    """
    if llm is None:
        llm = get_llm(temperature=0.3)
    #langchain
    chain = HYDE_PROMPT | llm | StrOutputParser()
    hypothetical_review = chain.invoke({"question": question})
    print(f"[HyDE] 假想评论: {hypothetical_review[:80]}...")

    embeddings = get_embeddings()  # 全局单例，避免每次搜索重新加载 100MB 模型
    hyde_vector = embeddings.embed_query(hypothetical_review)

    docs = vectorstore.similarity_search_by_vector(hyde_vector, k=k)
    print(f"[HyDE] 检索到 {len(docs)} 条评论")
    return docs


if __name__ == "__main__":
    from retrieval.store import load_vectorstore
    vs = load_vectorstore()
    results = hyde_retrieve("外观好看吗", vs)
    print("\n--- HyDE 检索结果 ---")
    for i, doc in enumerate(results, 1):
        s = "👍" if doc.metadata['sentiment'] == 'positive' else "👎"
        print(f"  {i}. {s} {doc.page_content[:100]}")
