"""
查询意图识别 + 子问题分解（Agentic RAG 核心）

6 种查询类型 + 对应检索策略 + 对应回答风格：

  opinion_summary  → "好不好"/"怎么样"      → 正反观点并呈
  problem_focused  → "有什么缺点"/"问题"     → 只列负面
  decision_advice  → "值不值得买"           → 明确结论+正反依据
  comparison       → "A和B有什么区别"       → 拆子问题并行检索+对比
  factual_lookup   → "多少钱"/"什么配置"     → 直接检索+简洁回答
  general          → 兜底                   → 现有逻辑

用法:
  from retrieval.query_decomposer import QueryAnalyzer
  analyzer = QueryAnalyzer()
  result = analyzer.analyze("这手机值不值得买")
  # → {type: "decision_advice", sub_queries: [...], answer_style: "..."}
"""

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm

# ══════════════════════════════════════════════════════
# 意图分类 prompt
# ══════════════════════════════════════════════════════

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """判断用户问题的类型，只回复一个标签。

标签说明：
- opinion_summary: 问某方面好不好、怎么样（"屏幕好吗"、"续航怎么样"、"拍照好不好"）
- problem_focused: 专门问缺点/问题/不足（"有什么缺点"、"有什么问题"、"发热严重吗"）
- decision_advice: 问值不值得买、推不推荐（"值得买吗"、"推荐吗"、"性价比高吗"）
- comparison: 对比两个或多个东西（"A和B哪个好"、"有什么区别"、"和竞品比"）
- factual_lookup: 问具体事实/参数（"多少钱"、"什么配置"、"像素多少"、"有几种颜色"）
- general: 以上都不是

只回复标签，不要解释。"""),
    ("human", "{question}")
])

# ══════════════════════════════════════════════════════
# 子问题分解 prompt（仅 comparison 和 decision_advice 用）
# ══════════════════════════════════════════════════════

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """将用户的复杂问题拆成 2-3 个独立的简单子问题，每个子问题一行。
每个子问题应该可以独立检索，不要依赖其他子问题的结果。

示例：
用户问"Phone X1和Phone X1 Pro有什么区别"
→ Phone X1的优点
→ Phone X1 Pro的优点
→ Phone X1的价格

用户问"这手机值不值得买"
→ 这手机的优点
→ 这手机的缺点
→ 这手机的价格

只回复子问题，每行一个，不要编号。"""),
    ("human", "{question}")
])

# ══════════════════════════════════════════════════════
# 回答风格模板（按查询类型）
# ══════════════════════════════════════════════════════

ANSWER_STYLES = {
    "opinion_summary": """你是电商客服。请基于真实用户评论回答买家的问题。

规则：
1. 正面和负面观点都要列出，用"多数用户认为..."、"也有用户提到..."来区分
2. 正面和负面各 2-3 条
3. 回答简洁（80-120字），用自然的口语

以下是相关用户评论：
{context}""",

    "problem_focused": """你是电商客服。用户想知道产品有哪些问题/缺点，请如实反馈。

规则：
1. 只列出负面观点和问题，正面不用提（除非负面很少）
2. 如果评论中没有明显的负面反馈，诚实地说"这方面的负面反馈比较少"
3. 每个问题标注严重程度（如"多位用户反映"/"个别用户提到"）
4. 回答简洁（80-120字）

以下是相关用户评论：
{context}""",

    "decision_advice": """你是电商客服。用户想知道是否值得购买，请给出明确建议。

规则：
1. 先给一个明确的结论（"总体来说值得买"/"需要权衡"）
2. 列出主要的优点和缺点，各 2-3 条
3. 最后给出适合什么样的人群
4. 回答 100-150 字

以下是相关用户评论：
{context}""",

    "comparison": """你是电商客服。用户想对比不同选项，请清晰对比。

规则：
1. 逐项对比（价格/性能/体验等），每项说清楚哪个更好
2. 如果有明显优胜者，给出推荐
3. 结构清晰，可以分点
4. 回答 120-180 字

以下是相关用户评论：
{context}""",

    "factual_lookup": """你是电商客服。用户想知道具体的事实信息。

规则：
1. 直接回答问题，给出准确信息
2. 如果评论中有多个来源，综合给出最常见的答案
3. 不要展开无关内容
4. 回答简洁（50-80字）

以下是相关用户评论：
{context}""",

    "general": """你是电商客服，基于用户真实评论来回答买家的问题。

规则：
1. 引用评论中的真实观点，正面和负面都要提
2. 如果多条评论观点一致，可以综合（如"多数用户认为..."）
3. 如果得不到足够信息，直接说"这方面的用户反馈比较少"
4. 回答简洁（100-150字），用自然的口语

以下是相关用户评论：
{context}""",
}


class QueryAnalyzer:
    """查询意图分析器 — 识别问题类型 + 拆解子问题 + 选回答风格"""

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0)

    def classify_intent(self, question: str) -> str:
        """识别查询意图类型"""
        try:
            chain = INTENT_PROMPT | self.llm | StrOutputParser()
            result = chain.invoke({"question": question}).strip().lower()

            valid_types = {
                "opinion_summary", "problem_focused", "decision_advice",
                "comparison", "factual_lookup", "general"
            }
            for t in valid_types:
                if t in result:
                    return t
            return "general"
        except Exception:
            return "general"

    def decompose(self, question: str) -> list[str]:
        """将复杂问题拆成子问题（仅 comparison/decision_advice 调用）"""
        try:
            chain = DECOMPOSE_PROMPT | self.llm | StrOutputParser()
            result = chain.invoke({"question": question}).strip()

            # 每行一个子问题，过滤空行
            sub_queries = [
                line.strip().lstrip("→-•·1234567890.、 ").strip()
                for line in result.split("\n")
                if line.strip() and len(line.strip()) > 3
            ]
            return sub_queries[:3]  # 最多3个子问题
        except Exception:
            return [question]  # 失败就用原问题

    def get_answer_style(self, query_type: str) -> str:
        """根据查询类型返回对应的回答风格 prompt"""
        return ANSWER_STYLES.get(query_type, ANSWER_STYLES["general"])

    def needs_decomposition(self, query_type: str) -> bool:
        """判断是否需要拆子问题"""
        return query_type in ("comparison", "decision_advice")

    def analyze(self, question: str) -> dict:
        """一站式分析：返回查询类型 + 子问题列表 + 回答风格

        Returns:
            {
                "query_type": str,         # 查询类型标签
                "sub_queries": [str],      # 子问题列表（可能为空）
                "answer_style": str,       # 回答风格 prompt
                "needs_decomposition": bool,
            }
        """
        query_type = self.classify_intent(question)
        answer_style = self.get_answer_style(query_type)

        sub_queries = []
        if self.needs_decomposition(query_type):
            sub_queries = self.decompose(question)
            print(f"[QueryAnalyzer] '{question[:30]}' → {query_type} (拆成 {len(sub_queries)} 个子问题)")
        else:
            print(f"[QueryAnalyzer] '{question[:30]}' → {query_type}")

        return {
            "query_type": query_type,
            "sub_queries": sub_queries,
            "answer_style": answer_style,
            "needs_decomposition": len(sub_queries) > 0,
        }


if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    tests = [
        "续航怎么样",
        "有什么缺点",
        "值不值得买",
        "Phone X1和Phone X1 Pro有什么区别",
        "像素多少",
        "这手机适合打游戏吗",
    ]
    for t in tests:
        r = analyzer.analyze(t)
        print(f"  [{r['query_type']}] {t}")
        if r["sub_queries"]:
            for sq in r["sub_queries"]:
                print(f"    → {sq}")
        print()
