"""
多轮对话管理器

面试怎么讲：
  "RAG 只是单轮问答，用户问完'续航怎么样'再问'那充电呢'就不认识了。
  我加了对话管理器——维护会话历史，检测追问/指代（'那''它''还有'），
  把上一轮话题自动拼入当前 query 的检索上下文。"
"""

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path
from datetime import datetime, timedelta
from config import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# 追问检测 prompt
FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """判断用户的提问是否是追问（follow-up question）。

追问的特征：
- 使用了代词指代上文（"那""它""这个""还有"）
- 省略了主语（如上一轮问"续航怎么样"，这轮问"那充电呢"——省略了"手机"）
- 是对上一轮话题的延续、细化、对比

如果是追问，请输出上一轮的核心话题（3-5个词），用于拼入检索 query。
如果不是追问，输出 NO。

上一轮对话：
用户: {last_question}
系统: {last_answer}

只回复话题关键词或 NO。"""),
    ("human", "{current_question}")
])


class ConversationManager:
    """多轮对话管理器（带 TTL 和最大 session 数限制，防内存泄漏）"""

    def __init__(self, max_history: int = 10, max_sessions: int = 1000, ttl_minutes: int = 60):
        self.sessions: dict[str, list[dict]] = {}
        self.max_history = max_history
        self.max_sessions = max_sessions
        self.ttl_minutes = ttl_minutes
        self._last_cleanup = datetime.now()

    def _evict_if_needed(self):
        """超过最大 session 数时淘汰最旧的，超过 TTL 的也清理"""
        now = datetime.now()
        # 每 5 分钟清理一次，避免每次都遍历
        if (now - self._last_cleanup).total_seconds() < 300 and len(self.sessions) <= self.max_sessions:
            return
        self._last_cleanup = now

        cutoff = now - timedelta(minutes=self.ttl_minutes)
        cutoff_str = cutoff.isoformat()
        stale = [
            sid for sid, msgs in self.sessions.items()
            if not msgs or msgs[-1]["time"] < cutoff_str
        ]
        for sid in stale:
            del self.sessions[sid]

        # 如果还超限，淘汰最旧的
        if len(self.sessions) > self.max_sessions:
            sorted_sids = sorted(
                self.sessions.keys(),
                key=lambda s: self.sessions[s][-1]["time"] if self.sessions[s] else "0"
            )
            excess = len(self.sessions) - self.max_sessions
            for sid in sorted_sids[:excess]:
                del self.sessions[sid]

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat()
        })
        # 只保留最近 max_history 条
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]

        self._evict_if_needed()

    def get_last_exchange(self, session_id: str) -> tuple[str, str] | None:
        """获取最近一轮完整的问答配对（用户问题, 系统回答）

        按时间顺序找到最后一条 user 消息，确保它后面有一条 assistant 回复
        （避免用户连发两条消息时错误配对）
        """
        history = self.sessions.get(session_id, [])
        if not history:
            return None

        # 从尾部找最后一条 user，确保它后面紧跟 assistant
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return None

        # 找到 user 之后最近的一条 assistant
        for j in range(last_user_idx + 1, len(history)):
            if history[j]["role"] == "assistant":
                return history[last_user_idx]["content"], history[j]["content"]

        return None  # user 发了消息但还没有 assistant 回复

    def enrich_query(self, current_question: str, session_id: str,
                     llm=None) -> str:
        """
        检测是否是追问，如果是则把上一轮话题拼入 query

        示例：
          上一轮："续航怎么样"
          当前："那充电呢"  → 返回 "那充电呢 续航 充电"（加入关键词）
          当前："信号好不好" → 返回 "信号好不好"（不是追问，原样返回）
        """
        last = self.get_last_exchange(session_id)
        if not last:
            return current_question

        last_q, last_a = last

        if llm is None:
            llm = get_llm(temperature=0)

        chain = FOLLOWUP_PROMPT | llm | StrOutputParser()
        result = chain.invoke({
            "last_question": last_q,
            "last_answer": last_a[:200],
            "current_question": current_question
        }).strip()

        if result.upper() == "NO":
            print(f"[Conv] '{current_question[:20]}' → 新话题")
            return current_question

        enriched = f"{current_question} {result}"
        print(f"[Conv] '{current_question[:20]}' → 追问，补关键词: {result}")
        return enriched

    def get_session(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])


# 全局单例
conversation_manager = ConversationManager()


if __name__ == "__main__":
    cm = ConversationManager()
    cm.add_message("test", "user", "续航怎么样")
    cm.add_message("test", "assistant", "多数用户认为续航不错...")
    enriched = cm.enrich_query("那充电呢", "test")
    print(f"增强后 query: {enriched}")
