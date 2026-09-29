"""
RAG 评测脚本 — LLM-as-Judge 评估（用 DeepSeek 代替 RAGAS）

指标:
  - Faithfulness: 答案是否基于检索证据（不编造）
  - Answer Relevancy: 答案是否直接回答问题
  - 延迟统计

用法: python eval/quality.py（从项目根目录运行）

注意：此文件在子目录中，运行时需要把项目根目录加入 sys.path。
"""

import os, sys, time, json
from pathlib import Path

# ── 确保项目根目录在 sys.path 中（文件在 eval/ 子目录下）──
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

# 确保离线模式（与 main.py / run.py 保持一致）
for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    if not os.environ.get(_key):
        os.environ[_key] = "1"

from langchain_core.documents import Document
from config import get_llm


TEST_QUERIES = [
    ("续航怎么样", "多数用户认为续航不错，电池大可以用一整天，但充电速度一般"),
    ("屏幕清晰吗", "屏幕清晰度受到用户好评，OLED屏幕色彩鲜艳分辨率高"),
    ("拍照好不好", "拍照效果评价两极分化，有人觉得清晰色彩好有人觉得夜景一般"),
    ("信号稳定吗", "大部分用户认为信号接收稳定，通话质量没问题"),
    ("外观好看吗", "外观设计普遍得到正面评价颜色好看手感好但容易沾指纹"),
    ("值得买吗", "综合来看性价比高适合预算有限但对性能有一定要求的用户"),
    ("有什么缺点", "主要缺点是发热和续航，部分用户反映玩游戏时发热明显"),
    ("充电快吗", "充电速度一般从零充满大约需要一个多小时"),
    ("音质怎么样", "外放音质中规中矩日常使用足够但对音质要求高的用户可能不满意"),
    ("系统流畅吗", "系统运行流畅日常使用无明显卡顿多任务切换也顺畅"),
    ("像素高吗", "主摄像头像素高拍照清晰度令人满意"),
    ("电池耐用吗", "电池容量大正常使用一天没问题待机时间长"),
    ("手感如何", "手感舒适重量适中单手操作方便做工精良"),
    ("发热严重吗", "长时间玩游戏会发热日常使用温度正常"),
    ("适合送父母吗", "操作简单功能齐全性价比高适合作为老人机使用"),
    ("颜色有几种", "提供多种颜色选择黑色和蓝色最受欢迎"),
    ("会不会卡顿", "日常使用流畅不卡顿系统优化做得不错"),
    ("防水吗", "支持生活防水但不建议浸泡雨水溅到没问题"),
    ("玩游戏性能怎么样", "游戏性能中等日常手游流畅但大型3D游戏帧率偏低"),
    ("和竞品比有什么优势", "性价比是最大优势同价位配置更高适合预算敏感用户"),
]


def llm_judge_faithfulness(query, answer, contexts, llm):
    """LLM 评分: 答案的每个观点是否都能在检索到的上下文中找到依据"""
    ctx = "\n".join(f"[{i+1}] {c[:150]}" for i, c in enumerate(contexts[:5]))
    prompt = f"""你是严格的事实核查员。判断以下客服回答中的观点是否全部能在这5条用户评论中找到依据。

客服回答: {answer[:300]}

检索到的用户评论:
{ctx}

请评分 (0-10, 精确到1位小数):
- 10分: 所有观点都有明确依据，没有任何编造
- 7-9分: 大部分有依据，1-2个细节无直接支撑但不影响结论
- 4-6分: 部分有依据，有明显的推理或补充
- 0-3分: 大量编造内容，与检索结果无关

只回复一个数字（如 8.5）。"""
    try:
        resp = llm.invoke(prompt)
        text = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
        return float(text.split()[0])  # 取第一个数字
    except Exception:
        return -1


def llm_judge_relevancy(query, answer, llm):
    """LLM 评分: 答案是否直接回答了用户的问题"""
    prompt = f"""判断以下客服回答是否直接、完整地回答了用户问题。

用户问题: {query}
客服回答: {answer[:300]}

请评分 (0-10, 精确到1位小数):
- 10分: 直接回答问题，信息完整，没有偏题
- 7-9分: 基本回答了问题，略有冗余或遗漏
- 4-6分: 部分回答了问题，有较多无关内容或遗漏关键信息
- 0-3分: 答非所问，完全偏离问题

只回复一个数字（如 8.0）。"""
    try:
        resp = llm.invoke(prompt)
        text = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
        return float(text.split()[0])
    except Exception:
        return -1


def run_eval():
    print("[STEP] 初始化 RAG 管道...")
    from retrieval.store import load_vectorstore
    from pipeline.orchestrator import RAGPipeline
    from retrieval.reranker import Reranker
    from pipeline.router import IntentRouter

    vs = load_vectorstore()
    raw = vs.get(include=["metadatas", "documents"])
    bm25_chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    pipeline = RAGPipeline(vs, bm25_chunks, reranker=Reranker(), router=IntentRouter())
    judge_llm = get_llm(temperature=0)
    print("[OK] 就绪\n")

    faith_scores, relev_scores, latencies = [], [], []

    for i, (query, ref) in enumerate(TEST_QUERIES, 1):
        t0 = time.time()
        answer, docs = pipeline.query(query, auto_route=True)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        contexts = [d.page_content for d in docs]
        faith = llm_judge_faithfulness(query, answer, contexts, judge_llm)
        relev = llm_judge_relevancy(query, answer, judge_llm)

        if faith >= 0: faith_scores.append(faith)
        if relev >= 0: relev_scores.append(relev)

        f_str = f"{faith:.1f}" if faith >= 0 else "ERR"
        r_str = f"{relev:.1f}" if relev >= 0 else "ERR"
        print(f"  [{i:2d}] {query:16s}  faith={f_str}  relev={r_str}  "
              f"src={len(docs)}  {elapsed:.1f}s")

    # ── 报告 ──
    print("\n" + "=" * 55)
    print("[REPORT] RAG 评测报告 (LLM-as-Judge, DeepSeek)")
    print("=" * 55)

    if faith_scores:
        af = sum(faith_scores)/len(faith_scores)
        print(f"\n  Faithfulness (答案是否基于证据, 0-10)")
        print(f"    平均: {af:.1f}  |  最高: {max(faith_scores):.1f}  |  最低: {min(faith_scores):.1f}")

    if relev_scores:
        ar = sum(relev_scores)/len(relev_scores)
        print(f"\n  Answer Relevancy (答案切题度, 0-10)")
        print(f"    平均: {ar:.1f}  |  最高: {max(relev_scores):.1f}  |  最低: {min(relev_scores):.1f}")

    if latencies:
        latencies.sort()
        avg = sum(latencies)/len(latencies)
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)] if len(latencies)>1 else latencies[-1]
        print(f"\n  E2E 延迟")
        print(f"    平均: {avg:.1f}s  |  P50: {p50:.1f}s  |  P95: {p95:.1f}s")

    print(f"\n  +-- [RESUME] 简历可用 ────────────────")
    if faith_scores:
        print(f"  |  Faithfulness: {af:.1f}/10")
    if relev_scores:
        print(f"  |  Answer Relevancy: {ar:.1f}/10")
    if latencies:
        print(f"  |  P95 检索延迟: {p95:.1f}s  |  平均: {avg:.1f}s")
    print(f"  +───────────────────────────────────\n")


if __name__ == "__main__":
    run_eval()
