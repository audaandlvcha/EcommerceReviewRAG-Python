"""
RAGAS 标准化评测 — 五项指标

指标:
  - Faithfulness:       答案是否基于检索证据（不编造）
  - Answer Relevancy:   答案是否直接回答了问题
  - Context Precision:  检索到的文档中有多少是真正有用的
  - Context Recall:     应该检索到的文档有多少被检索到了
  - Answer Correctness: 答案的事实正确性（需要 reference）

用法: python eval/ragas_eval.py（从项目根目录运行）
"""

import os, sys, time, json
from pathlib import Path
from types import ModuleType

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 兼容性修复
_m = ModuleType("langchain_community.chat_models.vertexai")
_m.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models", ModuleType("langchain_community.chat_models"))
sys.modules["langchain_community.chat_models.vertexai"] = _m

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    if not os.environ.get(_key):
        os.environ[_key] = "1"

from langchain_core.documents import Document

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


def run_ragas_eval(label="v2 (BGE Reranker)"):
    from retrieval.store import load_vectorstore
    from pipeline.orchestrator import RAGPipeline
    from retrieval.reranker import Reranker
    from pipeline.router import IntentRouter
    from config import get_llm, get_embeddings, DEEPSEEK_MODEL

    print(f"\n{'='*60}")
    print(f"[RAGAS] Evaluation: {label}")
    print(f"{'='*60}")

    vs = load_vectorstore()
    raw = vs.get(include=["metadatas", "documents"])
    bm25_chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    pipeline = RAGPipeline(vs, bm25_chunks, reranker=Reranker(), router=IntentRouter())

    # 逐条跑 RAG 管道
    results_raw = []
    latencies = []

    for i, (query, ref_answer) in enumerate(TEST_QUERIES, 1):
        t0 = time.time()
        answer, docs = pipeline.query(query, auto_route=True)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        contexts = [d.page_content for d in docs]
        results_raw.append({
            "user_input": query,
            "reference": ref_answer,
            "response": answer,
            "retrieved_contexts": contexts,
            "latency": round(elapsed, 2),
        })
        print(f"  [{i:2d}] {query:16s}  {elapsed:.1f}s  ctx={len(contexts)}")

    # RAGAS 评测
    print(f"\n[RAGAS] Scoring with {DEEPSEEK_MODEL}...")

    evaluator_llm = get_llm(temperature=0)

    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm_wrapper = LangchainLLMWrapper(evaluator_llm)
        emb_wrapper = LangchainEmbeddingsWrapper(get_embeddings())

        dataset = EvaluationDataset.from_list(results_raw)

        # 按需导入指标（版本兼容）
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm_wrapper,
            embeddings=emb_wrapper,
        )

        scores = {}
        for m in metrics:
            try:
                scores[m.name] = float(result[m.name])
            except (KeyError, TypeError):
                scores[m.name] = None

    except Exception as e:
        print(f"  [WARN] RAGAS full API failed: {e}")
        print(f"  [FALLBACK] Running 3-metric subset...")

        # Fallback: 只用 3 个不需要 reference/embedding 复杂设置的指标
        from ragas import evaluate, EvaluationDataset
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        all_scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}

        for r in results_raw:
            try:
                ds = EvaluationDataset.from_list([{
                    "user_input": r["user_input"],
                    "response": r["response"],
                    "retrieved_contexts": r["retrieved_contexts"],
                }])
                sr = evaluate(
                    dataset=ds,
                    metrics=[faithfulness, answer_relevancy, context_precision],
                    llm=LangchainLLMWrapper(evaluator_llm),
                )
                all_scores["faithfulness"].append(float(sr["faithfulness"]))
                all_scores["answer_relevancy"].append(float(sr["answer_relevancy"]))
                all_scores["context_precision"].append(float(sr["context_precision"]))
            except Exception as e2:
                print(f"    [SKIP] {r['user_input']}: {e2}")

        scores = {
            k: round(sum(v)/len(v), 3) if v else None
            for k, v in all_scores.items()
        }

    # 延迟统计
    latencies.sort()
    avg_lat = sum(latencies) / len(latencies)
    p95 = latencies[int(len(latencies)*0.95)] if len(latencies) > 1 else latencies[-1]

    # 报告
    print(f"\n{'='*60}")
    print(f"[RAGAS REPORT] {label}")
    print(f"{'='*60}")

    metric_labels = {
        "faithfulness":        "Faithfulness",
        "answer_relevancy":    "Answer Relevancy",
        "context_precision":   "Context Precision",
        "context_recall":      "Context Recall",
        "answer_correctness":  "Answer Correctness",
    }

    for key, label in metric_labels.items():
        if key in scores and scores[key] is not None:
            print(f"  {label:25s}  {scores[key]:.3f}")

    print(f"\n  {'E2E Avg Latency':25s}  {avg_lat:.1f}s")
    print(f"  {'E2E P95 Latency':25s}  {p95:.1f}s")

    # 简历可用
    print(f"\n  +-- [RESUME] -----------------------------------")
    for key, label in metric_labels.items():
        if key in scores and scores[key] is not None:
            print(f"  |  {label}: {scores[key]:.2f}")
    print(f"  |  P95 Latency: {p95:.1f}s  |  Avg: {avg_lat:.1f}s")
    print(f"  +-----------------------------------------------\n")

    return {
        "label": label,
        "scores": scores,
        "avg_latency": round(avg_lat, 1),
        "p95_latency": round(p95, 1),
    }


if __name__ == "__main__":
    result = run_ragas_eval(label="v2 (BGE Reranker)")

    output_file = _PROJECT_ROOT / "eval" / "ragas_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "label": result["label"],
            "scores": result["scores"],
            "avg_latency": result["avg_latency"],
            "p95_latency": result["p95_latency"],
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {output_file}")
