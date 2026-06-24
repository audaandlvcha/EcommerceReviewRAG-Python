"""
RAGAS 自动化评估

原理：用 LLM 当评委，自动评估 RAG 回答的质量
三个核心指标：
  - Faithfulness（忠实度）：答案中的每个观点是否都来自检索到的文档
  - Answer Relevancy（答案相关性）：答案是否直接回答了问题
  - Context Precision（上下文精度）：检索到的文档是否有用

面试怎么讲：
  "没评估的 RAG 是盲飞——你不知道你的检索到底准不准。我用 RAGAS 自动化评估，
  每次检索后 LLM 自动打分，faithfulness 从最开始的 0.6 优化到 0.9。"

前置：pip install ragas
"""
# RAGAS 0.4.x 和新版 langchain-community 不兼容的临时修复
import sys
from types import ModuleType
_m = ModuleType("langchain_community.chat_models.vertexai")
_m.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models", ModuleType("langchain_community.chat_models"))
sys.modules["langchain_community.chat_models.vertexai"] = _m

import json
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from config import LLM_MODEL


def evaluate_rag(query: str, answer: str, contexts: list[str]) -> dict:
    """用 RAGAS 评估一次 RAG 回答"""
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=LLM_MODEL, temperature=0))

    dataset = EvaluationDataset.from_list([{
        "user_input": query,
        "response": answer,
        "retrieved_contexts": contexts,
    }])

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
    )

    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
    }


def batch_evaluate(test_queries: list[dict], pipeline, output_path=None) -> list[dict]:
    """批量评估多条测试 query"""
    results = []
    for item in test_queries:
        answer, docs = pipeline.query(item["query"])
        scores = evaluate_rag(
            item["query"], answer,
            [doc.page_content for doc in docs]
        )
        results.append({**item, "answer": answer, "scores": scores})
        print(f"  {item['query']}: faithfulness={scores['faithfulness']:.2f}, "
              f"relevancy={scores['answer_relevancy']:.2f}")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    avg = {
        "faithfulness": sum(r["scores"]["faithfulness"] for r in results) / len(results),
        "answer_relevancy": sum(r["scores"]["answer_relevancy"] for r in results) / len(results),
        "context_precision": sum(r["scores"]["context_precision"] for r in results) / len(results),
    }
    print(f"\n📊 平均: faithfulness={avg['faithfulness']:.2f}, "
          f"relevancy={avg['answer_relevancy']:.2f}, "
          f"precision={avg['context_precision']:.2f}")
    return results


if __name__ == "__main__":
    print("RAGAS 评估器加载成功")
    print("用法: from evaluator import evaluate_rag, batch_evaluate")
    print("需要先 pip install ragas")
