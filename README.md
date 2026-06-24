# 电商评论分析 RAG

基于真实用户评论的检索增强生成系统 — 不是搜文档，是搜人说的话。

## 一句话定位

用户问"续航怎么样"→ 从 2000+ 条手机评论中检索最相关的真实观点 → LLM 汇总成自然语言答案。

和知识问答 RAG 的本质区别：分块策略按**观点**而非字符数，检索对象是**用户口语**而非教科书文本。

## 检索链路

```
用户问题: "续航怎么样"
  │
  ├─ ① MultiQuery: LLM 从 3 个角度重写查询（参数/体验/优缺点）
  ├─ ② HyDE: LLM 生成假想评论 → 用假评论的向量去检索
  ├─ ③ Hybrid: BM25 关键词 + 向量语义 → RRF 融合（召回 Top-20）
  ├─ ④ Cross-Encoder: BAAI/bge-reranker-base 逐对精排（20→5）
  └─ ⑤ LLM 汇总: 基于 5 条精选评论生成自然语言答案
```

## 技术栈

| 层 | 技术 |
|----|------|
| LLM | Ollama + qwen3:8b（本地推理） |
| Embedding | Ollama + nomic-embed-text |
| 向量库 | ChromaDB |
| 关键词检索 | BM25（rank-bm25） |
| 精排 | BAAI/bge-reranker-base（Cross-Encoder） |
| 编排 | LangChain LCEL |
| API | FastAPI + SSE 流式 |
| 评估 | RAGAS（Faithfulness / Answer Relevancy / Context Precision） |
| 数据 | ChineseNlpCorpus online_shopping_10_cats（2323 条手机评论） |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 确保 Ollama 模型已安装
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 3. 启动
python main.py

# 4. 浏览器打开
http://localhost:8000
```

## 项目结构

```
EcommerceReviewRAG/
├── main.py              # 入口
├── config.py            # 配置集中
├── review_loader.py     # 数据加载 + 观点级分块
├── vector_store.py      # ChromaDB 建库/检索
├── hyde_retriever.py    # HyDE：LLM生成假评论→向量检索
├── multi_query.py       # MultiQuery：LLM多角度扩展
├── hybrid_retriever.py  # BM25+向量混合检索 + RRF融合
├── reranker.py          # Cross-Encoder 精排
├── rag_pipeline.py      # 全链路组装
├── evaluator.py         # RAGAS 自动化评估
├── web.py               # FastAPI + SSE 流式
├── static/
│   └── index.html       # 前端页面
└── data/
    └── reviews.json     # 2323 条手机评论
```

## 和通用 RAG 的区别

| 维度 | 通用 RAG（知识问答） | 本项目（评论分析） |
|------|-------------------|------------------|
| 分块策略 | 按字符数切（RecursiveCharacterTextSplitter） | 按标点切观点（一条观点=一个chunk） |
| 检索对象 | 教科书/文档 | 用户口语评论 |
| 查询增强 | 无/简单 | HyDE + MultiQuery 双重 LLM 增强 |
| 检索方式 | 单路向量 | BM25 + 向量 → RRF → Cross-Encoder 两阶段 |
| 评估 | 无/手动 | RAGAS 自动化三指标 |
| 核心挑战 | 知识覆盖面 | 口语短 query ↔ 长评论的语义鸿沟 |

## 关键技术决策

**1. 为什么观点分块而不是字符分块？**
"屏幕好，续航差"→ 按字符切会把两个相反观点塞进同一个chunk，检索时语义混淆。按标点切成独立观点，每条chunk语义纯净。

**2. 为什么 RRF 融合后还要 Cross-Encoder？**
RRF 管召回（不错过），Cross-Encoder 管精度（不混入）。纯 RRF 结果混入了"菜单切换有延时"这种跑题结果，Cross-Encoder 逐字比对后能踢出去。

**3. 为什么用 qwen3:8b 而不是 deepseek-r1:1.5b？**
1.5b 对 few-shot 示例会照抄模板（问"定位"输出"屏幕"相关），换成 8b + 规则指令后完全纠正。

**4. 为什么 evaluator 不挂在主链路？**
RAGAS 每次评估要调多次 LLM 当评委，比检索本身还慢。正确的用法是离线跑测试集，把平均分数写在 README 里。

## API

```bash
# REST 检索
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "续航怎么样"}'

# SSE 流式
curl http://localhost:8000/api/search/stream?query=续航怎么样
```

## License

MIT
