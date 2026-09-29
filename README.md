# 电商评论分析 RAG

基于真实用户评论的检索增强生成系统 — 不是搜文档，是搜人说的话。

## 一句话定位

用户问"续航怎么样"→ 从 2000+ 条手机评论中检索最相关的真实观点 → LLM 汇总成自然语言答案。

和知识问答 RAG 的本质区别：分块策略按**观点**而非字符数，检索对象是**用户口语**而非教科书文本。

## 检索链路

```
用户问题: "续航怎么样"
  │
  ├─ ① IntentRouter: LLM 判断复杂度 → 决定开哪些增强（simple/medium/complex）
  ├─ ② Conversation: 多轮对话检测 → 追问自动补全关键词
  ├─ ③ MultiQuery: [可选] LLM 从 3 个角度重写查询（参数/体验/优缺点）
  ├─ ④ HyDE: [可选] LLM 生成假想评论 → 用假评论的向量去检索
  ├─ ⑤ Hybrid: BM25 关键词 + 向量语义 → RRF 融合（召回 Top-20）
  ├─ ⑥ Cross-Encoder: ms-marco-MiniLM-L-2-v2 逐对精排（20→5）
  ├─ ⑦ LLM 汇总: DeepSeek-chat 基于 5 条精选评论流式生成答案
  └─ 🔍 Langfuse: 全链路追踪（可选）
```

## 技术栈

| 层 | 技术 |
|----|------|
| LLM | DeepSeek API (deepseek-chat) |
| Embedding | HuggingFace BAAI/bge-small-zh-v1.5（本地推理） |
| 向量库 | ChromaDB |
| 关键词检索 | BM25（rank-bm25） |
| 精排 | cross-encoder/ms-marco-MiniLM-L-2-v2（Cross-Encoder） |
| 编排 | LangChain LCEL |
| API | FastAPI + SSE 流式 |
| 评估 | LLM-as-Judge（Faithfulness / Answer Relevancy 自研） |
| 追踪 | Langfuse（全链路可观测） |
| 数据 | ChineseNlpCorpus online_shopping_10_cats（2323 条手机评论） |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env（DeepSeek API Key + Langfuse 可选）
# DEEPSEEK_API_KEY=sk-xxx
# LANGFUSE_PUBLIC_KEY=pk-xxx  （可选）
# LANGFUSE_SECRET_KEY=sk-xxx   （可选）

# 3. 首次运行需下载 Embedding 模型（自动），或设置 HF_HUB_OFFLINE=1 使用本地缓存

# 4. 构建向量库
python -m retrieval.store

# 5. 启动
python main.py

# 6. 浏览器打开
http://localhost:8000
```

## 项目结构

```
EcommerceReviewRAG_Plus/
├── main.py                  # 入口（开发模式，热重载）
├── run.py                   # 入口（生产模式）
├── config.py                # 全局配置 + LLM/Embedding 工厂
├── langfuse_setup.py        # Langfuse 全链路追踪
├── web.py                   # FastAPI + SSE 真正流式
├── requirements.txt
├── .env                     # API Key 配置
├── data/
│   ├── loader.py            # 数据加载 + 观点级分块
│   └── reviews.json         # 2323 条手机评论
├── retrieval/
│   ├── store.py             # ChromaDB 建库/检索
│   ├── hyde.py              # HyDE：LLM生成假评论→向量检索
│   ├── multi_query.py       # MultiQuery：LLM多角度扩展
│   ├── hybrid.py            # BM25+向量混合检索 + RRF融合
│   └── reranker.py          # Cross-Encoder 精排
├── pipeline/
│   ├── router.py            # 意图路由：按复杂度分级
│   └── orchestrator.py      # 全链路编排
├── conversation/
│   └── manager.py           # 多轮对话 + 追问检测
├── eval/
│   ├── quality.py           # LLM-as-Judge 答案质量评估
│   └── recall.py            # Recall@5 多配置对比
└── static/
    └── index.html           # 前端页面
```

## 和通用 RAG 的区别

| 维度 | 通用 RAG（知识问答） | 本项目（评论分析） |
|------|-------------------|------------------|
| 分块策略 | 按字符数切（RecursiveCharacterTextSplitter） | 按标点切观点（一条观点=一个chunk） |
| 检索对象 | 教科书/文档 | 用户口语评论 |
| 查询增强 | 无/简单 | HyDE + MultiQuery 双重 LLM 增强 |
| 检索方式 | 单路向量 | BM25 + 向量 → RRF → Cross-Encoder 两阶段 |
| 评估 | 无/手动 | LLM-as-Judge 自研（Faithfulness + Answer Relevancy + Recall） |
| 核心挑战 | 知识覆盖面 | 口语短 query ↔ 长评论的语义鸿沟 |

## 关键技术决策

**1. 为什么观点分块而不是字符分块？**
"屏幕好，续航差"→ 按字符切会把两个相反观点塞进同一个chunk，检索时语义混淆。按标点切成独立观点，每条chunk语义纯净。

**2. 为什么 RRF 融合后还要 Cross-Encoder？**
RRF 管召回（不错过），Cross-Encoder 管精度（不混入）。纯 RRF 结果混入了"菜单切换有延时"这种跑题结果，Cross-Encoder 逐字比对后能踢出去。

**3. 为什么用 DeepSeek API 而不是本地 Ollama？**
DeepSeek API 推理质量更高，中文理解好，无需本地 GPU 资源。通过 API 调用模式更适合生产环境部署。

**4. 为什么评估脚本不挂在主链路？**
LLM-as-Judge 每次评估要调多次 LLM 当评委，比检索本身还慢。正确的用法是离线跑测试集，把平均分数写在 README 里。

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
