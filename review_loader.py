import json
import re
from pathlib import Path
from config import DATA_FILE,MIN_CHUNK_LENGTH,MAX_CHUNKS_PER_REVIEW
from langchain_core.documents import Document

def load_reviews(filepath: Path | None = None) -> list[dict]:
    if filepath is None:
        filepath = DATA_FILE
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            reviews = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"评论数据文件不存在: {filepath}，请检查 DATA_FILE 配置")
    except json.JSONDecodeError as e:
        raise ValueError(f"评论数据 JSON 格式错误: {filepath}，{e}")
    print(f"加载了 {len(reviews)} 条评论")
    return reviews


def split_into_opinions(text: str, min_length: int = MIN_CHUNK_LENGTH) -> list[str]:
    # 第1步：统一标点
    text = text.replace("!", "。").replace("?", "。").replace("～", "").replace("~", "")
    # 第2步：按强分隔符切第一轮
    sentences = re.split(r'[。！？\n]+', text)
    # 第3步：每句再按弱分隔符切第二轮
    opinions = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        parts = re.split(r'[，,；;、]+', sent)
        has_valid_part = False
        for part in parts:
            part = part.strip()
            if len(part) >= min_length:
                opinions.append(part)
                has_valid_part = True
        if not has_valid_part and len(sent) >= min_length:
            opinions.append(sent)
    # 第4步：整条评论切不出任何观点 → 整条作为一个观点
    if not opinions and len(text.strip()) >= min_length:
        opinions = [text.strip()]
    # 第5步：限制数量
    if len(opinions) > MAX_CHUNKS_PER_REVIEW:
        opinions = opinions[:MAX_CHUNKS_PER_REVIEW]
    return opinions


def build_opinion_chunks(
    reviews: list[dict] | None = None,
    sentiment_filter: str | None = None
) -> list[Document]:
    #防御性写法（负责兜底）
    if reviews is None:
        reviews = load_reviews()
    #当用户想要看自定的"sentiment": "positive" 或 "negative" 标签使用
    if sentiment_filter:
        reviews = [r for r in reviews if r['sentiment'] == sentiment_filter]
        print(f"过滤后保留 {len(reviews)} 条 {sentiment_filter} 评论")
    documents = []
    skipped = 0
    for review in reviews:
        opinions = split_into_opinions(review['text'])
        if not opinions:
            skipped += 1
            continue
        for opinion in opinions:
            doc = Document(
                page_content=opinion,
                metadata={
                    "review_id": review['id'],
                    "product": review['product'],
                    "rating": review['rating'],
                    "sentiment": review['sentiment'],
                    "source": f"review_{review['id']}"
                }
            )
            documents.append(doc)
    print(f"切出 {len(documents)} 个观点块（{skipped} 条评论因无有效内容跳过）")
    return documents

if __name__ == "__main__":
    #加载数据
    reviews = load_reviews()
    print("\n--- 观点分块示例 ---")
    sample = reviews[0]
    print(f"原文: {sample['text'][:100]}...")
    opinions = split_into_opinions(sample['text'])
    for i, op in enumerate(opinions, 1):
        print(f"  观点{i}: {op}")
    print("\n--- 全部评论分块 ---")
    chunks = build_opinion_chunks(reviews)
    pos_chunks = sum(1 for c in chunks if c.metadata['sentiment'] == 'positive')
    neg_chunks = sum(1 for c in chunks if c.metadata['sentiment'] == 'negative')
    avg = len(chunks) / len(reviews)
    print(f"\n分块统计:")
    print(f"  总chunk数: {len(chunks)}")
    print(f"  正面观点: {pos_chunks}")
    print(f"  负面观点: {neg_chunks}")
    print(f"  平均每条评论切出: {avg:.1f} 个观点块")