"""ChromaDB 向量库 —— 建库 + 检索"""

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_chroma import Chroma
from config import CHROMA_DIR, SEARCH_K, get_embeddings
from data.loader import load_reviews, build_opinion_chunks

def build_vectorstore(chunks, collection_name: str = "phone_reviews"):
    print(f"--- 开始建向量库（{len(chunks)} 个观点块）---")
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=collection_name
    )
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  已写入 {min(i + batch_size, len(chunks))}/{len(chunks)}")
    print(f"--- 向量库完成！共 {len(chunks)} 个向量 ---")
    return vectorstore


def load_vectorstore(collection_name: str = "phone_reviews") -> Chroma:
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=collection_name
    )
    count = vectorstore._collection.count()
    print(f"加载已有向量库：{count} 个向量")
    return vectorstore


def search(query: str, vectorstore: Chroma, k: int = SEARCH_K) -> list:
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    #获取数据
    reviews = load_reviews()
    #将数据分成chunks
    chunks = build_opinion_chunks(reviews)
    #将chunks变成向量
    vs = build_vectorstore(chunks)
    print("\n--- 测试检索 ---")
    queries = ["续航怎么样", "屏幕清晰吗", "拍照效果好不好"]
    for q in queries:
        #相似度匹配
        results = search(q, vs, k=3)
        print(f"\n查询: {q}")
        for i, doc in enumerate(results, 1):
            sentiment_tag = "👍" if doc.metadata['sentiment'] == 'positive' else "👎"
            print(f"  {i}. {sentiment_tag} [{doc.metadata['rating']}星] {doc.page_content[:80]}")
