import os
import io
import logging
import requests
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from PIL import Image
import chromadb
from chromadb.config import Settings

# ---------- 配置 ----------
MODEL_NAME = "clip-ViT-B-32"
ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
ES_INDEX = "robotloop_media"
CHROMA_PERSIST_DIR = "/data/chroma"
COLLECTION_NAME = "media_embeddings"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search-api")

# ---------- 初始化 ES ----------
es = Elasticsearch(hosts=ES_HOST)

# ---------- 初始化 Chroma ----------
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# 尝试获取集合，若不存在则创建空集合（防止启动崩溃）
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    logger.info(f"Chroma collection '{COLLECTION_NAME}' loaded, count: {collection.count()}")
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    logger.info(f"Chroma collection '{COLLECTION_NAME}' created (empty)")

# ---------- 加载多模态模型 ----------
logger.info(f"Loading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
logger.info("Model loaded")

# ---------- FastAPI 应用 ----------
app = FastAPI(title="RobotLoop Multi-modal Search API", version="2.1")


@app.get("/search")
async def search(
        labels: Optional[str] = Query(None, description="逗号分隔的标签关键词"),
        weather: Optional[str] = Query(None, description="天气条件（预留字段）"),
        text_query: Optional[str] = Query(None, description="用自然语言描述搜索场景"),
        image_url: Optional[str] = Query(None, description="用于以图搜图的图片 URL"),
        top_k: int = Query(10, ge=1, le=100),
):
    """
    多模态联合检索接口：
    1. 通过 ES 过滤标签等条件，获得候选集。
    2. 若提供 text_query 或 image_url，则在候选集内进行向量相似度搜索。
    3. 返回最终结果（包含文件路径和元数据）。
    """
    # Step 1：从 ES 获取候选 ID
    must = []
    if labels:
        for label in labels.split(","):
            must.append({"match": {"labels": label.strip()}})
    if weather:
        must.append({"term": {"weather": weather}})

    # 检查索引是否存在
    if not es.indices.exists(index=ES_INDEX):
        return {"count": 0, "results": []}

    if must:
        es_query = {"query": {"bool": {"must": must}}, "_source": False, "size": 1000}
        resp = es.search(index=ES_INDEX, body=es_query)
        candidate_ids = {hit["_id"] for hit in resp["hits"]["hits"]}
    else:
        resp = es.search(index=ES_INDEX, body={"query": {"match_all": {}}, "_source": False, "size": 1000})
        candidate_ids = {hit["_id"] for hit in resp["hits"]["hits"]}

    if not candidate_ids:
        return {"count": 0, "results": []}

    # Step 2：向量检索（如果有语义查询条件）
    final_ids = set(candidate_ids)
    if text_query or image_url:
        # 检查集合是否为空
        if collection.count() == 0:
            final_ids = set()  # 无向量数据，返回空结果
        else:
            # 生成查询向量
            if text_query:
                query_emb = model.encode(text_query).tolist()
            else:
                try:
                    img_data = requests.get(image_url, timeout=10).content
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    query_emb = model.encode(img).tolist()
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"图片下载或处理失败: {e}")

            # 在候选 ID 范围内搜索（Chroma 查询后筛选）
            results = collection.query(query_embeddings=[query_emb], n_results=top_k * 5)
            final_ids = set()
            for ids, metas in zip(results["ids"], results["metadatas"]):
                for id_, meta in zip(ids, metas):
                    if id_ in candidate_ids:
                        final_ids.add(id_)
                        if len(final_ids) >= top_k:
                            break
                if len(final_ids) >= top_k:
                    break
            logger.info(f"向量检索后剩余 ID 数量: {len(final_ids)}")

    # Step 3：根据最终 ID 从 ES 获取完整元数据
    if final_ids:
        es_res = es.mget(index=ES_INDEX, body={"ids": list(final_ids)})
        docs = [doc["_source"] for doc in es_res["docs"] if doc["found"]]
    else:
        docs = []

    return {"count": len(docs), "results": docs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
