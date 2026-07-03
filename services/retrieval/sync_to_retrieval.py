#!/usr/bin/env python3
"""同步已标注媒体数据到 ES 和 Chroma"""
import os, sys, json, logging, io
from datetime import datetime
import psycopg2, psycopg2.extras
import boto3
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from PIL import Image
import chromadb
from chromadb.config import Settings

# ---------- 配置 ----------
PG_HOST = os.getenv("PG_HOST", "postgres")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "robotloop")
PG_PASSWORD = os.getenv("PG_PASSWORD", "robotloop")
PG_DB = os.getenv("PG_DB", "robotloop")

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "robotloop-data")

ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
ES_INDEX = "robotloop_media"

CHROMA_PERSIST_DIR = "/data/chroma"
COLLECTION_NAME = "media_embeddings"

MODEL_NAME = "clip-ViT-B-32"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT,
                  aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY)
model = SentenceTransformer(MODEL_NAME)


def fetch_media_from_pg():
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT key, bucket, prelabel_result, uploaded_at FROM media_files WHERE prelabel_result IS NOT NULL")
        return cur.fetchall()


def build_label_text(prelabel_result):
    if not prelabel_result: return ""
    labels = set(d.get("class", "") for d in prelabel_result if d.get("class"))
    return " ".join(sorted(labels))


def download_image(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
    except:
        return None


def create_es_index(es):
    mapping = {"mappings": {
        "properties": {"key": {"type": "keyword"}, "bucket": {"type": "keyword"}, "labels": {"type": "text"},
                       "uploaded_at": {"type": "date"}}}}
    if not es.indices.exists(index=ES_INDEX): es.indices.create(index=ES_INDEX, body=mapping)


def index_to_es(es, records, label_texts):
    actions = [{"_index": ES_INDEX, "_id": r["key"], "_source": {"key": r["key"], "bucket": r["bucket"], "labels": lt,
                                                                 "uploaded_at": r[
                                                                     "uploaded_at"].isoformat() if isinstance(
                                                                     r["uploaded_at"], datetime) else r["uploaded_at"]}}
               for r, lt in zip(records, label_texts)]
    helpers.bulk(es, actions)
    logger.info(f"ES indexed {len(actions)} docs")


def main():
    records = fetch_media_from_pg()
    if not records: sys.exit(0)

    label_texts = [build_label_text(r["prelabel_result"]) for r in records]
    es = Elasticsearch(hosts=ES_HOST)
    create_es_index(es)
    index_to_es(es, records, label_texts)

    # 生成 Embedding（图像模式）
    valid_records, embeddings, valid_label_texts = [], [], []
    for rec, lt in zip(records, label_texts):
        img = download_image(rec["bucket"], rec["key"])
        if img:
            emb = model.encode(img)
            valid_records.append(rec)
            embeddings.append(emb)
            valid_label_texts.append(lt)

    if not valid_records: sys.exit(1)

    # 存入 Chroma（自动持久化）
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    ids = [r["key"] for r in valid_records]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=[{"labels": lt} for lt in valid_label_texts]
    )
    logger.info(f"Chroma indexed {len(ids)} vectors")


if __name__ == "__main__":
    main()
