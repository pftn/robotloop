"""
pre_label_service.py
RobotLoop 云端预标注服务（模拟）
消费 Kafka 中的预标注任务，调用模拟 CV 模型生成预标注结果，写回数据库和对象存储。
"""

import json
import os
import requests
import time
import logging
from datetime import datetime

import boto3
import psycopg2
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv
from kafka.errors import KafkaTimeoutError

load_dotenv()

# -------------------- 配置 --------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "robotloop-data")

PG_HOST = os.getenv("PG_HOST", "postgres")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_USER = os.getenv("PG_USER", "robotloop")
PG_PASSWORD = os.getenv("PG_PASSWORD", "robotloop")
PG_DB = os.getenv("PG_DB", "robotloop")
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model-api:9002")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------- 初始化外部连接 --------------------
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


def get_db_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
    )


def download_media(bucket, key):
    """下载媒体文件到本地临时目录（模拟，实际可能直接读取 S3）"""
    local_path = f"/tmp/{key.split('/')[-1]}"
    s3.download_file(bucket, key, local_path)
    return local_path


def run_detection(bucket, key):
    """从 MinIO 下载图片，调用模型 API 进行目标检测"""
    local_path = f"/tmp/{key.split('/')[-1]}"
    s3.download_file(bucket, key, local_path)
    with open(local_path, "rb") as f:
        files = {"file": (os.path.basename(local_path), f, "image/jpeg")}
        try:
            resp = requests.post(
                f"{MODEL_API_URL}/detect",
                files=files,
                params={"model_version": MODEL_VERSION},
                timeout=30
            )
            if resp.status_code != 200:
                logging.error(f"Model API returned {resp.status_code}: {resp.text}")
                return []
            data = resp.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("detections", data.get("error", []))
            else:
                return []
        except Exception as e:
            logging.error(f"Failed to call Model API: {e}")
            return []


def save_prelabel_to_db(media_key, detections):
    """
    将预标注结果写入 PostgreSQL，更新 media_files 表的预标注字段。
    """
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        # 假设 media_files 表有 prelabel_result 字段（JSONB）
        cur.execute(
            "UPDATE media_files SET prelabel_result = %s, prelabeled_at = %s WHERE key = %s",
            (json.dumps(detections), datetime.utcnow(), media_key),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        logging.error(f"数据库写入失败: {e}")
        conn.rollback()
    finally:
        conn.close()


def send_to_human_review(media_key, detections):
    """
    发送人工审核消息到标注平台（这里用 Kafka 模拟）。
    """
    task = {
        "media_key": media_key,
        "detections": detections,
        "status": "pending_review",
        "created_at": datetime.utcnow().isoformat(),
    }
    producer.send("human-review-tasks", value=task)
    logging.info(f"人工审核任务已发送: {media_key}")


def main():
    logging.info("预标注服务启动，等待任务...")
    for msg in consumer:
        try:
            task = msg.value
            media_key = task["media_key"]
            bucket = task.get("bucket", S3_BUCKET)
            logging.info(f"收到任务: {media_key}")

            # 1. 执行推理
            # 只对图像文件调用模型 API
            if media_key.lower().endswith(('.jpg', '.jpeg', '.png')):
                detections = run_detection(bucket, media_key)
                # 过滤掉非字典元素，确保安全
                detections = [d for d in detections if isinstance(d, dict)]
                logging.info(f"推理完成，检测到 {len(detections)} 个目标")
            else:
                logging.info(f"非图像文件，跳过模型推理: {media_key}")
                detections = []

            # 2. 存储预标注结果
            save_prelabel_to_db(media_key, detections)

            # 3. 触发人工审核（如果置信度较低）
            low_conf = [d for d in detections if d.get("confidence", 1.0) < 0.85]
            if low_conf:
                send_to_human_review(media_key, detections)

        except Exception as e:
            logging.error(f"处理任务失败: {e}", exc_info=True)


if __name__ == "__main__":
    # Kafka 消费者（预标注任务）
    while True:
        try:
            consumer = KafkaConsumer(
                "pre-label-tasks",
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                group_id="prelabel-service",
                auto_offset_reset="earliest",
            )
            logging.info("Kafka Consumer 连接成功")
            break
        except (KafkaTimeoutError, Exception) as e:
            logging.warning(f"Kafka 未就绪，等待重试: {e}")
            time.sleep(2)

    # Kafka 生产者（用于通知下游标注平台）
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    main()
