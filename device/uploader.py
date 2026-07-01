import json
import logging
import os
import random
import time

import boto3
from kafka import KafkaProducer, KafkaAdminClient
from kafka.errors import KafkaTimeoutError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "robotloop-data")
SOURCE_DIR = "/tmp/robot_data"
STATE_FILE = "/tmp/upload_state.json"

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

# 配置 Kafka
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")


def send_upload_event(key, bucket):
    event = {
        "key": key,
        "bucket": bucket,
        "timestamp": time.time()
    }
    producer.send("file-upload-events", value=event)
    producer.flush()
    logging.info(f"已发送上传事件: {key}")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def upload_file(file_path, key):
    file_size = os.path.getsize(file_path)
    chunk_size = 5 * 1024 * 1024
    state = load_state().get(key, {})
    upload_id = state.get("upload_id")
    parts = state.get("parts", [])

    if not upload_id:
        resp = s3.create_multipart_upload(Bucket=S3_BUCKET, Key=key)
        upload_id = resp["UploadId"]
        logging.info(f"开始上传 {key}，UploadId: {upload_id}")

    part_number = 1
    with open(file_path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            if any(p["PartNumber"] == part_number for p in parts):
                part_number += 1
                continue
            if os.getenv("SIMULATE_NETWORK_ISSUES", "false").lower() == "true" and random.random() < 0.2:
                logging.warning(f"模拟网络中断，part {part_number} 上传失败")
                break
            resp = s3.upload_part(
                Bucket=S3_BUCKET, Key=key,
                UploadId=upload_id, PartNumber=part_number,
                Body=data
            )
            parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
            save_state({key: {"upload_id": upload_id, "parts": parts}})
            logging.info(f"已上传 part {part_number}")
            part_number += 1

    total_parts = (file_size + chunk_size - 1) // chunk_size
    if len(parts) == total_parts:
        s3.complete_multipart_upload(
            Bucket=S3_BUCKET, Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts}
        )
        logging.info(f"文件 {key} 上传完成")
        send_upload_event(key, S3_BUCKET)
        # 清除状态
        state = load_state()
        if key in state:
            del state[key]
            save_state(state)
    else:
        logging.info(f"文件 {key} 上传中断，下次继续")


def wait_for_topic(bootstrap, topic, timeout=60):
    """等待指定 topic 在 Kafka 中存在，超时抛出异常"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=5000)
            metadata = admin.list_topics()
            if topic in metadata:
                admin.close()
                logging.info(f"Topic {topic} 已存在")
                return
            else:
                logging.warning(f"Topic {topic} 不存在，等待中...")
            admin.close()
        except Exception as e:
            logging.warning(f"连接 Kafka 失败，等待重试: {e}")
        time.sleep(2)
    raise Exception(f"等待 topic {topic} 超时")


def main():
    if not os.path.exists(SOURCE_DIR):
        logging.error(f"数据目录 {SOURCE_DIR} 不存在，请先运行 generator.py")
        return
    wait_for_topic(KAFKA_BOOTSTRAP, "file-upload-events")
    manifest = json.load(open(os.path.join(SOURCE_DIR, "manifest.json")))
    logging.info(f"开始上传 {len(manifest)} 个文件...")
    for filename in manifest:
        file_path = os.path.join(SOURCE_DIR, filename)
        if os.path.exists(file_path):
            upload_file(file_path, filename)
        else:
            logging.warning(f"文件 {filename} 不存在，跳过")
    logging.info("所有文件上传流程结束")


if __name__ == "__main__":
    # 创建 Kafka Producer，自动等待 Kafka 就绪
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=5,
                max_block_ms=5000
            )
            break
        except KafkaTimeoutError:
            logging.warning("Kafka 未就绪，重试中...")
            time.sleep(2)
    main()
