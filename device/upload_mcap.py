#!/usr/bin/env python3
"""设备端 MCAP/rosbag2 上传脚本（真实数据入口）。

    .mcap/.bag 文件 --上传--> MinIO robotloop-raw/raw/ --bucket notification--> Kafka
    --> Ray Worker 按扩展名分流解析（见 cloud/ray_worker/ray_pipeline.py）

Kafka 事件不需要本脚本手动发：v1 的 MinIO 已配置 bucket notification
（docker-compose.yml 的 minio-init: mc event add ... --event put），
对象一落盘就自动触发 raw-data-ingest 事件。

采集元数据（本体/任务/语言指令/来源）随对象metadata一起上传（x-amz-meta-*），
Ray Worker 解析时读取，免维护额外的清单文件。

用法：
    python upload_mcap.py --dir /data/bags --embodiment aloha \
        --task pick_red_cube --instruction "pick up the red cube" --source teleop
"""

import argparse
import logging
import os

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("upload-mcap")

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_RAW_BUCKET = os.getenv("S3_RAW_BUCKET", "robotloop-raw")

BAG_EXTENSIONS = (".mcap", ".bag")


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def upload_bags(
    src_dir: str,
    embodiment: str,
    task: str,
    instruction: str,
    source: str = "teleop",
    bucket: str = S3_RAW_BUCKET,
    prefix: str = "raw",
    camera_topic: str = "/top",
    joint_topic: str = "/joint_states",
    success: str = "",
) -> int:
    """把目录下所有 .mcap/.bag 上传到 MinIO，返回上传数量。"""
    s3 = make_s3()
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
        logger.info("Bucket %s created", bucket)

    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(BAG_EXTENSIONS))
    if not files:
        logger.warning("no .mcap/.bag files under %s", src_dir)
        return 0

    metadata = {
        "embodiment-tag": embodiment,
        "task": task,
        "language-instruction": instruction,
        "source": source,
        "camera-topic": camera_topic,
        "joint-topic": joint_topic,
    }
    if success:  # 未指定则不携带：Iceberg 里 success 为 NULL（未标注）
        metadata["success"] = success
    for name in files:
        key = f"{prefix.strip('/')}/{name}"
        path = os.path.join(src_dir, name)
        s3.upload_file(path, bucket, key, ExtraArgs={"Metadata": metadata})
        logger.info(
            "uploaded %s -> s3://%s/%s (Kafka event via bucket notification)",
            name,
            bucket,
            key,
        )
    logger.info(
        "done: %d bag(s) uploaded. MinIO notification -> Kafka raw-data-ingest -> Ray pipeline.",
        len(files),
    )
    return len(files)


def main():
    ap = argparse.ArgumentParser(description="上传 MCAP/rosbag2 到 RobotLoop MinIO")
    ap.add_argument("--dir", required=True, help="本地包目录")
    ap.add_argument(
        "--embodiment", required=True, help="本体标签，如 aloha / franka / agibot_g1"
    )
    ap.add_argument("--task", required=True, help="任务名，如 pick_red_cube")
    ap.add_argument("--instruction", required=True, help="语言指令（语义检索文本）")
    ap.add_argument("--source", default="teleop", choices=["teleop", "sim", "real"])
    ap.add_argument(
        "--success",
        default="",
        choices=["", "true", "false"],
        help="成功标注；不指定则 Episode.success=NULL（未标注），"
        "检索端 success=true/false 过滤不会命中",
    )
    ap.add_argument("--bucket", default=S3_RAW_BUCKET)
    ap.add_argument("--prefix", default="raw", help="bucket 内前缀，默认 raw/")
    ap.add_argument("--camera-topic", default="/top")
    ap.add_argument("--joint-topic", default="/joint_states")
    args = ap.parse_args()
    upload_bags(
        src_dir=args.dir,
        embodiment=args.embodiment,
        task=args.task,
        instruction=args.instruction,
        source=args.source,
        bucket=args.bucket,
        prefix=args.prefix,
        camera_topic=args.camera_topic,
        joint_topic=args.joint_topic,
        success=args.success,
    )


if __name__ == "__main__":
    main()
