import os
import json
import logging
import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "robotloop-data")
SOURCE_DIR = "/tmp/robot_data"

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


def ensure_bucket(bucket):
    try:
        s3.head_bucket(Bucket=bucket)
    except:
        s3.create_bucket(Bucket=bucket)
        logging.info(f"Bucket {bucket} created")


def upload_file(file_path, key):
    """上传单个文件到 MinIO"""
    s3_key = f"{key}"
    with open(file_path, "rb") as f:
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=f)
    logging.info(f"Uploaded {s3_key}")


def main():
    ensure_bucket(S3_BUCKET)
    manifest_path = os.path.join(SOURCE_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        logging.error("manifest.json not found, run generator first")
        return

    manifest = json.load(open(manifest_path))
    for filename in manifest:
        file_path = os.path.join(SOURCE_DIR, filename)
        if os.path.exists(file_path):
            upload_file(file_path, filename)

    logging.info(
        "All files uploaded. MinIO bucket notification will trigger downstream processing."
    )


if __name__ == "__main__":
    main()
