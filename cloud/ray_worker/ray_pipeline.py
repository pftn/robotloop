#!/usr/bin/env python3

import boto3
import json
import lancedb
import logging
import numpy as np
import os
import pyarrow as pa
import ray
import requests
import time
import traceback
from datetime import datetime
from functools import wraps
from kafka import KafkaConsumer
from pyiceberg.catalog import load_catalog
from pyiceberg.partitioning import PartitionSpec, PartitionField, HourTransform
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, TimestampType, FloatType
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


setup_logging()
logger = logging.getLogger("ray-pipeline")

# ==================== 环境变量 ====================
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "robotloop-data")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model-api:9002")
LANCE_DB_PATH = os.getenv("LANCE_PATH", "/tmp/lance")
TABLE_NAME = "robotloop_scenes"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")

ICEBERG_WAREHOUSE = os.getenv("ICEBERG_WAREHOUSE", f"s3://iceberg-warehouse")
ICEBERG_TABLE = os.getenv("ICEBERG_TABLE", "robotloop.scenes_meta")
ICEBERG_CATALOG_URI = os.getenv("ICEBERG_CATALOG_URI", "http://iceberg-rest:8181")

if not MILVUS_URI or not MILVUS_TOKEN:
    raise RuntimeError("MILVUS_URI and MILVUS_TOKEN must be set")


# ==================== 工具函数 ====================
def _make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def _load_clip_model(cache_dir: str, endpoint: str, key: str, secret: str):
    model_name = "clip-ViT-B-32"
    model_dir = os.path.join(cache_dir, model_name)

    if not os.path.exists(model_dir):
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
        )
        logging.info(f"[CLIP] Downloading {model_name} from MinIO...")
        os.makedirs(model_dir, exist_ok=True)
        resp = s3.list_objects_v2(Bucket="models", Prefix=f"{model_name}/")
        if "Contents" not in resp:
            raise RuntimeError(f"Model {model_name} not found in MinIO")
        for obj in resp["Contents"]:
            rel = obj["Key"][len(model_name) + 1 :]
            target = os.path.join(model_dir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            s3.download_file("models", obj["Key"], target)
        logging.info(f"[CLIP] Model downloaded to {model_dir}")
    else:
        logging.info(f"[CLIP] Model cached at {model_dir}")
    return SentenceTransformer(model_dir)


def init_ray():
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        logging.info("Connected to Ray cluster")
        logging.info("Resources: %s", ray.cluster_resources())
    except Exception as e:
        logging.error("Ray connection failed: %s", e)
        raise


# ==================== 异常转换装饰器 ====================
def _wrap_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            raise RuntimeError(msg) from None

    return wrapper


# ==================== Ray Actors ====================
@ray.remote
class SceneProcessor:
    def __init__(self):
        setup_logging()
        try:
            self.model = _load_clip_model(
                MODEL_CACHE_DIR, S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY
            )
            self.model_api_url = MODEL_API_URL
            logging.info("[Actor] SceneProcessor ready")
        except Exception as e:
            raise RuntimeError(f"SceneProcessor init failed: {e}") from None

    @_wrap_exception
    def process(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.model_api_url}/annotate",
            json={"scene_id": scene["scene_id"]},
            timeout=30,
        )
        if resp.status_code == 200:
            annot = resp.json()
            scene["scene_description"] = annot.get(
                "description", scene.get("scene_description", "")
            )
            scene["quality_score"] = annot.get(
                "quality", scene.get("quality_score", 0.0)
            )

        desc = scene.get("scene_description", "")
        emb = self.model.encode(desc).astype(np.float32).tolist()
        traj = np.random.randn(256).astype(np.float32)
        traj = traj / np.linalg.norm(traj)
        scene["scene_embedding"] = emb
        scene["trajectory_embedding"] = traj.tolist()
        return scene


@ray.remote
class LanceDBWriter:
    def __init__(self):
        setup_logging()
        try:
            os.makedirs(LANCE_DB_PATH, exist_ok=True)
            self.db = lancedb.connect(LANCE_DB_PATH)
            self.table_name = TABLE_NAME
            logging.info(f"[Actor] LanceDBWriter ready (path: {LANCE_DB_PATH})")
        except Exception as e:
            raise RuntimeError(f"LanceDBWriter init failed: {e}") from None

    @_wrap_exception
    def write(self, scenes: List[Dict[str, Any]]):
        rows = {
            "scene_id": [],
            "timestamp_start": [],
            "file_path": [],
            "scene_type": [],
            "weather": [],
            "quality_score": [],
            "scene_description": [],
            "scene_embedding": [],
            "trajectory_embedding": [],
        }
        for s in scenes:
            rows["scene_id"].append(s["scene_id"])
            rows["timestamp_start"].append(s["timestamp_start"])
            rows["file_path"].append(s["file_path"])
            rows["scene_type"].append(s["scene_type"])
            rows["weather"].append(s["weather"])
            rows["quality_score"].append(s["quality_score"])
            rows["scene_description"].append(s.get("scene_description", ""))
            rows["scene_embedding"].append(s["scene_embedding"])
            rows["trajectory_embedding"].append(s["trajectory_embedding"])

        table = pa.table(rows)
        if self.table_name in self.db.table_names():
            tbl = self.db.open_table(self.table_name)
            tbl.add(table)
            logging.info(f"[LanceDB] Appended {len(scenes)} scenes")
        else:
            self.db.create_table(self.table_name, table, mode="overwrite")
            logging.info(
                f"[LanceDB] Created table {self.table_name} with {len(scenes)} scenes"
            )


@ray.remote
class MilvusIndexer:
    def __init__(self):
        setup_logging()
        try:
            self.client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
            self.collection_name = "scene_vectors"
            self._ensure_collection()
            logging.info("[Actor] MilvusIndexer ready")
        except Exception as e:
            raise RuntimeError(f"MilvusIndexer init failed: {e}") from None

    def _ensure_collection(self):
        if self.client.has_collection(self.collection_name):
            try:
                info = self.client.describe_collection(self.collection_name)
                fields = {f["name"] for f in info.get("fields", [])}
                if "scene_id" in fields and "embedding" in fields:
                    logging.info(f"[Milvus] Collection valid")
                    self.client.load_collection(self.collection_name)
                    return
                else:
                    logging.info(f"[Milvus] Schema mismatch, recreating...")
                    self.client.drop_collection(self.collection_name)
            except Exception as e:
                logging.info(f"[Milvus] Check failed: {e}, recreating...")
                try:
                    self.client.drop_collection(self.collection_name)
                except Exception:
                    pass

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="scene_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            metric_type="COSINE",
        )
        idx = self.client.prepare_index_params()
        idx.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        self.client.create_index(self.collection_name, idx)
        self.client.load_collection(self.collection_name)
        logging.info(f"[Milvus] Created collection")

    @_wrap_exception
    def index(self, scenes: List[Dict[str, Any]]):
        data = [
            {"scene_id": s["scene_id"], "embedding": s["scene_embedding"]}
            for s in scenes
        ]
        self.client.insert(collection_name=self.collection_name, data=data)
        logging.info(f"[Milvus] Indexed {len(data)} vectors")


# ==================== Ray Tasks ====================
@ray.remote(num_cpus=1)
@_wrap_exception
def download_metadata(bucket: str, key: str) -> List[Dict[str, Any]]:
    setup_logging()
    s3 = _make_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


@ray.remote(num_cpus=1)
@_wrap_exception
def sync_to_iceberg(scenes: List[Dict[str, Any]]):
    setup_logging()
    catalog = load_catalog(
        "rest",
        **{
            "type": "rest",
            "uri": ICEBERG_CATALOG_URI,
            "warehouse": ICEBERG_WAREHOUSE,
            "s3.endpoint": S3_ENDPOINT,
            "s3.access-key-id": S3_ACCESS_KEY,
            "s3.secret-access-key": S3_SECRET_KEY,
            "s3.path-style-access": "true",
            "s3.region": "us-east-1",
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
        },
    )
    try:
        catalog.create_namespace("robotloop")
    except Exception:
        pass

    schema = Schema(
        NestedField(
            field_id=1, name="scene_id", field_type=StringType(), required=True
        ),
        NestedField(
            field_id=2,
            name="timestamp_start",
            field_type=TimestampType(),
            required=True,
        ),
        NestedField(
            field_id=3, name="file_path", field_type=StringType(), required=True
        ),
        NestedField(
            field_id=4, name="scene_type", field_type=StringType(), required=True
        ),
        NestedField(field_id=5, name="weather", field_type=StringType(), required=True),
        NestedField(
            field_id=6, name="quality_score", field_type=FloatType(), required=False
        ),
        NestedField(
            field_id=7,
            name="scene_description",
            field_type=StringType(),
            required=False,
        ),
    )

    partition = PartitionSpec(
        PartitionField(
            source_id=2, field_id=1000, transform=HourTransform(), name="ts_hour"
        )
    )

    try:
        table = catalog.load_table(ICEBERG_TABLE)
    except Exception:
        table = catalog.create_table(
            identifier=ICEBERG_TABLE,
            schema=schema,
            location=f"{ICEBERG_WAREHOUSE}/{ICEBERG_TABLE.split('.')[-1]}",
            partition_spec=partition,
        )

    rows = []
    for s in scenes:
        ts = s["timestamp_start"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif not isinstance(ts, datetime):
            raise RuntimeError(
                f"timestamp_start must be str or datetime, got {type(ts).__name__}"
            )

        rows.append(
            {
                "scene_id": s["scene_id"],
                "timestamp_start": ts,
                "file_path": s["file_path"],
                "scene_type": s["scene_type"],
                "weather": s["weather"],
                "quality_score": float(s["quality_score"]),
                "scene_description": s.get("scene_description", ""),
            }
        )

    pa_schema = pa.schema(
        [
            pa.field("scene_id", pa.string(), nullable=False),
            pa.field("timestamp_start", pa.timestamp("us"), nullable=False),
            pa.field("file_path", pa.string(), nullable=False),
            pa.field("scene_type", pa.string(), nullable=False),
            pa.field("weather", pa.string(), nullable=False),
            pa.field("quality_score", pa.float32(), nullable=True),
            pa.field("scene_description", pa.string(), nullable=True),
        ]
    )

    arrow = pa.Table.from_pylist(rows, schema=pa_schema)

    table.append(arrow)
    logger.info(f"[Iceberg] Synced {len(scenes)} scenes")


# ==================== Main ====================
def main():
    init_ray()

    processor = SceneProcessor.remote()
    lance_writer = LanceDBWriter.remote()
    milvus_indexer = MilvusIndexer.remote()

    consumer = KafkaConsumer(
        "raw-data-ingest",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="ray-worker",
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    logging.info("Processor started, listening for Kafka events...")

    for msg in consumer:
        event = msg.value
        records = event.get("Records", [])
        if not records:
            logging.warning("No records in event: %s", event)
            continue

        for record in records:
            key = record["s3"]["object"]["key"]
            bucket = record["s3"]["bucket"]["name"]

            if key == "scene_metadata.json":
                logging.info("Processing scene metadata...")

                scenes = ray.get(download_metadata.remote(bucket, key))
                processed = ray.get([processor.process.remote(s) for s in scenes])
                ray.get(
                    [
                        lance_writer.write.remote(processed),
                        milvus_indexer.index.remote(processed),
                        sync_to_iceberg.remote(processed),
                    ]
                )
                logging.info("Completed %d scenes", len(processed))


if __name__ == "__main__":
    main()
