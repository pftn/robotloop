#!/usr/bin/env python3
"""RobotLoop Ray 流水线。

主干不变：MinIO --bucket notification--> Kafka(raw-data-ingest) --> 本 Worker。

分流（注册表模式按扩展名挂处理器）：

    HANDLER_REGISTRY = {
        ".json":  旧模拟场景元数据（v1 逻辑原样保留，压测/演示用）
        ".mcap":  真实数据入口 -> 解析对齐 -> Episode -> 质检 -> 向量化 -> 写库
        ".bag":   同上（rosbags 纯 Python 库，零 ROS 依赖）
        ".jsonl": CI 模拟日志 -> Episode（同上）
    }

Bucket 分工：robotloop-raw 存设备上传的原始包（唯一配 bucket notification 的
bucket），robotloop-data 存处理产物（frames parquet，不触发事件）。

Episode 统一写库（元数据与帧数据分离存储）：
    帧数据 -> MinIO s3://robotloop-data/frames/{episode_id}.parquet（一 episode 一文件）
    元数据 -> Iceberg robotloop.episodes（REST catalog，只存 parquet_path 指针）
    向量   -> Milvus/Zilliz episode_vectors（CLIP 512d）+ LanceDB robotloop_episodes

质检：入库前自动执行失败过滤，被滤轨迹计数进 Prometheus。
监控：/metrics 暴露吞吐与质检过滤统计（:9100）。
"""

import boto3
import json
import lancedb
import logging
import numpy as np
import os
import pyarrow as pa
import ray
import requests
import shutil
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
from urllib.parse import unquote_plus


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
S3_BUCKET = os.getenv(
    "S3_BUCKET", "robotloop-data"
)  # frames 产物 bucket；原始包 bucket 从 Kafka 事件取
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model-api:9002")
LANCE_DB_PATH = os.getenv("LANCE_PATH", "/tmp/lance")
TABLE_NAME = "robotloop_scenes"
EPISODE_LANCE_TABLE = os.getenv("EPISODE_LANCE_TABLE", "robotloop_episodes")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")

ICEBERG_WAREHOUSE = os.getenv("ICEBERG_WAREHOUSE", f"s3://iceberg-warehouse")
ICEBERG_TABLE = os.getenv("ICEBERG_TABLE", "robotloop.scenes_meta")
ICEBERG_EPISODES_TABLE = os.getenv("ICEBERG_EPISODES_TABLE", "robotloop.episodes")
ICEBERG_CATALOG_URI = os.getenv("ICEBERG_CATALOG_URI", "http://iceberg-rest:8181")

METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))

if not MILVUS_URI or not MILVUS_TOKEN:
    raise RuntimeError("MILVUS_URI and MILVUS_TOKEN must be set")


# ==================== Prometheus 指标（吞吐 + 质检过滤统计） ====================
from prometheus_client import Counter, Histogram, start_http_server

EPISODES_INGESTED = Counter(
    "robotloop_episodes_ingested_total",
    "Episodes written to Iceberg+MinIO+vector stores",
    ["embodiment_tag", "source"],
)
EPISODES_FILTERED = Counter(
    "robotloop_episodes_filtered_total",
    "Episodes dropped by quality gate before writing",
    ["reason"],
)
FRAMES_WRITTEN = Counter(
    "robotloop_frames_written_total",
    "Frame rows written to MinIO parquet",
)
INGEST_DURATION = Histogram(
    "robotloop_ingest_duration_seconds",
    "End-to-end episode ingest latency",
    ["stage"],
)
PARSE_ERRORS = Counter(
    "robotloop_parse_errors_total",
    "Bag parse / handler failures",
    ["ext"],
)
SCENES_INGESTED = Counter(
    "robotloop_scenes_ingested_total",
    "Legacy simulated scenes processed",
)


# ==================== 工具函数 ====================
def _make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def _clip_model_complete(model_dir: str) -> bool:
    """模型目录完整性：config.json + 权重文件（safetensors/bin 二选一）。

    目录存在不等于完整 —— 并发下载时先 makedirs 后下文件，半成品目录
    曾被当成缓存命中，SentenceTransformer 直接报 no model.safetensors。
    """
    if not os.path.isdir(model_dir):
        return False
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return False
    return any(
        os.path.isfile(os.path.join(model_dir, w))
        for w in ("model.safetensors", "pytorch_model.bin")
    )


def _download_clip_model(model_dir: str, endpoint: str, key: str, secret: str):
    """下载到临时目录再原子 rename：任何时刻 model_dir 要么完整要么不存在。"""
    model_name = os.path.basename(model_dir)
    tmp_dir = f"{model_dir}.tmp-{os.getpid()}"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )
    logging.info(f"[CLIP] Downloading {model_name} from MinIO...")
    resp = s3.list_objects_v2(Bucket="models", Prefix=f"{model_name}/")
    if "Contents" not in resp:
        raise RuntimeError(f"Model {model_name} not found in MinIO")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)  # 清掉半成品
    os.makedirs(tmp_dir, exist_ok=True)
    for obj in resp["Contents"]:
        rel = obj["Key"][len(model_name) + 1 :]
        if not rel:
            continue
        target = os.path.join(tmp_dir, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        s3.download_file("models", obj["Key"], target)
    os.rename(tmp_dir, model_dir)
    logging.info(f"[CLIP] Model downloaded to {model_dir}")


def _load_clip_model(cache_dir: str, endpoint: str, key: str, secret: str):
    """加载 CLIP 模型（多 actor 并发安全）。

    SceneProcessor / EpisodeProcessor 等 actor 几乎同时创建、共享同一份
    /tmp/models 缓存：os.mkdir 原子锁保证只有一个进程下载，其余等锁后
    走完整性校验过的缓存；持锁进程崩溃有 stale 锁清理兜底。
    """
    model_name = "clip-ViT-B-32"
    model_dir = os.path.join(cache_dir, model_name)
    lock_dir = model_dir + ".download-lock"

    os.makedirs(cache_dir, exist_ok=True)
    for _ in range(300):  # 最长等 10 分钟
        if _clip_model_complete(model_dir):
            break
        try:
            os.mkdir(lock_dir)  # 原子：抢到锁的进程负责下载
        except FileExistsError:
            try:  # stale 锁（持锁进程崩溃）超过 10 分钟强制清理
                if time.time() - os.path.getctime(lock_dir) > 600:
                    os.rmdir(lock_dir)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(2)
            continue
        try:
            _download_clip_model(model_dir, endpoint, key, secret)
        finally:
            os.rmdir(lock_dir)  # 成败都放锁；成败由完整性校验裁决
    if not _clip_model_complete(model_dir):
        raise RuntimeError(f"CLIP model download incomplete at {model_dir}")
    logging.info(f"[CLIP] Loading model from {model_dir}")
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


# ==================== Ray Actors（v1 旧链路，原样保留） ====================
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


# ==================== Ray Actors（Episode 链路） ====================
@ray.remote
class EpisodeProcessor:
    """Episode 向量化：CLIP 编码语言指令（512d）+ 动作统计摘要轨迹向量。"""

    def __init__(self):
        setup_logging()
        try:
            self.model = _load_clip_model(
                MODEL_CACHE_DIR, S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY
            )
            logging.info("[Actor] EpisodeProcessor ready")
        except Exception as e:
            raise RuntimeError(f"EpisodeProcessor init failed: {e}") from None

    @_wrap_exception
    def process(self, ep: Dict[str, Any]) -> Dict[str, Any]:
        from robotloop.retrieval.encoder import encode_trajectory

        text = ep.get("language_instruction") or ep.get("task", "")
        ep["text_embedding"] = self.model.encode(text).astype(np.float32).tolist()
        actions = ep.pop("_actions", [])  # 解析侧随字典带来的动作序列
        ep["traj_embedding"] = encode_trajectory(actions).astype(np.float32).tolist()
        return ep


@ray.remote
class EpisodeLanceWriter:
    """Episode 向量/元数据写 LanceDB（本地特征镜像，与 v1 分层一致）。"""

    def __init__(self):
        setup_logging()
        try:
            os.makedirs(LANCE_DB_PATH, exist_ok=True)
            self.db = lancedb.connect(LANCE_DB_PATH)
            self.table_name = EPISODE_LANCE_TABLE
            logging.info(f"[Actor] EpisodeLanceWriter ready (table: {self.table_name})")
        except Exception as e:
            raise RuntimeError(f"EpisodeLanceWriter init failed: {e}") from None

    @_wrap_exception
    def write(self, episodes: List[Dict[str, Any]]):
        rows = {
            "episode_id": [e["episode_id"] for e in episodes],
            "embodiment_tag": [e["embodiment_tag"] for e in episodes],
            "task": [e["task"] for e in episodes],
            "language_instruction": [
                e.get("language_instruction", "") for e in episodes
            ],
            "source": [e.get("source", "") for e in episodes],
            "success": [e.get("success") for e in episodes],
            "duration": [float(e.get("duration", 0.0)) for e in episodes],
            "parquet_path": [e.get("parquet_path", "") for e in episodes],
            "text_embedding": [e["text_embedding"] for e in episodes],
            "traj_embedding": [e["traj_embedding"] for e in episodes],
        }
        table = pa.table(rows)
        if self.table_name in self.db.table_names():
            self.db.open_table(self.table_name).add(table)
        else:
            self.db.create_table(self.table_name, table, mode="overwrite")
        logging.info(
            f"[LanceDB] Appended {len(episodes)} episodes -> {self.table_name}"
        )


@ray.remote
class EpisodeMilvusIndexer:
    """Episode CLIP 文本向量写 Milvus/Zilliz（episode_vectors, 512d）。"""

    def __init__(self):
        setup_logging()
        try:
            self.client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
            self.collection_name = "episode_vectors"
            self._ensure_collection()
            logging.info("[Actor] EpisodeMilvusIndexer ready")
        except Exception as e:
            raise RuntimeError(f"EpisodeMilvusIndexer init failed: {e}") from None

    def _ensure_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="episode_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512
        )
        self.client.create_collection(
            collection_name=self.collection_name, schema=schema, metric_type="COSINE"
        )
        idx = self.client.prepare_index_params()
        idx.add_index(
            field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
        )
        self.client.create_index(self.collection_name, idx)
        self.client.load_collection(self.collection_name)
        logging.info("[Milvus] Created collection episode_vectors")

    @_wrap_exception
    def index(self, episodes: List[Dict[str, Any]]):
        data = [
            {"episode_id": e["episode_id"], "embedding": e["text_embedding"]}
            for e in episodes
        ]
        self.client.insert(collection_name=self.collection_name, data=data)
        logging.info(f"[Milvus] Indexed {len(data)} episode vectors")


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
    """v1 旧链路 Iceberg 写入（scenes_meta 表），原样保留。"""
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


@ray.remote(num_cpus=1)
@_wrap_exception
def parse_bag_to_episodes(bucket: str, key: str) -> Dict[str, Any]:
    """下载 bag -> 注册表解析 -> 质检 -> Episode dicts。

    返回 {"episodes", "timings", "filter_breakdown"}：episodes 为通过质检的
    Episode 字典列表（含 _actions 供向量化）。指标数据随返回值带回 driver
    端统一记录 —— Prometheus 指标对象含线程锁不可 pickle，且 worker 进程里
    计数不会出现在 driver 的 /metrics 端点上。
    """
    setup_logging()
    from robotloop.ingest.registry import ParseContext, get_parser
    from robotloop.quality.failure_filter import filter_episodes

    ext = os.path.splitext(key)[1].lower()
    parser = get_parser(key)
    if parser is None:
        raise RuntimeError(f"no parser registered for {key}")

    s3 = _make_s3_client()
    head = s3.head_object(Bucket=bucket, Key=key)
    ctx = ParseContext.from_s3_metadata(bucket, key, head.get("Metadata"))

    local_path = f"/tmp/robotloop_bags/{key.replace('/', '_')}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    t0 = time.perf_counter()
    s3.download_file(bucket, key, local_path)
    t_download = time.perf_counter() - t0

    t0 = time.perf_counter()
    episodes = parser(local_path, ctx)
    t_parse = time.perf_counter() - t0

    # ---- 质检在入库前自动执行 ----
    # 设备自采数据 success 多为 None（未标注），keep_unlabeled=True 只剔硬伤
    # （截断/时长异常/缺指令/动作全零）；公开数据集灌库时按 success 过滤在
    # 灌库脚本侧做，阈值对着真实数据分布定。
    filter_result = filter_episodes(episodes, keep_unlabeled=True)
    kept = filter_result.kept
    summary = filter_result.summary
    if summary["removed"]:
        logger.info(
            "[Quality] %s: filtered %d/%d episodes: %s",
            key,
            summary["removed"],
            summary["total"],
            summary["reason_breakdown"],
        )

    out = []
    for ep in kept:
        d = ep.meta_dict()
        d["_actions"] = [list(s.action) for s in ep.steps]
        d["_episode"] = ep  # sink 阶段需要完整 Episode（帧数据）
        out.append(d)
    return {
        "episodes": out,
        "timings": {"download": t_download, "parse_align": t_parse},
        "filter_breakdown": dict(summary["reason_breakdown"]),
    }


@ray.remote(num_cpus=1)
@_wrap_exception
def sink_episodes_to_iceberg(episode_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Episode 统一写库（元数据与帧数据分离存储）：

    帧数据 -> MinIO frames/{episode_id}.parquet（一 episode 一文件）
    元数据 -> Iceberg robotloop.episodes（REST catalog，只存 parquet_path 指针）

    返回 {"episodes", "timings", "frames_total", "ingested_labels"}。
    """
    setup_logging()
    from robotloop.schema.sink import EpisodeSink

    episodes = [d.pop("_episode") for d in episode_dicts]
    sink = EpisodeSink()
    t0 = time.perf_counter()
    n = sink.write(episodes)
    t_sink = time.perf_counter() - t0
    # sink 写完帧后 parquet_path 已回填，同步回字典（供 LanceDB 行使用）；
    # 指标数据带回 driver 端记录（原因同 parse_bag_to_episodes）
    frames_total = 0
    ingested_labels = []
    for d, ep in zip(episode_dicts, episodes):
        d["parquet_path"] = ep.parquet_path
        frames_total += ep.num_frames
        ingested_labels.append((ep.embodiment_tag, ep.source.value))
    logger.info(
        f"[Iceberg] Synced {n} episodes -> {ICEBERG_EPISODES_TABLE} (+MinIO frames)"
    )
    return {
        "episodes": episode_dicts,
        "timings": {"iceberg_minio_sink": t_sink},
        "frames_total": frames_total,
        "ingested_labels": ingested_labels,
    }


# ==================== 事件分流（注册表模式） ====================
# 每个 handler 接收 (bucket, key, actors) —— actors 是 main() 里建好的 actor 句柄字典。
# 新增文件类型时 @register_handler(".hdf5") 挂一个新函数即可，主干零改动。
HANDLER_REGISTRY: Dict[str, Any] = {}


def register_handler(*extensions: str):
    def deco(fn):
        for ext in extensions:
            ext = ext.lower()
            if not ext.startswith("."):
                ext = "." + ext
            HANDLER_REGISTRY[ext] = fn
        return fn

    return deco


@register_handler(".json")
def handle_legacy_scenes(bucket: str, key: str, actors: Dict[str, Any]):
    """v1 旧模拟场景元数据（旧逻辑保留，压测/演示仍要用）。"""
    if os.path.basename(key) != "scene_metadata.json":
        logging.warning("skip non-scene json: %s", key)
        return
    logging.info("Processing scene metadata (legacy v1 pipeline)...")
    scenes = ray.get(download_metadata.remote(bucket, key))
    processed = ray.get([actors["scene_processor"].process.remote(s) for s in scenes])
    ray.get(
        [
            actors["lance_writer"].write.remote(processed),
            actors["milvus_indexer"].index.remote(processed),
            sync_to_iceberg.remote(processed),
        ]
    )
    SCENES_INGESTED.inc(len(processed))
    logging.info("Completed %d scenes", len(processed))


@register_handler(".mcap", ".bag", ".jsonl")
def handle_episode_bag(bucket: str, key: str, actors: Dict[str, Any]):
    """真实数据入口：bag -> Episode -> 质检 -> 向量化 -> 写库。

    Prometheus 指标全部在 driver 端（本进程）记录：remote 任务只带回
    timings / 计数数据。worker 进程里的 Counter/Histogram 与 driver 的
    /metrics 端点是两套进程内 registry，在 remote 里计数既序列化不了、
    暴露了也看不见。
    """
    logging.info("Processing episode bag: s3://%s/%s", bucket, key)
    parse_result = ray.get(parse_bag_to_episodes.remote(bucket, key))
    for stage, sec in parse_result["timings"].items():
        INGEST_DURATION.labels(stage=stage).observe(sec)
    for reason, n in parse_result["filter_breakdown"].items():
        EPISODES_FILTERED.labels(reason=reason).inc(n)

    ep_dicts = parse_result["episodes"]
    if not ep_dicts:
        logging.warning(
            "[Quality] all episodes from %s filtered out, nothing to write", key
        )
        return
    # 1) Iceberg 元数据 + MinIO 帧 parquet（先落库拿到 parquet_path）
    sink_result = ray.get(sink_episodes_to_iceberg.remote(ep_dicts))
    INGEST_DURATION.labels(stage="iceberg_minio_sink").observe(
        sink_result["timings"]["iceberg_minio_sink"]
    )
    FRAMES_WRITTEN.inc(sink_result["frames_total"])
    for tag, src in sink_result["ingested_labels"]:
        EPISODES_INGESTED.labels(embodiment_tag=tag, source=src).inc()

    ep_dicts = sink_result["episodes"]
    # 2) 向量化 + 向量库
    processed = ray.get(
        [actors["episode_processor"].process.remote(d) for d in ep_dicts]
    )
    ray.get(
        [
            actors["episode_lance_writer"].write.remote(processed),
            actors["episode_milvus_indexer"].index.remote(processed),
        ]
    )
    logging.info("Completed %d episodes from %s", len(processed), key)


# ==================== Main ====================
def main():
    init_ray()

    # 启动即建 episodes 表（幂等）：避免检索 API 在首批数据入库前查询时
    # iceberg-rest 报 Table does not exist
    from robotloop.schema.iceberg import create_tables
    from robotloop.schema.sink import load_rest_catalog

    for attempt in range(30):
        try:
            create_tables(load_rest_catalog())
            logging.info("Iceberg episodes table ready")
            break
        except Exception as e:
            logging.warning(
                "waiting for iceberg-rest (%s), retry %d/30", e, attempt + 1
            )
            time.sleep(2)

    # Prometheus /metrics（吞吐与质检过滤统计进 Grafana）
    start_http_server(METRICS_PORT)
    logging.info("Prometheus metrics on :%d/metrics", METRICS_PORT)

    # 预热 CLIP 模型：actor 共享 driver 同容器的 /tmp/models 缓存，
    # 预热后各 actor 创建时全部走完整缓存（并发下载有锁兜底）
    _load_clip_model(MODEL_CACHE_DIR, S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY)

    actors = {
        "scene_processor": SceneProcessor.remote(),
        "lance_writer": LanceDBWriter.remote(),
        "milvus_indexer": MilvusIndexer.remote(),
        "episode_processor": EpisodeProcessor.remote(),
        "episode_lance_writer": EpisodeLanceWriter.remote(),
        "episode_milvus_indexer": EpisodeMilvusIndexer.remote(),
    }

    consumer = KafkaConsumer(
        "raw-data-ingest",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="ray-worker",
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    logging.info(
        "Processor started, listening for Kafka events... (handlers: %s)",
        sorted(HANDLER_REGISTRY),
    )

    for msg in consumer:
        event = msg.value
        records = event.get("Records", [])
        if not records:
            logging.warning("No records in event: %s", event)
            continue

        for record in records:
            # MinIO/S3 事件通知里的 object key 是 URL 编码的（QueryEscape
            # 风格，空格为 +）：目录前缀的 / 会变成 %2F，必须先解码再用于
            # head_object/download_file，否则 key 与 metadata 都取不到
            key = unquote_plus(record["s3"]["object"]["key"])
            bucket = record["s3"]["bucket"]["name"]
            ext = os.path.splitext(key)[1].lower()
            handler = HANDLER_REGISTRY.get(ext)
            if handler is None:
                # 辅助文件（.png 图像等）静默跳过：bucket notification 对
                # 所有对象事件都发 Kafka，只有注册过扩展名的 key 需要处理
                logging.debug("skip %s (no handler for %s)", key, ext or "<none>")
                continue
            try:
                handler(bucket, key, actors)
            except Exception as e:
                PARSE_ERRORS.labels(ext=ext).inc()
                logging.error("handler failed for %s: %s", key, e)


if __name__ == "__main__":
    main()
