import os, logging
from fastapi import FastAPI, Query, Request
import boto3
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import lancedb
import pyarrow as pa
import pyarrow.compute as pc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="RobotLoop API - LanceDB + Milvus")

# ---------- 环境变量 ----------
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = "models"
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
LANCE_DB_PATH = os.getenv("LANCE_PATH", "/tmp/lance")
TABLE_NAME = "robotloop_scenes"
EPISODE_LANCE_TABLE = os.getenv("EPISODE_LANCE_TABLE", "robotloop_episodes")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")

ICEBERG_CATALOG_URI = os.getenv("ICEBERG_CATALOG_URI", "http://iceberg-rest:8181")
ICEBERG_WAREHOUSE = os.getenv("ICEBERG_WAREHOUSE", "s3://iceberg-warehouse")
ICEBERG_EPISODES_TABLE = os.getenv("ICEBERG_EPISODES_TABLE", "robotloop.episodes")
EPISODE_COLLECTION = "episode_vectors"

if not MILVUS_URI or not MILVUS_TOKEN:
    raise RuntimeError("MILVUS_URI and MILVUS_TOKEN must be set")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

# ---------- Milvus 初始化 ----------
client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
COLLECTION_NAME = "scene_vectors"
if client.has_collection(COLLECTION_NAME):
    client.load_collection(COLLECTION_NAME)

# ---------- LanceDB 初始化 ----------
db = lancedb.connect(LANCE_DB_PATH)
table = None


def _get_lance_table():
    """懒加载打开 scenes 表：API 启动通常早于 Ray 链路建表，
    启动时的一次性判断会把"没有表"缓存成永久状态 —— 改为每次查询前重试。"""
    global table
    if table is None and TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        logger.info(f"Opened LanceDB table {TABLE_NAME}, rows: {table.count_rows()}")
    return table


def _to_jsonable(obj, _path="$"):
    """递归转成 JSON 原生类型（numpy/bytes）。

    FastAPI 的 jsonable_encoder 遇到 numpy scalar/ndarray 会直接 500；
    转换发生时打 warning 并带字段路径，上游哪一列混入非原生对象一眼可查。
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v, f"{_path}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        logger.warning("non-jsonable ndarray at %s, converted to list", _path)
        return obj.tolist()
    if isinstance(obj, np.generic):
        logger.warning(
            "non-jsonable %s at %s, converted to scalar", type(obj).__name__, _path
        )
        return obj.item()
    if isinstance(obj, bytes):
        logger.warning("non-jsonable bytes at %s, decoded", _path)
        return obj.decode("utf-8", "replace")
    return obj


_get_lance_table()
if table is None:
    logger.warning(f"LanceDB table {TABLE_NAME} not found. Will retry on each request.")


# ---------- CLIP 模型加载（从 MinIO 下载） ----------
def load_clip_model():
    model_name = "clip-ViT-B-32"
    model_dir = os.path.join(MODEL_CACHE_DIR, model_name)
    if not os.path.exists(model_dir):
        logger.info(f"Downloading {model_name} from MinIO...")
        os.makedirs(model_dir, exist_ok=True)
        objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{model_name}/")
        if "Contents" not in objects:
            raise RuntimeError(
                f"Model {model_name} not found in MinIO bucket '{S3_BUCKET}'"
            )
        for obj in objects["Contents"]:
            key = obj["Key"]
            rel_path = key[len(model_name) + 1 :]
            target_path = os.path.join(model_dir, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            s3.download_file(S3_BUCKET, key, target_path)
        logger.info(f"Model {model_name} downloaded to {model_dir}")
    else:
        logger.info(f"Model {model_name} found in cache {model_dir}")
    return SentenceTransformer(model_dir)


model = load_clip_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search_scenes(
    request: Request,
    text_query: str = Query(None),
    quality_min: float = Query(0.0),
    scene_type: str = Query(None),
    top_k: int = Query(10),
):
    """
    混合搜索: 语义向量搜索 + 结构化过滤
    支持 Query 参数或 JSON body
    """
    # 尝试从 JSON body 读取 (兼容两种调用方式)
    try:
        body = request.json()
        if body:
            text_query = body.get("text_query", text_query)
            quality_min = body.get("quality_min", quality_min)
            scene_type = body.get("scene_type", scene_type)
            top_k = body.get("top_k", top_k)
    except:
        pass

    # ── 调试信息收集 ──
    debug_info = {
        "input_params": {
            "text_query": text_query,
            "quality_min": quality_min,
            "scene_type": scene_type,
            "top_k": top_k,
        },
        "table_status": "unknown",
        "milvus_status": "unknown",
        "lancedb_status": "unknown",
    }

    # ── 检查 LanceDB 表 ──
    if _get_lance_table() is None:
        debug_info["table_status"] = "not_initialized"
        return {
            "count": 0,
            "results": [],
            "message": "LanceDB 表未初始化，请先运行数据预处理",
            "debug": debug_info,
        }

    debug_info["table_status"] = f"initialized, rows={table.count_rows()}"

    # ── Step 1: Milvus 向量搜索获取候选 scene_id ──
    hit_ids = []

    if text_query:
        try:
            query_emb = model.encode(text_query).tolist()
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_emb],
                limit=top_k * 10,  # 扩大范围避免过滤后无结果
                output_fields=["scene_id"],
            )

            # 兼容不同版本 pymilvus
            for hit in results[0]:
                hit_id = (
                    hit.get("pk") or hit.get("id") or getattr(hit.entity, "id", None)
                )
                if hit_id:
                    hit_ids.append(hit_id)

            debug_info["milvus_status"] = f"success, hits={len(hit_ids)}"
            debug_info["milvus_raw_ids"] = hit_ids[:20]

        except Exception as e:
            debug_info["milvus_status"] = f"error: {str(e)}"
    else:
        # 无文本查询，获取所有 ID
        try:
            query_results = client.query(
                collection_name=COLLECTION_NAME,
                filter="scene_id != ''",
                output_fields=["scene_id"],
                limit=10000,
            )
            hit_ids = [r["scene_id"] for r in query_results]
            debug_info["milvus_status"] = f"query_all, ids={len(hit_ids)}"
        except Exception as e:
            debug_info["milvus_status"] = f"query_error: {str(e)}"

    if not hit_ids:
        return {
            "count": 0,
            "results": [],
            "message": "Milvus 向量搜索未返回结果",
            "debug": debug_info,
            "suggestion": "检查 Milvus 集合是否有数据，或尝试不同的查询文本",
        }

    # ── Step 2: LanceDB PyArrow 过滤和排序 ──
    try:
        # 读取整个表为 Arrow
        full_arrow = table.to_arrow()

        # 构建布尔掩码
        mask = pc.field("scene_id").isin(hit_ids)

        if quality_min > 0:
            mask = mask & (pc.field("quality_score") >= quality_min)

        if scene_type:
            mask = mask & (pc.field("scene_type") == scene_type)

        debug_info["lancedb_filter"] = (
            f"scene_id in {hit_ids[:5]}... AND quality>={quality_min} AND type={scene_type}"
        )

        # 应用过滤
        filtered = full_arrow.filter(mask)
        debug_info["lancedb_status"] = f"success, filtered_count={len(filtered)}"

        if len(filtered) == 0:
            return {
                "count": 0,
                "results": [],
                "message": "LanceDB 过滤后无结果",
                "debug": debug_info,
                "suggestion": f"尝试放宽条件: quality_min=0.0 或不指定 scene_type",
            }

        # 按 Milvus 返回的顺序排序（最近似在前）
        id_order = {sid: idx for idx, sid in enumerate(hit_ids)}
        order_arr = pa.array(
            [id_order[sid] for sid in filtered.column("scene_id").to_pylist()]
        )
        indices = pc.sort_indices(order_arr)
        sorted_table = filtered.take(indices[:top_k])

        # 转换为字典列表
        columns = [
            "scene_id",
            "file_path",
            "scene_type",
            "quality_score",
            "scene_description",
            "weather",
        ]
        results = []
        for i in range(len(sorted_table)):
            row = {col: sorted_table.column(col)[i].as_py() for col in columns}
            results.append(row)

    except Exception as e:
        debug_info["lancedb_status"] = f"error: {str(e)}"
        return {
            "count": 0,
            "results": [],
            "message": f"LanceDB 查询失败: {str(e)}",
            "debug": debug_info,
        }

    # ── 返回结果 ──
    return _to_jsonable(
        {
            "count": len(results),
            "results": results,
            "debug": debug_info,
            "suggestion": None if len(results) > 0 else "尝试放宽过滤条件",
        }
    )


# ====================================================================
# Episode 混合检索：
#   CLIP 语义（Milvus episode_vectors） × Iceberg 结构化过滤
#   （embodiment / success / source / task）联合命中
# ====================================================================
_iceberg_catalog = None


def _get_iceberg_catalog():
    """惰性加载 REST catalog（首次请求时连接，Iceberg REST 未起不拖垮 API）。"""
    global _iceberg_catalog
    if _iceberg_catalog is None:
        from pyiceberg.catalog import load_catalog

        _iceberg_catalog = load_catalog(
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
            },
        )
    return _iceberg_catalog


def _episodes_row_filter(embodiment, success, source, task, dataset_name) -> str:
    """Iceberg row_filter SQL（与 robotloop.retrieval.store._iceberg_row_filter 同语义）。"""
    clauses = []
    if embodiment:
        clauses.append(f"embodiment_tag = '{embodiment}'")
    if success is not None:
        clauses.append(f"success = {'true' if success else 'false'}")
    if source:
        clauses.append(f"source = '{source}'")
    if task:
        clauses.append(f"task = '{task}'")
    if dataset_name:
        clauses.append(f"dataset_name = '{dataset_name}'")
    return " AND ".join(clauses) if clauses else "1=1"


@app.post("/episodes/search")
def search_episodes(
    text_query: str = Query(None),
    embodiment: str = Query(
        None, description="本体标签，如 aloha / widowx / agibot_g1"
    ),
    success: bool = Query(None, description="成功标记；不传则不过滤"),
    source: str = Query(None, description="teleop | sim | real"),
    task: str = Query(None, description="任务名，如 pick_red_cube"),
    dataset_name: str = Query(None),
    top_k: int = Query(10),
):
    """Episode 混合检索：CLIP 语义 × Iceberg 结构化过滤。

    演示 query：「ALOHA 双臂成功抓取红色方块的相似轨迹」
    = text_query("抓取红色方块") + embodiment=aloha AND success=true。
    """
    debug = {
        "params": {
            "text_query": text_query,
            "embodiment": embodiment,
            "success": success,
            "source": source,
            "task": task,
            "top_k": top_k,
        }
    }

    # ── Step 1: Iceberg 结构化过滤 → 候选 episode_id 集合 ──
    try:
        tbl = _get_iceberg_catalog().load_table(ICEBERG_EPISODES_TABLE)
        row_filter = _episodes_row_filter(
            embodiment, success, source, task, dataset_name
        )
        meta = tbl.scan(row_filter=row_filter).to_arrow()
        debug["iceberg_filter"] = row_filter
        debug["iceberg_hits"] = meta.num_rows
    except Exception as e:
        return {
            "count": 0,
            "results": [],
            "message": f"Iceberg 查询失败: {e}",
            "debug": debug,
        }

    if meta.num_rows == 0:
        return {
            "count": 0,
            "results": [],
            "message": "结构化过滤后无候选",
            "debug": debug,
        }

    candidate_ids = meta.column("episode_id").to_pylist()

    # ── Step 2: CLIP 语义检索 → 向量候选（在结构化候选内取交集） ──
    if text_query:
        try:
            qvec = model.encode(text_query).tolist()
            res = client.search(
                collection_name=EPISODE_COLLECTION,
                data=[qvec],
                limit=max(top_k * 10, 100),
                output_fields=["episode_id"],
            )
            ranked_ids = []
            for hit in res[0]:
                hid = hit.get("pk") or hit.get("id") or getattr(hit.entity, "id", None)
                if hid:
                    ranked_ids.append(str(hid))
            debug["milvus_hits"] = len(ranked_ids)
            in_candidates = set(candidate_ids)
            ordered = [i for i in ranked_ids if i in in_candidates]
            # 向量未召回但满足结构化条件的排在后面（保召回）
            ordered += [i for i in candidate_ids if i not in set(ordered)]
        except Exception as e:
            debug["milvus_error"] = str(e)
            ordered = candidate_ids
    else:
        ordered = candidate_ids

    ordered = ordered[:top_k]

    # ── Step 3: 按命中顺序取回元数据行 ──
    rows = {r["episode_id"]: r for r in meta.to_pylist()}
    results = [rows[i] for i in ordered if i in rows]
    for r in results:
        if r.get("created_at") is not None:
            r["created_at"] = str(r["created_at"])

    return _to_jsonable({"count": len(results), "results": results, "debug": debug})


@app.get("/episodes/stats")
def episode_stats():
    """任务分布 / 本体分布 / 成功率（质检统计的查询入口）。"""
    try:
        tbl = _get_iceberg_catalog().load_table(ICEBERG_EPISODES_TABLE)
        meta = tbl.scan().to_arrow().to_pandas()
    except Exception as e:
        return {"error": f"Iceberg 查询失败: {e}"}
    out = {"total_episodes": int(len(meta))}
    if len(meta):
        out["by_embodiment"] = meta["embodiment_tag"].value_counts().to_dict()
        out["by_task"] = meta["task"].value_counts().head(20).to_dict()
        out["by_source"] = meta["source"].value_counts().to_dict()
        labeled = meta.dropna(subset=["success"])
        out["success_rate"] = (
            round(float(labeled["success"].mean()), 4) if len(labeled) else None
        )
    return _to_jsonable(out)


@app.get("/debug/stats")
def get_stats():
    """返回数据统计信息"""
    if _get_lance_table() is None:
        return {"error": "Table not initialized"}

    df = table.to_pandas()
    stats = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "scene_types": (
            df["scene_type"].value_counts().to_dict()
            if "scene_type" in df.columns
            else {}
        ),
        "quality_range": {
            "min": (
                float(df["quality_score"].min())
                if "quality_score" in df.columns
                else None
            ),
            "max": (
                float(df["quality_score"].max())
                if "quality_score" in df.columns
                else None
            ),
            "mean": (
                float(df["quality_score"].mean())
                if "quality_score" in df.columns
                else None
            ),
        },
        "weather_types": (
            df["weather"].value_counts().to_dict() if "weather" in df.columns else {}
        ),
    }
    return _to_jsonable(stats)


@app.get("/debug/data")
def get_raw_data(limit: int = Query(10)):
    """返回原始数据样本"""
    if _get_lance_table() is None:
        return {"error": "Table not initialized"}

    df = table.to_pandas().head(limit)
    return _to_jsonable(
        {
            "total_rows": table.count_rows(),
            "columns": list(df.columns),
            "sample": df.to_dict(orient="records"),
        }
    )
