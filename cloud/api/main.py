import os, logging
from fastapi import FastAPI, Query, Request
import boto3
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
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")

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
if TABLE_NAME in db.table_names():
    table = db.open_table(TABLE_NAME)
    logger.info(f"Opened LanceDB table {TABLE_NAME}, rows: {table.count_rows()}")
else:
    logger.warning(f"LanceDB table {TABLE_NAME} not found. Run ray-processor first.")


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
    if table is None:
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
    return {
        "count": len(results),
        "results": results,
        "debug": debug_info,
        "suggestion": None if len(results) > 0 else "尝试放宽过滤条件",
    }


@app.get("/debug/stats")
def get_stats():
    """返回数据统计信息"""
    if table is None:
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
    return stats


@app.get("/debug/data")
def get_raw_data(limit: int = Query(10)):
    """返回原始数据样本"""
    if table is None:
        return {"error": "Table not initialized"}

    df = table.to_pandas().head(limit)
    return {
        "total_rows": table.count_rows(),
        "columns": list(df.columns),
        "sample": df.to_dict(orient="records"),
    }
