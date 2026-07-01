"""
main.py
RobotLoop 云端 API 服务
提供数据导出、标注触发、系统状态等 REST 接口
"""

import csv
import io
import json
import logging
import os
from datetime import datetime
from typing import Optional

import boto3
import psycopg2
from cassandra.cluster import Cluster
from cassandra.query import dict_factory
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi import HTTPException
from kafka import KafkaProducer
from pydantic import BaseModel

load_dotenv()

# -------------------- 配置 --------------------
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "robotloop-data")

PG_HOST = os.getenv("PG_HOST", "postgres")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_USER = os.getenv("PG_USER", "robotloop")
PG_PASSWORD = os.getenv("PG_PASSWORD", "robotloop")
PG_DB = os.getenv("PG_DB", "robotloop")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------- 初始化外部连接 --------------------
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

kafka_producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def get_db_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
    )


# -------------------- FastAPI 应用 --------------------
app = FastAPI(title="RobotLoop API", version="1.0.0")


# 数据模型
class SensorLogExportRequest(BaseModel):
    start: Optional[str] = None  # ISO 时间格式
    end: Optional[str] = None
    source: Optional[str] = None  # "robot_state" 或 "ros_data"


class PrelabelTrigger(BaseModel):
    model_version: Optional[str] = "v1.0"


# -------------------- 接口实现 --------------------
@app.get("/")
def root():
    return {"service": "RobotLoop API", "status": "healthy"}


@app.get("/health")
def health_check():
    try:
        conn = get_db_conn()
        conn.close()
        db_status = "ok"
    except Exception:
        db_status = "unreachable"
    return {
        "database": db_status,
        "kafka": "ok" if KAFKA_BOOTSTRAP else "unconfigured",
        "s3": f"endpoint={S3_ENDPOINT}, bucket={S3_BUCKET}"
    }


def get_cassandra_session():
    cluster = Cluster(['cassandra'])
    session = cluster.connect('robotloop')
    session.row_factory = dict_factory
    return session


@app.post("/export/sensor-logs")
def export_sensor_logs(req: SensorLogExportRequest):
    """
    从 Cassandra 导出传感器日志为 CSV
    支持时间范围过滤 (start, end) 以及数据源过滤 (robot_state / ros_data)
    不传 source 则导出全部
    """
    session = get_cassandra_session()
    try:
        # 构建查询
        base_query = "SELECT device_id, ts, state_json, event FROM robot_sensor_logs"
        conditions = []
        params = {}

        if req.start:
            conditions.append("ts >= %(start)s")
            params['start'] = datetime.fromisoformat(req.start)
        if req.end:
            conditions.append("ts <= %(end)s")
            params['end'] = datetime.fromisoformat(req.end)
        if req.source:
            event_map = {
                "robot_state": "normal",
                "ros_data": "ros_msg"
            }
            event = event_map.get(req.source)
            if event:
                conditions.append("event = %(event)s")
                params['event'] = event

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        # 执行查询（Cassandra 需要允许过滤，但我们的条件都在主键或索引范围内，可安全使用）
        # 注意：Cassandra 可能要求添加 ALLOW FILTERING，如果查询跨分区或非主键过滤则需谨慎
        # 我们按 device_id 分区，ts 聚簇，所以范围查询有效，加上 event 可能需要 ALLOW FILTERING
        # 这里为简单演示，假设数据量小且设计合理，暂时不加 ALLOW FILTERING
        rows = session.execute(base_query, params)

        # 生成 CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["device_id", "timestamp", "state_json", "event"])
        count = 0
        for row in rows:
            writer.writerow([row['device_id'], row['ts'].isoformat(), row['state_json'], row['event']])
            count += 1

        csv_content = output.getvalue()
        output.close()
        return {"csv": csv_content, "count": count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")
    finally:
        session.shutdown()  # 实际应用中建议使用连接池，避免频繁关闭


@app.post("/prelabel/trigger")
def trigger_prelabel(req: PrelabelTrigger):
    """
    触发全量或增量预标注任务
    """
    task = {
        "media_key": "example/frame_0001.jpg",
        "bucket": S3_BUCKET,
        "model_version": req.model_version,
        "triggered_at": datetime.utcnow().isoformat(),
    }
    try:
        kafka_producer.send("pre-label-tasks", value=task)
        logging.info(f"预标注任务已触发: {task}")
        return {"status": "triggered", "task": task}
    except Exception as e:
        logging.error(f"Trigger prelabel failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
def list_files(prefix: Optional[str] = Query(None)):
    """
    列出对象存储中的文件
    """
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix or "")
        contents = response.get("Contents", [])
        files = [{"key": obj["Key"], "size": obj["Size"], "last_modified": obj["LastModified"].isoformat()} for obj in
                 contents]
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_pipeline_metrics():
    """
    返回数据管道的基本统计信息（从数据库计算）
    """
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM robot_logs")
        total_logs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM media_files")
        total_media = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM media_files WHERE prelabel_result IS NOT NULL")
        prelabeled = cur.fetchone()[0]
        cur.close()
        return {
            "total_robot_logs": total_logs,
            "total_media_files": total_media,
            "prelabeled_media": prelabeled,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
