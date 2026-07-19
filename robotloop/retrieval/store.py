"""检索存储后端：向量侧 + 结构化侧的两种实现。

- ``LocalStore``        本地 parquet 镜像（离线 demo / CI / 单元测试）
- ``MilvusIcebergStore`` 生产后端 —— Milvus/Zilliz 做 ANN，Iceberg 做结构化过滤，
                        与 RobotLoop 既有平台的分层一致

结构化过滤 DSL（两种后端共用语义）::

    {
      "embodiment_tag": "aloha",        # 等值
      "source": "sim",                 # teleop|sim|real
      "task": "pick_red_cube",
      "dataset_name": "openx/bridge_v2",
      "success": true,                 # bool
      "duration_min": 1.0, "duration_max": 60.0,
      "num_frames_min": 10,
    }
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from robotloop.schema.iceberg import EPISODES_PA_SCHEMA

EQ_FIELDS = {
    "embodiment_tag",
    "source",
    "task",
    "dataset_name",
    "robot_type",
    "episode_id",
}


def _arrow_mask(tbl: pa.Table, filters: Dict[str, Any]):
    """在 Table 上求布尔掩码（pc.compute 直算，兼容 Table.filter）。

    NULL（如 success 未标注）在过滤语义下视为不满足。
    """
    mask = None

    def _and(arr):
        nonlocal mask
        mask = arr if mask is None else pc.and_(mask, arr)

    for k, v in (filters or {}).items():
        if v is None:
            continue
        if k in EQ_FIELDS:
            _and(pc.equal(tbl.column(k), str(v)))
        elif k == "success":
            _and(pc.equal(tbl.column("success"), bool(v)))
        elif k == "duration_min":
            _and(pc.greater_equal(tbl.column("duration"), float(v)))
        elif k == "duration_max":
            _and(pc.less_equal(tbl.column("duration"), float(v)))
        elif k == "num_frames_min":
            _and(pc.greater_equal(tbl.column("num_frames"), int(v)))
        elif k == "num_frames_max":
            _and(pc.less_equal(tbl.column("num_frames"), int(v)))
        else:
            raise ValueError(f"不支持的过滤字段: {k}")
    if mask is None:
        mask = pa.array([True] * tbl.num_rows)
    return pc.fill_null(mask, False)


def _iceberg_row_filter(filters: Dict[str, Any]) -> Optional[str]:
    """pyiceberg row_filter SQL 字符串；无条件时返回 None（不传 row_filter）——
    pyiceberg 的表达式 parser 只接受"列名 op 字面量"形式，"1=1" 之类的
    常量表达式会 ParseException。"""
    clauses = []
    for k, v in (filters or {}).items():
        if v is None:
            continue
        if k in EQ_FIELDS:
            clauses.append(f"{k} = '{str(v).replace(chr(39), '')}'")
        elif k == "success":
            clauses.append(f"success = {'true' if v else 'false'}")
        elif k == "duration_min":
            clauses.append(f"duration >= {float(v)}")
        elif k == "duration_max":
            clauses.append(f"duration <= {float(v)}")
        elif k == "num_frames_min":
            clauses.append(f"num_frames >= {int(v)}")
        elif k == "num_frames_max":
            clauses.append(f"num_frames <= {int(v)}")
        else:
            raise ValueError(f"不支持的过滤字段: {k}")
    return " AND ".join(clauses) if clauses else None


class LocalStore:
    """本地镜像：episodes.parquet（结构化）+ embeddings.parquet（向量）+ frames/（帧级）。

    存储分层在本地镜像同样落地：帧数据一个 episode 一个 Parquet 文件存
    ``frames/{episode_id}.parquet``，episodes.parquet 里只存 parquet_path
    指针 —— 与生产侧 MinIO 布局完全一致，本地 demo 与线上行为同构。
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._meta_file = os.path.join(path, "episodes.parquet")
        self._emb_file = os.path.join(path, "embeddings.parquet")
        from robotloop.schema.frame_store import FrameStore

        self._frames = FrameStore(backend="local", base_dir=path)

    # ---------- 写 ----------
    def add(
        self,
        meta_rows: List[Dict[str, Any]],
        text_vecs: np.ndarray,
        traj_vecs: np.ndarray,
    ) -> None:
        from robotloop.schema.iceberg import episodes_to_arrow

        new_meta = episodes_to_arrow(meta_rows)
        emb_tbl = pa.table(
            {
                "episode_id": [r["episode_id"] for r in meta_rows],
                "text_embedding": [v.tolist() for v in text_vecs],
                "traj_embedding": [v.tolist() for v in traj_vecs],
            }
        )
        for file, new in ((self._meta_file, new_meta), (self._emb_file, emb_tbl)):
            if os.path.exists(file):
                old = pq.read_table(file)
                # 按 episode_id 去重（幂等灌库：同 id 覆盖）
                existing = set(new.column("episode_id").to_pylist())
                keep = pc.invert(pc.field("episode_id").isin(existing))
                new = pa.concat_tables(
                    [old.filter(keep), new], promote_options="default"
                )
            pq.write_table(new, file)

    def add_steps(self, step_rows: List[Dict[str, Any]]) -> None:
        """帧级数据落盘：按 episode 分组，一组一个 Parquet 文件，路径回填 meta。

        幂等：同 episode_id 重写整个帧文件（episode 是不可变单元）。
        """
        if not step_rows:
            return
        from collections import defaultdict

        from robotloop.schema.iceberg import FRAME_PARQUET_SCHEMA, steps_to_arrow

        by_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in step_rows:
            by_ep[r["episode_id"]].append(r)

        paths: Dict[str, str] = {}
        for eid, rows in by_ep.items():
            table = steps_to_arrow(rows)
            key = f"frames/{eid}.parquet"
            fpath = os.path.join(self.path, key)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            pq.write_table(table, fpath, compression="zstd")
            paths[eid] = fpath

        # 回填 episodes.parquet 的 parquet_path 列
        if os.path.exists(self._meta_file):
            meta = pq.read_table(self._meta_file)
            eids = meta.column("episode_id").to_pylist()
            pp = [
                paths.get(e)
                or (
                    meta.column("parquet_path")[i].as_py()
                    if "parquet_path" in meta.column_names
                    else ""
                )
                for i, e in enumerate(eids)
            ]
            if "parquet_path" in meta.column_names:
                meta = meta.set_column(
                    meta.column_names.index("parquet_path"),
                    "parquet_path",
                    pa.array(pp, type=pa.string()),
                )
            else:
                meta = meta.append_column(
                    "parquet_path", pa.array(pp, type=pa.string())
                )
            pq.write_table(meta, self._meta_file)

    # ---------- 读 ----------
    def count(self) -> int:
        if not os.path.exists(self._meta_file):
            return 0
        return pq.read_table(self._meta_file).num_rows

    def filter_meta(self, filters: Optional[Dict[str, Any]]) -> pa.Table:
        if not os.path.exists(self._meta_file):
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in EPISODES_PA_SCHEMA}
            )
        tbl = pq.read_table(self._meta_file)
        return tbl.filter(_arrow_mask(tbl, filters or {}))

    def all_meta(self) -> pa.Table:
        return self.filter_meta({})

    def read_steps(self, episode_ids: Optional[Sequence[str]] = None) -> pa.Table:
        """读帧级数据：从 meta 表取 parquet_path 指针，逐文件读出后拼接。"""
        from robotloop.schema.iceberg import FRAME_PARQUET_SCHEMA

        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in FRAME_PARQUET_SCHEMA}
        )
        if not os.path.exists(self._meta_file):
            return empty
        meta = pq.read_table(self._meta_file)
        if "parquet_path" not in meta.column_names:
            return empty
        if episode_ids is not None:
            meta = meta.filter(pc.field("episode_id").isin(list(episode_ids)))
        tables = []
        for p in meta.column("parquet_path").to_pylist():
            if p and os.path.exists(p):
                tables.append(pq.read_table(p))
        if not tables:
            return empty
        out = pa.concat_tables(tables, promote_options="default")
        return out.sort_by([("episode_id", "ascending"), ("frame_index", "ascending")])

    def embeddings(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """返回 (episode_ids, text_vecs[N,512], traj_vecs[N,D])。"""
        if not os.path.exists(self._emb_file):
            return (
                [],
                np.zeros((0, 512), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
            )
        tbl = pq.read_table(self._emb_file)
        ids = tbl.column("episode_id").to_pylist()
        text = np.asarray(tbl.column("text_embedding").to_pylist(), dtype=np.float32)
        traj = np.asarray(tbl.column("traj_embedding").to_pylist(), dtype=np.float32)
        return ids, text, traj

    def search_vectors(
        self, vec: np.ndarray, top_k: int, field: str = "text_embedding"
    ) -> List[Tuple[str, float]]:
        """暴力余弦（demo 规模足够；生产走 Milvus 后端）。"""
        if not os.path.exists(self._emb_file):
            return []
        tbl = pq.read_table(self._emb_file)
        ids = tbl.column("episode_id").to_pylist()
        mat = np.asarray(tbl.column(field).to_pylist(), dtype=np.float32)
        if mat.size == 0:
            return []
        v = vec.astype(np.float32)
        vn = np.linalg.norm(v)
        if vn > 0:
            v = v / vn
        mn = np.linalg.norm(mat, axis=1, keepdims=True)
        mn[mn == 0] = 1.0
        sims = (mat / mn) @ v
        order = np.argsort(-sims)[:top_k]
        return [(ids[i], float(sims[i])) for i in order]


class MilvusIcebergStore:
    """生产后端：Milvus/Zilliz（向量 ANN）+ Iceberg（结构化过滤）。

    复用 RobotLoop 平台环境变量：MILVUS_URI / MILVUS_TOKEN / ICEBERG_*。
    """

    COLLECTION = "episode_vectors"

    def __init__(
        self,
        milvus_uri: Optional[str] = None,
        milvus_token: Optional[str] = None,
        iceberg_catalog_uri: Optional[str] = None,
        iceberg_warehouse: Optional[str] = None,
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        episodes_table: str = "robotloop.episodes",
    ):
        from pyiceberg.catalog import load_catalog

        # Milvus 惰性连接：导出/结构化过滤只走 Iceberg+MinIO，不该被
        # pymilvus 依赖或 MILVUS_URI 缺失卡住；首次向量操作时才真正连接
        self._milvus_uri = milvus_uri
        self._milvus_token = milvus_token
        self._milvus_client = None
        self._catalog = load_catalog(
            "rest",
            **{
                "type": "rest",
                "uri": iceberg_catalog_uri
                or os.getenv("ICEBERG_CATALOG_URI", "http://iceberg-rest:8181"),
                "warehouse": iceberg_warehouse
                or os.getenv("ICEBERG_WAREHOUSE", "s3://iceberg-warehouse"),
                "s3.endpoint": s3_endpoint
                or os.getenv("S3_ENDPOINT", "http://minio:9000"),
                "s3.access-key-id": s3_access_key
                or os.getenv("S3_ACCESS_KEY", "minioadmin"),
                "s3.secret-access-key": s3_secret_key
                or os.getenv("S3_SECRET_KEY", "minioadmin"),
                "s3.path-style-access": "true",
                "s3.region": "us-east-1",
            },
        )
        self._episodes_table = episodes_table

    @property
    def _milvus(self):
        """首次向量操作时连接 Milvus（见 __init__ 注释）。"""
        if self._milvus_client is None:
            from pymilvus import MilvusClient

            self._milvus_client = MilvusClient(
                uri=self._milvus_uri or os.environ["MILVUS_URI"],
                token=self._milvus_token or os.environ.get("MILVUS_TOKEN"),
            )
        return self._milvus_client

    def count(self) -> int:
        return self._catalog.load_table(self._episodes_table).scan().to_arrow().num_rows

    def filter_meta(self, filters: Optional[Dict[str, Any]]) -> pa.Table:
        tbl = self._catalog.load_table(self._episodes_table)
        rf = _iceberg_row_filter(filters or {})
        scan = tbl.scan(row_filter=rf) if rf else tbl.scan()
        return scan.to_arrow()

    def all_meta(self) -> pa.Table:
        return self.filter_meta({})

    def read_steps(self, episode_ids: Optional[Sequence[str]] = None) -> pa.Table:
        """生产侧帧读取：Iceberg 过滤出 parquet_path，从 MinIO 逐文件读回拼接。"""
        from robotloop.schema.frame_store import make_s3_frame_store
        from robotloop.schema.iceberg import FRAME_PARQUET_SCHEMA

        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in FRAME_PARQUET_SCHEMA}
        )
        filters: Dict[str, Any] = {}
        meta = self.filter_meta(filters)
        if episode_ids is not None:
            meta = meta.filter(pc.field("episode_id").isin(list(episode_ids)))
        if meta.num_rows == 0 or "parquet_path" not in meta.column_names:
            return empty
        fs = make_s3_frame_store(
            endpoint=os.getenv("S3_ENDPOINT", "http://minio:9000"),
            access_key=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        )
        tables = []
        for p in meta.column("parquet_path").to_pylist():
            if p:
                tables.append(fs.read_frames(p))
        if not tables:
            return empty
        out = pa.concat_tables(tables, promote_options="default")
        return out.sort_by([("episode_id", "ascending"), ("frame_index", "ascending")])

    def search_vectors(
        self, vec: np.ndarray, top_k: int, field: str = "text_embedding"
    ) -> List[Tuple[str, float]]:
        res = self._milvus.search(
            collection_name=self.COLLECTION,
            data=[vec.astype(np.float32).tolist()],
            limit=top_k,
            output_fields=["episode_id"],
        )
        out = []
        for hit in res[0]:
            hid = hit.get("id") or hit.get("pk")
            score = hit.get("distance", 0.0)
            out.append((str(hid), float(score)))
        return out
