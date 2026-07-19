"""帧数据对象存储 —— 存储分层的物理实现：Iceberg 只存指针，帧走 Parquet 文件。

布局（与 LeRobot v3 文件块思路一致）：

    s3://{bucket}/frames/{episode_id}.parquet      # 一个 episode 一个文件

生产环境写 MinIO（boto3），本地开发/单测写本地目录（file:// 或直接路径），
两种后端同一套 API。episodes 表的 ``parquet_path`` 列记录完整 URI，
训练导出（export/lerobot_export.py）按指针直接读文件。
"""

from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .iceberg import steps_to_arrow

if TYPE_CHECKING:  # pragma: no cover
    from .episode import Episode

logger = logging.getLogger(__name__)

FRAMES_PREFIX = "frames"


class FrameStore:
    """帧 Parquet 读写器。backend="s3" 写 MinIO，backend="local" 写本地目录。"""

    def __init__(
        self,
        backend: str = "local",
        base_dir: str = "./frames",
        s3_client=None,
        bucket: str = "robotloop-data",
        prefix: str = FRAMES_PREFIX,
    ):
        self.backend = backend
        self.base_dir = base_dir
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if backend == "local":
            os.makedirs(os.path.join(base_dir, self.prefix), exist_ok=True)

    # ------------------------------------------------------------------ 写
    def write_episode_frames(self, episode: "Episode") -> str:
        """把一条 Episode 的全部帧写成一个 Parquet，返回 parquet_path 并回填到 episode。"""
        table = steps_to_arrow(episode.step_dicts())
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd")
        data = buf.getvalue()

        key = f"{self.prefix}/{episode.episode_id}.parquet"
        if self.backend == "s3":
            assert self.s3 is not None, "s3 backend 需要 s3_client"
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
            path = f"s3://{self.bucket}/{key}"
        else:
            path = os.path.join(self.base_dir, key)
            with open(path, "wb") as f:
                f.write(data)
        episode.parquet_path = path
        logger.debug("[FrameStore] %s -> %s (%d frames)", episode.episode_id, path, len(episode.steps))
        return path

    # ------------------------------------------------------------------ 读
    def read_frames(self, parquet_path: str) -> pa.Table:
        """按 parquet_path 读回帧级 Arrow 表。"""
        if parquet_path.startswith("s3://"):
            assert self.s3 is not None, "读取 s3:// 路径需要 s3_client"
            rest = parquet_path[len("s3://"):]
            bucket, key = rest.split("/", 1)
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            return pq.read_table(io.BytesIO(obj["Body"].read()))
        return pq.read_table(parquet_path)

    def read_episode_steps(self, parquet_path: str) -> list:
        """读回帧级行（dict 列表，map 列还原为 dict）。"""
        table = self.read_frames(parquet_path)
        rows = table.to_pylist()
        for r in rows:
            ip = r.get("image_paths") or []
            if isinstance(ip, list):
                r["image_paths"] = dict(ip)
        return rows


def make_s3_frame_store(
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str = "robotloop-data",
    prefix: str = FRAMES_PREFIX,
) -> FrameStore:
    """生产入口：构造指向 MinIO 的 FrameStore。"""
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return FrameStore(backend="s3", s3_client=s3, bucket=bucket, prefix=prefix)
