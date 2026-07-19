"""生产写入端 —— 复用 v1 的 Iceberg REST catalog + MinIO，落地 Episode 统一模型。

数据流（写库段）::

    Episode 列表
      ├─ 帧数据 ──► MinIO  s3://robotloop-data/frames/{episode_id}.parquet
      └─ 元数据 ──► Iceberg robotloop.episodes（REST catalog, iceberg-rest:8181）
                    行里只带 parquet_path 指针

catalog 连接参数与 v1 ``ray_pipeline.sync_to_iceberg`` 完全一致（REST catalog），
只是把写入目标从 scenes_meta 切到 episodes 表。
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

from .episode import Episode
from .frame_store import FrameStore, make_s3_frame_store
from .iceberg import create_tables, episodes_to_arrow

logger = logging.getLogger(__name__)


def load_rest_catalog(
    uri: Optional[str] = None,
    warehouse: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
):
    """加载 v1 的 Iceberg REST catalog（参数与 ray_pipeline.sync_to_iceberg 对齐）。"""
    from pyiceberg.catalog import load_catalog

    return load_catalog(
        "rest",
        **{
            "type": "rest",
            "uri": uri or os.getenv("ICEBERG_CATALOG_URI", "http://iceberg-rest:8181"),
            "warehouse": warehouse
            or os.getenv("ICEBERG_WAREHOUSE", "s3://iceberg-warehouse"),
            "s3.endpoint": s3_endpoint or os.getenv("S3_ENDPOINT", "http://minio:9000"),
            "s3.access-key-id": s3_access_key
            or os.getenv("S3_ACCESS_KEY", "minioadmin"),
            "s3.secret-access-key": s3_secret_key
            or os.getenv("S3_SECRET_KEY", "minioadmin"),
            "s3.path-style-access": "true",
            "s3.region": "us-east-1",
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
        },
    )


class EpisodeSink:
    """Episode 一体化写入端：帧 → MinIO Parquet，元数据 → Iceberg episodes 表。"""

    def __init__(self, catalog=None, frame_store: Optional[FrameStore] = None):
        self.catalog = catalog or load_rest_catalog()
        self.frames = frame_store or make_s3_frame_store(
            endpoint=os.getenv("S3_ENDPOINT", "http://minio:9000"),
            access_key=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            bucket=os.getenv("S3_BUCKET", "robotloop-data"),
        )
        self.table = create_tables(self.catalog)

    def write(self, episodes: Iterable[Episode]) -> int:
        eps = list(episodes)
        if not eps:
            return 0
        # 1) 帧数据：一个 episode 一个 Parquet → MinIO，回填 parquet_path
        for ep in eps:
            if ep.steps:
                self._stage_images(ep)  # 本地帧图像 → MinIO，路径就地重写为 s3://
                self.frames.write_episode_frames(ep)
        # 2) 元数据：append 进 Iceberg episodes 表（只带指针，不带帧）
        arrow = episodes_to_arrow([e.meta_dict() for e in eps])
        self.table.append(arrow)
        logger.info(
            "[EpisodeSink] wrote %d episodes -> Iceberg + MinIO frames", len(eps)
        )
        return len(eps)

    def _stage_images(self, ep: Episode) -> None:
        """本地帧图像 → MinIO（仅 s3 backend；详见 stage_images_to_s3）。"""
        if getattr(self.frames, "backend", None) != "s3" or self.frames.s3 is None:
            return
        stage_images_to_s3(ep, self.frames.s3, self.frames.bucket)


def stage_images_to_s3(ep: Episode, s3_client, bucket: str) -> int:
    """把 steps 里的本地图像上传到 MinIO，observation["images"] 的值就地
    重写为 s3://{bucket}/frames/{episode_id}/images/{topic}/{fname}。

    必须在写 frames parquet 之前做 —— parquet 里的 image_paths 要存重写
    后的 s3 URI，导出训练侧才能跨环境读到像素（worker 容器本地路径对
    导出环境不可读）。本地文件缺失时保留原值，由导出端给出带上下文的
    报错。返回上传帧数。
    """
    n_up = 0
    for s in ep.steps:
        images = s.observation.get("images") or {}
        for topic, ref in list(images.items()):
            if not isinstance(ref, str) or ref.startswith("s3://"):
                continue
            if not os.path.exists(ref):
                logger.warning("[EpisodeSink] 图像文件缺失，保留原路径: %s", ref)
                continue
            slug = topic.strip("/").replace("/", "_") or "cam"
            key = f"frames/{ep.episode_id}/images/{slug}/{os.path.basename(ref)}"
            with open(ref, "rb") as f:
                s3_client.put_object(Bucket=bucket, Key=key, Body=f.read())
            images[topic] = f"s3://{bucket}/{key}"
            n_up += 1
    if n_up:
        logger.info(
            "[EpisodeSink] %s: uploaded %d frame images -> s3://%s/frames/%s/images/",
            ep.episode_id,
            n_up,
            bucket,
            ep.episode_id,
        )
    return n_up
