"""统一灌库：任意来源的 Episode 列表 → 结构化湖 + 向量索引。

生产：Iceberg（episodes 元数据表）+ MinIO（frames/ Parquet 帧数据）+ Milvus（episode_vectors）
离线：LocalStore parquet 镜像

同一条 ``ingest_episodes`` 通路服务三种来源（Open X / LeRobot Hub /
AgiBot World）与自采数据 —— 入库即归一，检索零差异。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

import numpy as np

from robotloop.retrieval.encoder import encode_trajectory, get_text_encoder
from robotloop.schema.episode import Episode

logger = logging.getLogger("robotloop.datasets.load")


def compute_embeddings(episodes: Sequence[Episode], encoder=None):
    """文本（language_instruction）向量 + 轨迹向量（跨本体维度补零对齐）。"""
    encoder = encoder or get_text_encoder()
    text_vecs = np.stack(
        [np.asarray(encoder.encode(ep.language_instruction or ep.task), dtype=np.float32) for ep in episodes]
    )
    raw_traj = [encode_trajectory([s.action for s in ep.steps]) for ep in episodes]
    max_dim = max(len(v) for v in raw_traj) if raw_traj else 0
    traj_vecs = np.stack([np.pad(v, (0, max_dim - len(v))) for v in raw_traj]).astype(np.float32)
    return text_vecs, traj_vecs


def ingest_episodes(
    episodes: Sequence[Episode],
    store,
    encoder=None,
    batch_size: int = 128,
    episode_sink=None,
    milvus_client=None,
) -> Dict[str, Any]:
    """把 Episode 批量写入存储。

    - store（必需）：LocalStore（离线镜像）或 MilvusIcebergStore
    - episode_sink（可选）：robotloop.schema.sink.EpisodeSink —— 生产侧一体化
      写入（帧 → MinIO parquet，元数据 → Iceberg episodes 表）
    - milvus_client（可选）：同时写入 Milvus episode_vectors 集合
    """
    total_meta, total_steps = 0, 0
    for lo in range(0, len(episodes), batch_size):
        batch = list(episodes[lo: lo + batch_size])

        # 生产侧先落 Iceberg+MinIO（回填 parquet_path），再进镜像/向量库
        if episode_sink is not None:
            episode_sink.write(batch)

        text_vecs, traj_vecs = compute_embeddings(batch, encoder)
        meta_rows = [ep.meta_dict() for ep in batch]
        step_rows = [r for ep in batch for r in ep.step_dicts()]

        store.add(meta_rows, text_vecs, traj_vecs)
        if hasattr(store, "add_steps"):
            store.add_steps(step_rows)

        if milvus_client is not None:
            data = [
                {"episode_id": r["episode_id"], "embedding": v.tolist()}
                for r, v in zip(meta_rows, text_vecs)
            ]
            milvus_client.insert(collection_name="episode_vectors", data=data)

        total_meta += len(meta_rows)
        total_steps += len(step_rows)
        logger.info("ingested batch: %d episodes / %d steps", len(batch), len(step_rows))

    return {"episodes": total_meta, "steps": total_steps, "store_count": store.count()}
