"""轨迹相似度去重 —— 数据集的"隐形重复"清理。

重复采集（同一遥操作员反复录同一动作）会让训练分布被少数模式主导。
做法：轨迹向量两两余弦相似度 > 阈值 → 判重；union-find 聚类，
每簇保留最早入库的一条。

轨迹向量默认用 retrieval.encoder.encode_trajectory 的统计摘要；
生产可换 TCN/Transformer 编码器，接口不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class DedupResult:
    kept_ids: List[str] = field(default_factory=list)
    duplicate_clusters: List[List[str]] = field(default_factory=list)  # 每簇含被保留的首元素
    removed_ids: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.kept_ids) + len(self.removed_ids),
            "kept": len(self.kept_ids),
            "removed": len(self.removed_ids),
            "clusters": len(self.duplicate_clusters),
        }


def dedup_by_similarity(
    episode_ids: Sequence[str],
    embeddings: np.ndarray,
    threshold: float = 0.98,
) -> DedupResult:
    """按轨迹向量相似度去重。

    参数：
        episode_ids: 与 embeddings 行对齐
        embeddings:  [N, D] 轨迹向量（无需预先归一化）
        threshold:   余弦相似度判重阈值（0.98 是保守起点；调到 0.95 更激进）
    """
    ids = list(episode_ids)
    mat = np.asarray(embeddings, dtype=np.float32)
    if len(ids) != mat.shape[0]:
        raise ValueError("episode_ids 与 embeddings 数量不一致")
    if len(ids) < 2:
        return DedupResult(kept_ids=ids)

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    sim = mat @ mat.T

    uf = _UnionFind(len(ids))
    iu = np.triu_indices(len(ids), k=1)
    dup_pairs = np.where(sim[iu] >= threshold)[0]
    for p in dup_pairs:
        uf.union(int(iu[0][p]), int(iu[1][p]))

    clusters: Dict[int, List[int]] = {}
    for i in range(len(ids)):
        clusters.setdefault(uf.find(i), []).append(i)

    result = DedupResult()
    for members in clusters.values():
        members.sort()  # 保留下标最小（最早入库）的一条
        keep = members[0]
        result.kept_ids.append(ids[keep])
        if len(members) > 1:
            cluster_ids = [ids[m] for m in members]
            result.duplicate_clusters.append(cluster_ids)
            result.removed_ids.extend(ids[m] for m in members[1:])
    return result
