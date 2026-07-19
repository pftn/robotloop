"""LeRobot v2.1 <-> v3.0 文件级转换。

布局约定（与官方迁移脚本一致）：

v2.1::
    data/chunk-XXX/episode_YYYYYY.parquet      每 episode 一个文件
    meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}

v3.0::
    data/chunk-XXX/file-YYY.parquet            多 episode 聚合，按目标大小轮转
    meta/{info.json, tasks.parquet, stats.json}
    meta/episodes/chunk-XXX/file-YYY.parquet   episode 元数据聚合

文件级转换不过领域模型：parquet 帧数据原样搬运（只改文件组织），
元数据 jsonl <-> parquet 互转，全局统计由 per-episode 统计聚合。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger("lerobot_convert")

CHUNK_SIZE = 1000
V21_DATA_TEMPLATE = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
V30_DATA_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V30_EPISODES_META_TEMPLATE = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def detect_version(root: str) -> str:
    """读 meta/info.json 的 codebase_version 判定数据集版本。"""
    info_path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"{root} 缺少 meta/info.json，不是 LeRobot 数据集")
    info = json.load(open(info_path))
    ver = info.get("codebase_version", "")
    if ver.startswith("v2"):
        return "v2.1"
    if ver.startswith("v3"):
        return "v3.0"
    raise ValueError(f"无法识别的 codebase_version: {ver!r}（{info_path}）")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _episode_files_v21(root: str) -> List[str]:
    data_root = os.path.join(root, "data")
    files = []
    for dirpath, _, names in os.walk(data_root):
        for n in sorted(names):
            if n.startswith("episode_") and n.endswith(".parquet"):
                files.append(os.path.join(dirpath, n))
    return sorted(files)


def _data_files_v30(root: str) -> List[str]:
    data_root = os.path.join(root, "data")
    files = []
    for dirpath, _, names in os.walk(data_root):
        for n in sorted(names):
            if n.startswith("file-") and n.endswith(".parquet"):
                files.append(os.path.join(dirpath, n))
    return sorted(files)


# ---------------------------------------------------------------------------
# v2.1 -> v3.0
# ---------------------------------------------------------------------------
def convert_v21_to_v30(
    src: str,
    dst: str,
    data_file_size_mb: int = 100,
    copy_videos: bool = True,
) -> Dict[str, Any]:
    """v2.1 数据集目录 → v3.0 数据集目录。返回 v3.0 info.json 内容。"""
    if detect_version(src) != "v2.1":
        raise ValueError(f"{src} 不是 v2.1 数据集")
    info = json.load(open(os.path.join(src, "meta", "info.json")))
    episodes_meta = _read_jsonl(os.path.join(src, "meta", "episodes.jsonl"))
    tasks = _read_jsonl(os.path.join(src, "meta", "tasks.jsonl"))
    ep_stats = _read_jsonl(os.path.join(src, "meta", "episodes_stats.jsonl"))

    ep_files = _episode_files_v21(src)
    if len(ep_files) != len(episodes_meta):
        raise ValueError(
            f"episode 文件数({len(ep_files)}) 与 episodes.jsonl({len(episodes_meta)}) 不一致"
        )

    os.makedirs(dst, exist_ok=True)
    limit = data_file_size_mb * 1024 * 1024

    # ---- 数据：按目标文件大小贪心轮转聚合 ----
    chunk, file_idx, cur_bytes = 0, 0, 0
    pending: List[pa.Table] = []
    assignments: Dict[int, Dict[str, int]] = (
        {}
    )  # episode_index -> {chunk_index, file_index}
    from_idx = 0

    def _flush(chunk_i: int, file_i: int, tables: List[pa.Table]):
        out = pa.concat_tables(tables, promote_options="default")
        path = os.path.join(
            dst, V30_DATA_TEMPLATE.format(chunk_index=chunk_i, file_index=file_i)
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(out, path, compression="zstd")
        logger.info("wrote %s (%d rows)", path, out.num_rows)

    for ep_idx, fpath in enumerate(ep_files):
        tbl = pq.read_table(fpath)
        size = os.path.getsize(fpath)
        if cur_bytes + size > limit and pending:
            _flush(chunk, file_idx, pending)
            pending, cur_bytes = [], 0
            file_idx += 1
            if file_idx >= CHUNK_SIZE:
                chunk, file_idx = chunk + 1, 0
        assignments[ep_idx] = {
            "chunk_index": chunk,
            "file_index": file_idx,
            "dataset_from_index": from_idx,
        }
        pending.append(tbl)
        from_idx += tbl.num_rows
        cur_bytes += size
    if pending:
        _flush(chunk, file_idx, pending)

    # ---- 元数据：jsonl -> parquet ----
    # tasks.jsonl -> tasks.parquet
    tasks_tbl = (
        pa.Table.from_pylist(
            [{"task_index": t["task_index"], "task": t["task"]} for t in tasks]
        )
        if tasks
        else pa.table({"task_index": [], "task": []})
    )
    os.makedirs(os.path.join(dst, "meta"), exist_ok=True)
    pq.write_table(tasks_tbl, os.path.join(dst, "meta", "tasks.parquet"))

    # episodes.jsonl -> meta/episodes/chunk-000/file-000.parquet
    ep_rows = []
    running = 0
    for i, m in enumerate(episodes_meta):
        a = assignments[i]
        ep_rows.append(
            {
                "episode_index": m["episode_index"],
                "data/chunk_index": a["chunk_index"],
                "data/file_index": a["file_index"],
                "dataset_from_index": running,
                "dataset_to_index": running + m["length"],
                "tasks": m.get("tasks", []),
                "length": m["length"],
            }
        )
        running += m["length"]
    eps_meta_tbl = pa.Table.from_pylist(ep_rows) if ep_rows else pa.table({})
    meta_path = os.path.join(
        dst, V30_EPISODES_META_TEMPLATE.format(chunk_index=0, file_index=0)
    )
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    pq.write_table(eps_meta_tbl, meta_path)

    # episodes_stats.jsonl -> meta/stats.json（全局聚合）
    stats = _aggregate_stats(ep_stats)
    with open(os.path.join(dst, "meta", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # videos 目录原样复制（v3.0 视频聚合是独立步骤，不阻塞数据转换）
    if copy_videos and os.path.isdir(os.path.join(src, "videos")):
        shutil.copytree(
            os.path.join(src, "videos"), os.path.join(dst, "videos"), dirs_exist_ok=True
        )

    out_info = dict(info)
    out_info.update(
        {
            "codebase_version": "v3.0",
            "data_file_size_in_mb": data_file_size_mb,
            "data_path": V30_DATA_TEMPLATE,
            "video_path": info.get("video_path"),
        }
    )
    with open(os.path.join(dst, "meta", "info.json"), "w") as f:
        json.dump(out_info, f, indent=2, ensure_ascii=False)
    logger.info("v2.1 -> v3.0 done: %d episodes -> %s", len(episodes_meta), dst)
    return out_info


def _aggregate_stats(ep_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """per-episode 统计 → 全局统计（count 求和，min/max 取极值，mean/std 用矩合并）。"""
    if not ep_stats:
        return {}
    out: Dict[str, Any] = {}
    keys = ep_stats[0].get("stats", {}).keys()
    for key in keys:
        per = [e["stats"][key] for e in ep_stats if key in e.get("stats", {})]
        if not per:
            continue
        merged: Dict[str, Any] = {}
        counts = [
            int(
                s["count"][0] if isinstance(s.get("count"), list) else s.get("count", 0)
            )
            for s in per
        ]
        total = sum(counts)
        merged["count"] = [total]
        for stat_key in ("min", "max"):
            vals = [s[stat_key] for s in per if stat_key in s]
            if vals:
                import numpy as np

                merged[stat_key] = (np.min if stat_key == "min" else np.max)(
                    np.asarray(vals, dtype=np.float64), axis=0
                ).tolist()
        # mean/std 按 count 加权合并（通道独立）
        import numpy as np

        means = [np.asarray(s["mean"], dtype=np.float64) for s in per if "mean" in s]
        if means and total > 0:
            w = np.asarray(
                [c for c, s in zip(counts, per) if "mean" in s], dtype=np.float64
            )
            w = w / w.sum()
            gmean = (np.stack(means) * w[:, None]).sum(axis=0)
            merged["mean"] = gmean.tolist()
            if all("std" in s or "sq_sum" in s for s in per):
                stds = [
                    np.asarray(s.get("std", np.zeros_like(m)), dtype=np.float64)
                    for s, m in zip(per, means)
                ]
                var = (
                    np.stack([st**2 + m**2 for st, m in zip(stds, means)]) * w[:, None]
                ).sum(axis=0) - gmean**2
                merged["std"] = np.sqrt(np.clip(var, 0, None)).tolist()
        out[key] = merged
    return {"stats": out}


# ---------------------------------------------------------------------------
# v3.0 -> v2.1
# ---------------------------------------------------------------------------
def convert_v30_to_v21(src: str, dst: str, copy_videos: bool = True) -> Dict[str, Any]:
    """v3.0 数据集目录 → v2.1 数据集目录。返回 v2.1 info.json 内容。"""
    if detect_version(src) != "v3.0":
        raise ValueError(f"{src} 不是 v3.0 数据集")
    info = json.load(open(os.path.join(src, "meta", "info.json")))

    # episodes 元数据（parquet 聚合，可能多文件）
    ep_meta_rows: List[Dict[str, Any]] = []
    meta_root = os.path.join(src, "meta", "episodes")
    for dirpath, _, names in os.walk(meta_root):
        for n in sorted(names):
            if n.endswith(".parquet"):
                ep_meta_rows.extend(pq.read_table(os.path.join(dirpath, n)).to_pylist())
    ep_meta_rows.sort(key=lambda r: r["episode_index"])

    os.makedirs(dst, exist_ok=True)

    # ---- 数据：聚合文件按 episode_index 切回单 episode 文件 ----
    episodes_meta_jsonl: List[Dict[str, Any]] = []
    for m in ep_meta_rows:
        ep_idx = m["episode_index"]
        src_file = os.path.join(
            src,
            V30_DATA_TEMPLATE.format(
                chunk_index=m["data/chunk_index"], file_index=m["data/file_index"]
            ),
        )
        tbl = pq.read_table(src_file)
        mask = pc.equal(tbl.column("episode_index"), ep_idx)
        ep_tbl = tbl.filter(mask)

        chunk = ep_idx // CHUNK_SIZE
        out_path = os.path.join(
            dst, V21_DATA_TEMPLATE.format(chunk_index=chunk, episode_index=ep_idx)
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pq.write_table(ep_tbl, out_path, compression="zstd")
        episodes_meta_jsonl.append(
            {
                "episode_index": ep_idx,
                "tasks": m.get("tasks", []),
                "length": int(m.get("length", ep_tbl.num_rows)),
            }
        )

    # ---- 元数据：parquet -> jsonl ----
    tasks_path = os.path.join(src, "meta", "tasks.parquet")
    tasks = pq.read_table(tasks_path).to_pylist() if os.path.exists(tasks_path) else []
    _write_jsonl(os.path.join(dst, "meta", "tasks.jsonl"), tasks)
    _write_jsonl(os.path.join(dst, "meta", "episodes.jsonl"), episodes_meta_jsonl)

    # stats.json 全局统计 -> episodes_stats.jsonl 只能近似（全局无法还原 per-episode），
    # 如实记录：per-episode 统计标记为从全局复制 + note 字段说明
    stats_path = os.path.join(src, "meta", "stats.json")
    ep_stats = []
    if os.path.exists(stats_path):
        g = json.load(open(stats_path))
        for m in episodes_meta_jsonl:
            ep_stats.append(
                {
                    "episode_index": m["episode_index"],
                    "stats": g.get("stats", {}),
                    "note": "aggregated from v3.0 global stats; per-episode stats not recoverable",
                }
            )
    _write_jsonl(os.path.join(dst, "meta", "episodes_stats.jsonl"), ep_stats)

    if copy_videos and os.path.isdir(os.path.join(src, "videos")):
        shutil.copytree(
            os.path.join(src, "videos"), os.path.join(dst, "videos"), dirs_exist_ok=True
        )

    out_info = dict(info)
    out_info.update(
        {
            "codebase_version": "v2.1",
            "data_path": V21_DATA_TEMPLATE,
            "total_episodes": len(episodes_meta_jsonl),
            "total_chunks": (
                (len(episodes_meta_jsonl) - 1) // CHUNK_SIZE + 1
                if episodes_meta_jsonl
                else 0
            ),
            "chunks_size": CHUNK_SIZE,
        }
    )
    out_info.pop("data_file_size_in_mb", None)
    os.makedirs(os.path.join(dst, "meta"), exist_ok=True)
    with open(os.path.join(dst, "meta", "info.json"), "w") as f:
        json.dump(out_info, f, indent=2, ensure_ascii=False)
    logger.info("v3.0 -> v2.1 done: %d episodes -> %s", len(episodes_meta_jsonl), dst)
    return out_info
