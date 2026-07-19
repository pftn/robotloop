"""LeRobot v3.0 读写 —— 多 episode 聚合大文件 + Parquet 元数据。

v3.0（lerobot >= 0.4.0）相对 v2.1 的核心变化：

- 数据：``data/chunk-XXX/file-YYY.parquet``，**多 episode 合并**到少量大文件
- 视频：``videos/<key>/chunk-XXX/file-YYY.mp4``，同样多 episode 合并，
  episode 边界靠元数据里的 from/to_timestamp 定位
- 元数据：JSONL → Parquet（``meta/episodes/``、``meta/tasks.parquet``、
  ``meta/stats.json`` 全局统计）
- info.json：``codebase_version: v3.0`` + 路径模板 + 目标文件大小

动机是万级 episode 时小文件爆炸（inode / I/O / Hub 流式加载）。
官方转换脚本：``lerobot.datasets.v30.convert_dataset_v21_to_v30``。

本模块的读写与 v2.1 模块同构，方便做 v2.1 ↔ v3.0 双向互转（走 Episode 中转）。
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robotloop.convert.common import (
    build_task_table,
    episode_action_array,
    episode_state_array,
    feature_stats,
    merge_stats,
)
from robotloop.convert.lerobot_v21 import ImageStatsAcc, _load_image
from robotloop.schema.episode import DataSource, Episode, Step

CODEBASE_VERSION = "v3.0"
CHUNK_SIZE = 1000  # 每个 chunk 最多文件数（官方 DEFAULT_CHUNK_SIZE）
DATA_FILE_SIZE_IN_MB = 100  # 官方 DEFAULT_DATA_FILE_SIZE_IN_MB
VIDEO_FILE_SIZE_IN_MB = 500  # 官方 DEFAULT_VIDEO_FILE_SIZE_IN_MB
DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TEMPLATE = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
EPISODES_META_TEMPLATE = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)


def _frame_schema(action_dim: int, state_dim: int) -> pa.Schema:
    fields = [
        pa.field("action", pa.list_(pa.float32(), action_dim)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ]
    if state_dim > 0:
        fields.insert(
            1, pa.field("observation.state", pa.list_(pa.float32(), state_dim))
        )
    return pa.schema(fields)


def _estimate_frame_bytes(action_dim: int, state_dim: int) -> int:
    return (action_dim + state_dim) * 4 + 64  # 向量 + 标量列 + parquet 开销的粗估


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------
def write_lerobot_v30(
    episodes: Sequence[Episode],
    root: str,
    *,
    robot_type: str = "",
    fps: Optional[float] = None,
    video_keys: Optional[Sequence[str]] = None,
    video_size: Optional[tuple] = None,
    data_file_size_mb: int = DATA_FILE_SIZE_IN_MB,
) -> Dict[str, Any]:
    """把 Episode 列表写成 v3.0 数据集目录，返回 info.json 内容。"""
    from robotloop.schema.episode import check_consistent_dims, check_consistent_fps

    check_consistent_dims(list(episodes))
    check_consistent_fps(list(episodes))
    if not episodes:
        raise ValueError("episodes 不能为空")
    video_keys = list(video_keys or [])
    fps = fps or episodes[0].fps or 0.0
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)

    task_table = build_task_table(episodes)
    action_dim = episodes[0].action_dim
    state_dim = 0
    if episodes[0].steps and episodes[0].steps[0].observation.get("state"):
        state_dim = len(episodes[0].steps[0].observation["state"])
    schema = _frame_schema(action_dim, state_dim)
    est_bytes = _estimate_frame_bytes(action_dim, state_dim)
    max_bytes = data_file_size_mb * 1024 * 1024

    # ---- 第一步：按目标文件大小贪心分配 episode → (chunk, file) ----
    assignments: List[tuple] = []  # (chunk_index, file_index)
    cur_chunk, cur_file, cur_bytes = 0, 0, 0
    for ep in episodes:
        need = ep.num_frames * est_bytes
        if cur_bytes > 0 and cur_bytes + need > max_bytes:
            cur_file += 1
            cur_bytes = 0
            if cur_file >= CHUNK_SIZE:
                cur_chunk += 1
                cur_file = 0
        assignments.append((cur_chunk, cur_file))
        cur_bytes += need

    # ---- 第二步：逐文件拼接写 parquet，同时收集 episode 元数据 ----
    global_index = 0
    ep_meta_rows: List[Dict[str, Any]] = []
    ep_stats: List[Dict[str, list]] = []
    state_stats: List[Dict[str, list]] = []

    file_groups: Dict[tuple, List[int]] = {}
    for ep_idx, loc in enumerate(assignments):
        file_groups.setdefault(loc, []).append(ep_idx)

    for (chunk, file), ep_indices in sorted(file_groups.items()):
        frames: Dict[str, list] = {f.name: [] for f in schema}
        file_row = 0
        for ep_idx in ep_indices:
            ep = episodes[ep_idx]
            actions = episode_action_array(ep)
            states = episode_state_array(ep)
            n = ep.num_frames
            t0 = ep.steps[0].timestamp
            frames["action"].extend(actions.tolist())
            if state_dim > 0:
                frames["observation.state"].extend(states.tolist())
            frames["timestamp"].extend([s.timestamp - t0 for s in ep.steps])
            frames["frame_index"].extend(range(n))
            frames["episode_index"].extend([ep_idx] * n)
            frames["index"].extend(range(global_index, global_index + n))
            frames["task_index"].extend([task_table[ep.task]] * n)

            ep_meta_rows.append(
                {
                    "episode_index": ep_idx,
                    "data/chunk_index": chunk,
                    "data/file_index": file,
                    "dataset_from_index": file_row,
                    "dataset_to_index": file_row + n,
                    "length": n,
                    "tasks": [ep.task],
                }
            )
            file_row += n
            global_index += n
            ep_stats.append(feature_stats([a for a in actions]))
            if state_dim > 0:
                state_stats.append(feature_stats([s for s in states]))

        data_dir = os.path.join(root, "data", f"chunk-{chunk:03d}")
        os.makedirs(data_dir, exist_ok=True)
        pq.write_table(
            pa.table(frames, schema=schema),
            os.path.join(data_dir, f"file-{file:03d}.parquet"),
        )

    # ---- 第三步：视频（可选，每路相机聚合为少量 mp4，记录时间区间）----
    total_videos = 0
    img_stats_lists: Dict[str, List[Dict[str, Any]]] = {}
    if video_keys:
        total_videos, img_stats_lists = _write_videos_v30(
            episodes, root, video_keys, fps, video_size, ep_meta_rows
        )

    # ---- 第四步：元数据 parquet / stats / info ----
    meta_dir = os.path.join(root, "meta", "episodes", "chunk-000")
    os.makedirs(meta_dir, exist_ok=True)
    meta_schema = pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("data/chunk_index", pa.int64()),
            pa.field("data/file_index", pa.int64()),
            pa.field("dataset_from_index", pa.int64()),
            pa.field("dataset_to_index", pa.int64()),
            pa.field("length", pa.int64()),
            pa.field("tasks", pa.list_(pa.string())),
        ]
    )
    ep_meta_rows.sort(key=lambda r: r["episode_index"])
    pq.write_table(
        pa.Table.from_pylist(ep_meta_rows, schema=meta_schema),
        os.path.join(meta_dir, "file-000.parquet"),
    )

    tasks_rows = [
        {"task_index": i, "task": t}
        for t, i in sorted(task_table.items(), key=lambda kv: kv[1])
    ]
    pq.write_table(
        pa.Table.from_pylist(
            tasks_rows,
            schema=pa.schema(
                [pa.field("task_index", pa.int64()), pa.field("task", pa.string())]
            ),
        ),
        os.path.join(root, "meta", "tasks.parquet"),
    )

    stats = {"action": merge_stats(ep_stats)}
    if state_dim > 0:
        stats["observation.state"] = merge_stats(state_stats)
    for vk, lst in img_stats_lists.items():
        if lst:
            stats[vk] = _merge_image_stats(lst)
    with open(os.path.join(root, "meta", "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    features: Dict[str, Any] = {
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if state_dim > 0:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": None,
        }
    for vk in video_keys:
        h, w = video_size or (480, 640)
        features[vk] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": fps, "video.codec": "h264"},
        }

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type
        or episodes[0].robot_type
        or episodes[0].embodiment_tag,
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": len(task_table),
        "total_videos": total_videos,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE if video_keys else None,
        "features": features,
        "data_files_size_in_mb": data_file_size_mb,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_IN_MB,
        "robotloop": {
            "episode_ids": [ep.episode_id for ep in episodes],
            "success": [ep.success for ep in episodes],
            "source": [ep.source.value for ep in episodes],
        },
    }
    with open(os.path.join(root, "meta", "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def _merge_image_stats(per_ep: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把各 episode 的图像 stats（(3,1,1) 嵌套 list）聚合为全局 stats。
    merge_stats 的广播假设是一维特征，这里先展平再还原。"""
    flat = [
        {
            k: (
                np.asarray(v, dtype=np.float64).reshape(-1).tolist()
                if k != "count"
                else v
            )
            for k, v in s.items()
        }
        for s in per_ep
    ]
    merged = merge_stats(flat)
    return {
        k: (
            np.asarray(v, dtype=np.float64).reshape(3, 1, 1).tolist()
            if k != "count"
            else v
        )
        for k, v in merged.items()
    }


def _write_videos_v30(
    episodes: Sequence[Episode],
    root: str,
    video_keys: Sequence[str],
    fps: float,
    video_size: Optional[tuple],
    ep_meta_rows: List[Dict[str, Any]],
) -> "tuple[int, Dict[str, List[Dict[str, Any]]]]":
    """每路相机把所有 episode 顺序拼进 file-000.mp4，记录 from/to_timestamp。
    返回 (视频文件数, {vk: [各 episode 的图像 stats]}) —— stats 供 meta/stats.json
    聚合（lerobot 训练侧要求每个相机特征有 mean/std，否则 KeyError）。"""
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401 —— 缺它时 imageio 静默降级 tifffile
    except ImportError as e:
        raise ImportError(
            "写视频需要 imageio + imageio-ffmpeg: pip install imageio imageio-ffmpeg"
        ) from e

    total = 0
    all_stats: Dict[str, List[Dict[str, Any]]] = {vk: [] for vk in video_keys}
    for vk in video_keys:
        vdir = os.path.join(root, "videos", vk, "chunk-000")
        os.makedirs(vdir, exist_ok=True)
        vpath = os.path.join(vdir, "file-000.mp4")
        writer = imageio.get_writer(vpath, fps=fps or 10, codec="libx264", quality=7)
        cursor = 0.0
        per_ep = {}
        try:
            for ep_idx, ep in enumerate(episodes):
                start = cursor
                acc = ImageStatsAcc()
                for s in ep.steps:
                    ref = s.observation.get("images", {}).get(vk)
                    if ref is None:
                        continue
                    try:
                        frame = _load_image(ref, video_size)
                    except Exception as e:
                        raise RuntimeError(
                            f"编码视频失败: episode={ep.episode_id} key={vk} "
                            f"frame_index={s.frame_index} ref={str(ref)[:120]}: {e}"
                        ) from e
                    writer.append_data(frame)
                    acc.add(frame)
                    cursor += 1.0 / (fps or 10)
                per_ep[ep_idx] = (start, cursor)
                st = acc.stats()
                if st is not None:
                    all_stats[vk].append(st)
        finally:
            writer.close()
        for row in ep_meta_rows:
            st, ed = per_ep.get(row["episode_index"], (0.0, 0.0))
            row[f"videos/{vk}/chunk_index"] = 0
            row[f"videos/{vk}/file_index"] = 0
            row[f"videos/{vk}/from_timestamp"] = st
            row[f"videos/{vk}/to_timestamp"] = ed
        total += 1
    return total, all_stats


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------
def read_lerobot_v30(
    root: str,
    *,
    dataset_name: str = "",
    embodiment_tag: str = "",
    source: DataSource = DataSource.TELEOP,
) -> List[Episode]:
    """读 v3.0 数据集目录为 Episode 列表。"""
    with open(os.path.join(root, "meta", "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise ValueError(
            f"不是 v3.0 数据集: codebase_version={info.get('codebase_version')}"
        )

    tasks_tbl = pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pylist()
    tasks = {r["task_index"]: r["task"] for r in tasks_tbl}

    meta_files = sorted(
        glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet"))
    )
    if not meta_files:
        raise ValueError("缺少 meta/episodes/ 元数据 parquet")
    ep_meta: List[Dict[str, Any]] = []
    for mf in meta_files:
        ep_meta.extend(pq.read_table(mf).to_pylist())
    ep_meta.sort(key=lambda r: r["episode_index"])

    rl_ext = info.get("robotloop", {})
    ep_ids = rl_ext.get("episode_ids", [])
    successes = rl_ext.get("success", [])
    sources = rl_ext.get("source", [])
    fps = info.get("fps", 0.0)
    video_keys = [
        k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"
    ]

    # 数据文件缓存：同一文件内的多 episode 只读一次
    data_cache: Dict[tuple, List[Dict[str, Any]]] = {}
    episodes: List[Episode] = []
    for em in ep_meta:
        ep_idx = int(em["episode_index"])
        loc = (int(em["data/chunk_index"]), int(em["data/file_index"]))
        if loc not in data_cache:
            data_cache[loc] = pq.read_table(
                os.path.join(
                    root, "data", f"chunk-{loc[0]:03d}", f"file-{loc[1]:03d}.parquet"
                )
            ).to_pylist()
        rows = data_cache[loc][
            int(em["dataset_from_index"]) : int(em["dataset_to_index"])
        ]

        task = em["tasks"][0] if em.get("tasks") else ""
        n = int(em["length"])
        steps: List[Step] = []
        for i, row in enumerate(rows):
            images = {}
            for vk in video_keys:
                vc = em.get(f"videos/{vk}/chunk_index", 0)
                vf = em.get(f"videos/{vk}/file_index", 0)
                images[vk] = os.path.join(
                    root, "videos", vk, f"chunk-{vc:03d}", f"file-{vf:03d}.mp4"
                )
            steps.append(
                Step(
                    frame_index=int(row["frame_index"]),
                    timestamp=float(row["timestamp"]),
                    observation={
                        "images": images,
                        "state": row.get("observation.state") or [],
                    },
                    action=[float(a) for a in row["action"]],
                    is_terminal=(i == n - 1),
                    language_instruction=task,
                )
            )
        ep = Episode(
            task=task,
            language_instruction=task,
            embodiment_tag=embodiment_tag or info.get("robot_type") or "unknown",
            source=DataSource(sources[ep_idx]) if ep_idx < len(sources) else source,
            success=successes[ep_idx] if ep_idx < len(successes) else None,
            episode_id=(
                ep_ids[ep_idx] if ep_idx < len(ep_ids) else f"lerobot_{ep_idx:06d}"
            ),
            duration=(
                (rows[-1]["timestamp"] - rows[0]["timestamp"]) if len(rows) > 1 else 0.0
            ),
            dataset_name=dataset_name or os.path.basename(root.rstrip("/")),
            episode_index=ep_idx,
            fps=fps,
            robot_type=info.get("robot_type") or "",
            steps=steps,
        ).validate()
        episodes.append(ep)
    return episodes
