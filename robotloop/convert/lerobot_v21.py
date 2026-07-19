"""LeRobot v2.1 读写 —— 每 episode 一个 parquet + jsonl 元数据。

v2.1 是当下训练侧兼容性的"硬通货"：π0/OpenPI、GR00T（N1.5/N1.6 的
GR00T-flavored LeRobot v2）都吃这个布局。

布局::

    root/
    ├── data/chunk-000/episode_000000.parquet     # 每 episode 一个文件
    ├── videos/chunk-000/<video_key>/episode_000000.mp4
    └── meta/
        ├── info.json          # codebase_version / fps / features / data_path 模板
        ├── tasks.jsonl        # {task_index, task}
        ├── episodes.jsonl     # {episode_index, tasks, length}
        └── episodes_stats.jsonl
"""

from __future__ import annotations

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
    read_jsonl,
    write_jsonl,
)
from robotloop.schema.episode import DataSource, Episode, Step

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
# 占位符名必须与 lerobot 加载器一致：lerobot 用
# data_path.format(episode_chunk=..., episode_index=...) 拼路径，
# 写 {chunk_index} 会在加载时 KeyError
DATA_PATH_TEMPLATE = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


def _chunk_of(episode_index: int) -> int:
    return episode_index // CHUNK_SIZE


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------
def write_lerobot_v21(
    episodes: Sequence[Episode],
    root: str,
    *,
    robot_type: str = "",
    fps: Optional[float] = None,
    video_keys: Optional[Sequence[str]] = None,
    video_size: Optional[tuple] = None,  # (H, W)，写视频时必填
) -> Dict[str, Any]:
    """把 Episode 列表写成 v2.1 数据集目录，返回 info.json 内容。"""
    from robotloop.schema.episode import check_consistent_dims, check_consistent_fps

    check_consistent_dims(list(episodes))
    check_consistent_fps(list(episodes))
    if not episodes:
        raise ValueError("episodes 不能为空")
    video_keys = list(video_keys or [])
    fps = fps or episodes[0].fps or 0.0
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)

    task_table = build_task_table(episodes)
    action_dim = episodes[0].action_dim
    state_dim = 0
    if episodes[0].steps and episodes[0].steps[0].observation.get("state"):
        state_dim = len(episodes[0].steps[0].observation["state"])

    global_index = 0
    episodes_meta: List[Dict[str, Any]] = []
    episodes_stats: List[Dict[str, Any]] = []
    total_videos = 0

    for ep_idx, ep in enumerate(episodes):
        chunk = _chunk_of(ep_idx)
        data_dir = os.path.join(root, "data", f"chunk-{chunk:03d}")
        os.makedirs(data_dir, exist_ok=True)

        actions = episode_action_array(ep)
        states = episode_state_array(ep)
        n = ep.num_frames

        frame = {
            "action": actions.tolist(),
            "timestamp": [s.timestamp - ep.steps[0].timestamp for s in ep.steps],
            "frame_index": list(range(n)),
            "episode_index": [ep_idx] * n,
            "index": list(range(global_index, global_index + n)),
            "task_index": [task_table[ep.task]] * n,
        }
        schema_fields = [
            pa.field("action", pa.list_(pa.float32(), action_dim)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
        if state_dim > 0:
            frame["observation.state"] = states.tolist()
            schema_fields.insert(
                1, pa.field("observation.state", pa.list_(pa.float32(), state_dim))
            )

        table = pa.table(frame, schema=pa.schema(schema_fields))
        pq.write_table(table, os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet"))
        global_index += n

        # 视频（可选）：images 里的每路相机按帧编码为 mp4；编码同时累积
        # 图像归一化统计（lerobot 训练侧 make_dataset 要求每个相机特征在
        # episodes_stats.jsonl 里有 mean/std，否则 KeyError）
        img_stats: Dict[str, Dict[str, Any]] = {}
        for vk in video_keys:
            vdir = os.path.join(root, "videos", f"chunk-{chunk:03d}", vk)
            os.makedirs(vdir, exist_ok=True)
            vpath = os.path.join(vdir, f"episode_{ep_idx:06d}.mp4")
            vstats = _write_episode_video(ep, vk, vpath, fps=fps, size=video_size)
            total_videos += 1
            if vstats is not None:
                img_stats[vk] = vstats

        episodes_meta.append({"episode_index": ep_idx, "tasks": [ep.task], "length": n})
        stat = {
            "episode_index": ep_idx,
            "stats": {"action": feature_stats([a for a in actions])},
        }
        if state_dim > 0:
            stat["stats"]["observation.state"] = feature_stats([s for s in states])
        for vk, vstats in img_stats.items():
            stat["stats"][vk] = vstats
        episodes_stats.append(stat)

    # ---- meta ----
    write_jsonl(
        os.path.join(root, "meta", "tasks.jsonl"),
        [
            {"task_index": i, "task": t}
            for t, i in sorted(task_table.items(), key=lambda kv: kv[1])
        ],
    )
    write_jsonl(os.path.join(root, "meta", "episodes.jsonl"), episodes_meta)
    write_jsonl(os.path.join(root, "meta", "episodes_stats.jsonl"), episodes_stats)

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
        "total_chunks": _chunk_of(len(episodes) - 1) + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE if video_keys else None,
        "features": features,
        # RobotLoop 扩展元数据（不破坏 v2.1 规范，训练侧忽略即可）
        "robotloop": {
            "episode_ids": [ep.episode_id for ep in episodes],
            "success": [ep.success for ep in episodes],
            "source": [ep.source.value for ep in episodes],
        },
    }
    with open(os.path.join(root, "meta", "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


class ImageStatsAcc:
    """图像归一化统计累积器（lerobot 约定：值域 [0,1]、min/max/mean/std
    形状 (3,1,1)、count 为帧数 —— 与官方 compute_episode_stats 对 video
    特征的产出一致；lerobot 加载端 _assert_type_and_shape 校验该形状，
    make_dataset 按相机 key 取 stats，缺失直接 KeyError）。"""

    def __init__(self) -> None:
        self.px_sum = np.zeros(3, dtype=np.float64)
        self.px_sq = np.zeros(3, dtype=np.float64)
        self.px_min = np.full(3, np.inf)
        self.px_max = np.full(3, -np.inf)
        self.n_px = 0
        self.n_frames = 0

    def add(self, frame: np.ndarray) -> None:
        """在最终写入的帧上累积（resize 后与 mp4 内容一致），归一化到 [0,1]。"""
        flat = (frame.astype(np.float64) / 255.0).reshape(-1, 3)
        self.px_sum += flat.sum(axis=0)
        self.px_sq += (flat**2).sum(axis=0)
        self.px_min = np.minimum(self.px_min, flat.min(axis=0))
        self.px_max = np.maximum(self.px_max, flat.max(axis=0))
        self.n_px += flat.shape[0]
        self.n_frames += 1

    def stats(self) -> Optional[Dict[str, Any]]:
        if self.n_frames == 0:
            return None
        mean = self.px_sum / self.n_px
        std = np.sqrt(np.maximum(self.px_sq / self.n_px - mean**2, 0.0))
        return {
            "min": self.px_min.reshape(3, 1, 1).tolist(),
            "max": self.px_max.reshape(3, 1, 1).tolist(),
            "mean": mean.reshape(3, 1, 1).tolist(),
            "std": std.reshape(3, 1, 1).tolist(),
            "count": [self.n_frames],
        }


def _write_episode_video(
    ep: Episode, video_key: str, out_path: str, *, fps: float, size: Optional[tuple]
) -> Optional[Dict[str, Any]]:
    """把 episode 中某路相机帧编码为 mp4（imageio + ffmpeg，惰性导入），
    并返回该 episode 的图像归一化统计（见 ImageStatsAcc）；无帧返回 None。"""
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401 —— 显式检查：缺它时 imageio 会把
    except ImportError as e:  # .mp4 静默降级给 tifffile 插件，
        raise ImportError(  # 直到 append_data 才炸出无关报错
            "写视频需要 imageio + imageio-ffmpeg: pip install imageio imageio-ffmpeg"
        ) from e

    acc = ImageStatsAcc()
    writer = imageio.get_writer(out_path, fps=fps or 10, codec="libx264", quality=7)
    try:
        for s in ep.steps:
            ref = s.observation.get("images", {}).get(video_key)
            if ref is None:
                continue
            try:
                frame = _load_image(ref, size)
            except Exception as e:
                raise RuntimeError(
                    f"编码视频失败: episode={ep.episode_id} key={video_key} "
                    f"frame_index={s.frame_index} ref={str(ref)[:120]}: {e}"
                ) from e
            writer.append_data(frame)
            acc.add(frame)
    finally:
        writer.close()

    return acc.stats()


_S3_CLIENT = None


def _s3_client():
    """惰性构造 boto3 s3 client（MinIO 兼容）。环境变量与平台其余组件对齐：
    S3_ENDPOINT_URL / AWS_ENDPOINT_URL / S3_ENDPOINT（docker-compose 约定），
    凭证 AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY 或 S3_ACCESS_KEY/S3_SECRET_KEY。"""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        try:
            import boto3
        except ImportError as e:
            raise ImportError("读取 s3:// 图像需要 boto3: pip install boto3") from e
        endpoint = (
            os.getenv("S3_ENDPOINT_URL")
            or os.getenv("AWS_ENDPOINT_URL")
            or os.getenv("S3_ENDPOINT")
        )
        ak = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY")
        sk = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_KEY")
        kwargs: Dict[str, Any] = {}
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        if ak and sk:
            kwargs.update(aws_access_key_id=ak, aws_secret_access_key=sk)
        _S3_CLIENT = boto3.client("s3", **kwargs)
    return _S3_CLIENT


def _load_image(ref: Any, size: Optional[tuple]) -> np.ndarray:
    import io

    from PIL import Image

    if isinstance(ref, str):
        if ref.startswith("s3://"):
            # 真实链路图像持久化在 MinIO：bucket/key 拆开拉字节
            rest = ref[len("s3://") :]
            bucket, key = rest.split("/", 1)
            obj = _s3_client().get_object(Bucket=bucket, Key=key)
            img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
        else:
            if not os.path.exists(ref):
                raise FileNotFoundError(
                    f"图像文件不存在: {ref} —— 图像需对导出环境可读；"
                    f"真实链路请确认帧图像已随 episode 持久化到共享存储（MinIO），"
                    f"worker 容器内的 /tmp 临时路径在导出侧不可读"
                )
            img = Image.open(ref).convert("RGB")
    elif isinstance(ref, np.ndarray):
        img = Image.fromarray(ref)
    else:  # bytes
        img = Image.open(io.BytesIO(ref)).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]))
    return np.asarray(img)


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------
def read_lerobot_v21(
    root: str,
    *,
    dataset_name: str = "",
    embodiment_tag: str = "",
    source: DataSource = DataSource.TELEOP,
) -> List[Episode]:
    """读 v2.1 数据集目录为 Episode 列表（图像以视频路径形式引用，不解码）。"""
    with open(os.path.join(root, "meta", "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise ValueError(
            f"不是 v2.1 数据集: codebase_version={info.get('codebase_version')}"
        )

    tasks = {
        r["task_index"]: r["task"]
        for r in read_jsonl(os.path.join(root, "meta", "tasks.jsonl"))
    }
    episodes_meta = read_jsonl(os.path.join(root, "meta", "episodes.jsonl"))
    rl_ext = info.get("robotloop", {})
    ep_ids = rl_ext.get("episode_ids", [])
    successes = rl_ext.get("success", [])
    sources = rl_ext.get("source", [])
    fps = info.get("fps", 0.0)
    video_keys = [
        k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"
    ]

    episodes: List[Episode] = []
    for em in episodes_meta:
        ep_idx = em["episode_index"]
        pq_path = os.path.join(
            root,
            "data",
            f"chunk-{_chunk_of(ep_idx):03d}",
            f"episode_{ep_idx:06d}.parquet",
        )
        tbl = pq.read_table(pq_path).to_pylist()
        task = em["tasks"][0] if em.get("tasks") else ""
        n = em["length"]

        steps: List[Step] = []
        for i, row in enumerate(tbl):
            images = {
                vk: os.path.join(
                    root,
                    "videos",
                    f"chunk-{_chunk_of(ep_idx):03d}",
                    vk,
                    f"episode_{ep_idx:06d}.mp4",
                )
                for vk in video_keys
            }
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
                (tbl[-1]["timestamp"] - tbl[0]["timestamp"]) if len(tbl) > 1 else 0.0
            ),
            dataset_name=dataset_name or os.path.basename(root.rstrip("/")),
            episode_index=ep_idx,
            fps=fps,
            robot_type=info.get("robot_type") or "",
            steps=steps,
        ).validate()
        episodes.append(ep)
    return episodes
