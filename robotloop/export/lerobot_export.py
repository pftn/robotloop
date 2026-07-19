"""训练侧一键导出：RobotLoop 存储 → LeRobot 数据集。

这是"闭环"的出口半段：数据在湖里经过检索/质检后，按条件筛出子集，
物化成训练框架可直接消费的标准格式。

默认导出 **v2.1** —— 因为 π0/OpenPI、GR00T（N1.5/N1.6）当前都要求
LeRobot v2 布局；需要喂新版官方工具链时 ``version='v3.0'`` 一行切换。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from robotloop.schema.episode import DataSource, Episode, Step

logger = logging.getLogger("robotloop.export")


def episodes_from_store(
    store,
    filters: Optional[Dict[str, Any]] = None,
    episode_ids: Optional[Sequence[str]] = None,
) -> List[Episode]:
    """从存储（LocalStore / 冰山湖）重建 Episode 对象。"""
    meta_tbl = store.filter_meta(filters or {})
    meta_rows = meta_tbl.to_pylist()
    if episode_ids is not None:
        wanted = set(episode_ids)
        meta_rows = [r for r in meta_rows if r["episode_id"] in wanted]
    if not meta_rows:
        return []

    ids = [r["episode_id"] for r in meta_rows]
    steps_tbl = store.read_steps(ids) if hasattr(store, "read_steps") else None
    steps_by_ep: Dict[str, List[Dict[str, Any]]] = {eid: [] for eid in ids}
    if steps_tbl is not None:
        for row in steps_tbl.to_pylist():
            steps_by_ep.setdefault(row["episode_id"], []).append(row)

    episodes: List[Episode] = []
    for r in meta_rows:
        steps = []
        for sr in sorted(
            steps_by_ep.get(r["episode_id"], []), key=lambda x: x["frame_index"]
        ):
            ip = sr.get("image_paths") or {}
            if not isinstance(ip, dict):
                ip = dict(ip)  # parquet map 读出来是 tuple 列表
            steps.append(
                Step(
                    frame_index=int(sr["frame_index"]),
                    timestamp=float(sr["timestamp"]),
                    observation={"images": ip, "state": list(sr.get("state") or [])},
                    action=list(sr.get("action") or []),
                    reward=sr.get("reward"),
                    is_terminal=bool(sr.get("is_terminal", False)),
                    language_instruction=r["language_instruction"] or "",
                )
            )
        episodes.append(
            Episode(
                task=r["task"],
                language_instruction=r["language_instruction"] or r["task"],
                embodiment_tag=r["embodiment_tag"],
                source=DataSource(r.get("source") or "teleop"),
                success=r.get("success"),
                episode_id=r["episode_id"],
                duration=float(r.get("duration") or 0.0),
                dataset_name=r.get("dataset_name") or "",
                episode_index=int(r.get("episode_index") or 0),
                fps=float(r.get("fps") or 0.0),
                robot_type=r.get("robot_type") or "",
                steps=steps,
            ).validate()
        )
    return episodes


def _probe_image_size(ref: Any) -> Optional[tuple]:
    """探测一帧图像的 (H, W)（本地路径读文件头，s3:// 下载首帧，代价仅一张图）。
    读不到返回 None，由写出端退化为默认尺寸声明。"""
    import io

    try:
        if isinstance(ref, np.ndarray):
            return (int(ref.shape[0]), int(ref.shape[1]))
        from PIL import Image

        if isinstance(ref, str) and ref.startswith("s3://"):
            from robotloop.convert.lerobot_v21 import _s3_client

            rest = ref[len("s3://") :]
            bucket, key = rest.split("/", 1)
            obj = _s3_client().get_object(Bucket=bucket, Key=key)
            with Image.open(io.BytesIO(obj["Body"].read())) as im:
                return (im.size[1], im.size[0])
        if isinstance(ref, str):
            with Image.open(ref) as im:
                return (im.size[1], im.size[0])
        if isinstance(ref, (bytes, bytearray)):
            with Image.open(io.BytesIO(bytes(ref))) as im:
                return (im.size[1], im.size[0])
    except Exception:
        pass
    return None


def _wire_image_features(
    episodes: List[Episode],
    camera_map: Optional[Dict[str, str]] = None,
) -> "tuple[List[str], Optional[tuple]]":
    """把 steps 里的图像键从 topic 名改成 LeRobot 特征名，返回
    (video_keys, video_size)。

    解析端（MCAP/RLDS）的 observation["images"] 以 topic 为键（/cam/image），
    LeRobot 约定 observation.images.<name>；就地重命名后 write 端用同一
    名字查 steps 取帧、写 features。video_size 从第一帧可读图像探测，
    让 info.json 的 features shape 反映真实分辨率而非默认 480x640。

    camera_map: 显式键名映射 {topic 或特征名: 目标特征名} —— 对齐仿真
    环境的相机键（如 aloha env 只认 observation.images.top，topic 是
    /cam/image 的自采数据需要 {"/cam/image": "observation.images.top"}），
    否则训练出的 policy 在 eval 时找不到相机输入直接 KeyError。
    """
    explicit = dict(camera_map or {})
    mapping: Dict[str, str] = {}
    for ep in episodes:
        for s in ep.steps:
            images = s.observation.get("images") or {}
            for topic in list(images):
                if topic in explicit:
                    mapping[topic] = explicit[topic]
                elif topic.startswith("observation.images."):
                    mapping[topic] = topic
                elif topic not in mapping:
                    name = topic.strip("/").replace("/", "_") or "cam"
                    mapping[topic] = f"observation.images.{name}"
        if mapping:
            break
    if not mapping:
        return [], None
    # 重命名必须覆盖全部 episode（提前 break 会让后续 episode 键名失配、
    # 视频写出端查不到帧而生成空 mp4）
    for ep in episodes:
        for s in ep.steps:
            images = s.observation.get("images")
            if not images:
                continue
            s.observation["images"] = {mapping.get(k, k): v for k, v in images.items()}
    # 尺寸探测独立循环，找到第一帧可读图像即停
    video_size: Optional[tuple] = None
    mapped = set(mapping.values())
    for ep in episodes:
        for s in ep.steps:
            for k, v in (s.observation.get("images") or {}).items():
                if k in mapped:
                    video_size = _probe_image_size(v)
                    if video_size:
                        break
            if video_size:
                break
        if video_size:
            break
    video_keys = list(dict.fromkeys(mapping.values()))
    if video_size is None:
        logging.warning(
            "[Export] 未能探测图像尺寸（帧不可读？），features shape 将用默认值"
        )
    logging.info("[Export] image features: %s size=%s", video_keys, video_size)
    return video_keys, video_size


def export_to_lerobot(
    store,
    out_dir: str,
    filters: Optional[Dict[str, Any]] = None,
    episode_ids: Optional[Sequence[str]] = None,
    version: str = "v2.1",
    success_only: bool = False,
    min_frames: int = 5,
    fps: Optional[float] = None,
    robot_type: str = "",
    video_size: Optional[tuple] = None,
    camera_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """按条件从存储导出 LeRobot 数据集（训练框架即插即用）。

    参数：
        filters:      结构化过滤（同检索 DSL，如 {"embodiment_tag": "franka", "source": "sim"}）
        episode_ids:  显式指定导出哪些轨迹（检索结果直接喂进来）
        version:      "v2.1"（π0/GR00T）| "v3.0"（新版工具链）
        success_only: 质检联动 —— 只导出成功轨迹
        video_size:   显式 (H, W)：帧 resize 到该尺寸（对齐仿真 env 渲染分辨率，
                      如 aloha env 480x640）；缺省用首帧探测值
        camera_map:   显式相机键名映射（对齐仿真 env 相机键，如
                      {"/cam/image": "observation.images.top"}）
    """
    episodes = episodes_from_store(store, filters=filters, episode_ids=episode_ids)
    if not episodes:
        raise ValueError("过滤后没有可导出的 episode")

    if success_only:
        before = len(episodes)
        episodes = [ep for ep in episodes if ep.success is True]
        logger.info("success_only: %d → %d episodes", before, len(episodes))
    episodes = [ep for ep in episodes if ep.num_frames >= min_frames]
    if not episodes:
        raise ValueError("质检过滤后没有可导出的 episode")

    # 维度一致性前置校验：混本体（如 7 维 franka + 14 维 agibot）导出同一
    # 数据集必然失败，提前给出可操作提示而非 pyarrow 底层 ArrowInvalid
    from robotloop.schema.episode import check_consistent_dims, check_consistent_fps

    check_consistent_dims(episodes)
    check_consistent_fps(episodes)

    # 图像特征接线：解析端 observation["images"] 以 topic 为键（/cam/image），
    # LeRobot 特征名约定 observation.images.<name>。ACT 等策略要求至少一路
    # 图像输入（observation.state 不算数），不传 video_keys 导出的数据集
    # 训练时过不了 validate_features。
    video_keys, probed_size = _wire_image_features(episodes, camera_map=camera_map)
    video_size = video_size or probed_size

    if version in ("v2.1", "2.1", "v21"):
        from robotloop.convert.lerobot_v21 import write_lerobot_v21

        info = write_lerobot_v21(
            episodes,
            out_dir,
            fps=fps,
            robot_type=robot_type,
            video_keys=video_keys,
            video_size=video_size,
        )
    elif version in ("v3.0", "3.0", "v30"):
        from robotloop.convert.lerobot_v30 import write_lerobot_v30

        info = write_lerobot_v30(
            episodes,
            out_dir,
            fps=fps,
            robot_type=robot_type,
            video_keys=video_keys,
            video_size=video_size,
        )
    else:
        raise ValueError(f"未知版本: {version}")

    logger.info(
        "导出完成: %d episodes / %d frames → %s (%s)",
        info["total_episodes"],
        info["total_frames"],
        out_dir,
        version,
    )
    return info
