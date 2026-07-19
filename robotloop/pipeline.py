"""高层管道：采集日志 → 对齐 → Episode，一步完成。

把 ingest（read/align）与 schema（Episode）串成一个调用，
供 CLI 与灌库脚本复用。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from robotloop.ingest.align import align_streams
from robotloop.ingest.mcap import (
    TopicExtractor,
    frames_to_episode,
    image_extractor_factory,
    joint_state_extractor,
    read_jsonl_log,
    read_mcap,
)
from robotloop.schema.episode import DataSource, Episode


def bag_to_episode(
    bag_path: str,
    *,
    camera_topics: List[str],
    state_topic: str,
    action_topic: Optional[str] = None,
    task: str,
    language_instruction: str,
    embodiment_tag: str,
    source: DataSource = DataSource.TELEOP,
    success: Optional[bool] = None,
    reference_topic: Optional[str] = None,
    target_fps: Optional[float] = None,
    tolerance: float = 0.050,
    image_dir: Optional[str] = None,
    extractors: Optional[Dict[str, TopicExtractor]] = None,
    dataset_name: str = "",
) -> Episode:
    """MCAP / rosbag2 / JSONL 日志 → 同步对齐的 Episode。

    - 图像 topic 用 ``image_extractor_factory`` 落盘（image_dir 默认 bag 同目录 images/）
    - state / action topic 默认按 JointState.position 抽取，可用 extractors 覆盖
    - 参考轴：reference_topic（缺省取第一个相机 topic）或 target_fps 重采样
    """
    image_dir = image_dir or os.path.join(os.path.dirname(os.path.abspath(bag_path)), "images")
    ext: Dict[str, TopicExtractor] = dict(extractors or {})
    for cam in camera_topics:
        ext.setdefault(cam, image_extractor_factory(os.path.join(image_dir, cam.strip("/").replace("/", "_"))))
    ext.setdefault(state_topic, joint_state_extractor)
    if action_topic:
        ext.setdefault(action_topic, joint_state_extractor)

    if bag_path.endswith(".mcap"):
        streams = read_mcap(bag_path, ext)
    elif bag_path.endswith(".jsonl"):
        streams = read_jsonl_log(bag_path, ext)
    else:
        from robotloop.ingest.rosbag2 import read_rosbag2

        streams = read_rosbag2(bag_path, ext)

    if not streams:
        raise ValueError(f"{bag_path} 中没有可抽取的 topic 数据")
    missing = [t for t in [state_topic, *camera_topics] if t not in streams]
    if missing:
        raise ValueError(f"日志中缺少必需 topic: {missing}（实际读到: {list(streams)}）")

    frames, report = align_streams(
        list(streams.values()),
        reference_topic=reference_topic or (camera_topics[0] if camera_topics else None),
        target_fps=None if (reference_topic or camera_topics) else (target_fps or 30.0),
        tolerance=tolerance,
        required=[state_topic, *camera_topics],
    )
    episode = frames_to_episode(
        frames,
        task=task,
        language_instruction=language_instruction,
        embodiment_tag=embodiment_tag,
        source=source,
        success=success,
        image_topics=camera_topics,
        state_topic=state_topic,
        action_topic=action_topic,
        action_from_state_diff=(action_topic is None),
        dataset_name=dataset_name or os.path.basename(bag_path),
        metadata={"alignment_report": report.as_dict(), "bag_path": bag_path},
    )
    return episode
