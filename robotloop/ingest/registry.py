"""解析器注册表 —— 按扩展名挂载解析器。

Ray Worker 收到 MinIO 事件后只做一件事::

    parser = PARSER_REGISTRY.get(ext_of(key))
    episodes = parser(local_path, ctx)

后续接 HDF5 / zarr / LeRobot 目录等新格式时，只需 ``@register_parser(".hdf5")``
挂一个新函数，Worker 主干零改动。

解析器统一签名::

    parser(local_path: str, ctx: ParseContext) -> List[Episode]

``ctx`` 携带对象 metadata（upload_mcap.py 写入的 x-amz-meta-*：
embodiment/task/instruction/source/camera-topic/joint-topic）与 bucket/key。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from robotloop.schema.episode import DataSource, Episode

logger = logging.getLogger(__name__)


@dataclass
class ParseContext:
    """一次解析调用的上下文（来自 MinIO 对象与其 metadata）。"""

    bucket: str = ""
    key: str = ""
    embodiment_tag: str = "unknown"
    task: str = "unknown_task"
    language_instruction: str = ""
    source: str = "teleop"
    camera_topic: str = "/top"
    joint_topic: str = "/joint_states"
    work_dir: str = "/tmp/robotloop_parse"
    success: Optional[bool] = None  # None = 未标注
    extra: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_s3_metadata(
        cls, bucket: str, key: str, metadata: Optional[Dict[str, str]] = None
    ) -> "ParseContext":
        md = {k.lower(): v for k, v in (metadata or {}).items()}
        raw_success = md.get("success", "").lower()
        return cls(
            bucket=bucket,
            key=key,
            embodiment_tag=md.get("embodiment-tag", "unknown"),
            task=md.get("task", "unknown_task"),
            language_instruction=md.get("language-instruction", ""),
            source=md.get("source", "teleop"),
            camera_topic=md.get("camera-topic", "/top"),
            joint_topic=md.get("joint-topic", "/joint_states"),
            success=(
                True
                if raw_success == "true"
                else (False if raw_success == "false" else None)
            ),
            extra=md,
        )


ParserFn = Callable[[str, ParseContext], List[Episode]]

PARSER_REGISTRY: Dict[str, ParserFn] = {}


def register_parser(*extensions: str):
    """装饰器：把解析函数挂到扩展名上（可一次挂多个）。"""

    def deco(fn: ParserFn) -> ParserFn:
        for ext in extensions:
            ext = ext.lower()
            if not ext.startswith("."):
                ext = "." + ext
            if ext in PARSER_REGISTRY:
                raise ValueError(
                    f"扩展名 {ext} 已注册到 {PARSER_REGISTRY[ext].__name__}"
                )
            PARSER_REGISTRY[ext] = fn
        return fn

    return deco


def get_parser(key: str) -> Optional[ParserFn]:
    """按对象 key 的扩展名取解析器；未注册返回 None。"""
    ext = os.path.splitext(key)[1].lower()
    return PARSER_REGISTRY.get(ext)


def registered_extensions() -> List[str]:
    return sorted(PARSER_REGISTRY)


# ---------------------------------------------------------------------------
# 内置解析器
# ---------------------------------------------------------------------------
@register_parser(".mcap", ".bag")
def parse_bag(local_path: str, ctx: ParseContext) -> List[Episode]:
    """MCAP / rosbag2 → 相机帧锚点 + 关节流最近邻对齐 → Episode。

    用 rosbags/mcap 纯 Python 库读包（服务器零 ROS 依赖）。
    """
    from robotloop.pipeline import bag_to_episode

    ep = bag_to_episode(
        local_path,
        camera_topics=[ctx.camera_topic],
        state_topic=ctx.joint_topic,
        task=ctx.task,
        language_instruction=ctx.language_instruction or ctx.task,
        embodiment_tag=ctx.embodiment_tag,
        source=DataSource(ctx.source),
        success=ctx.success,
        tolerance=0.050,  # 50ms 容差窗
        dataset_name=f"s3://{ctx.bucket}/{ctx.key}",
    )
    return [ep]


@register_parser(".jsonl")
def parse_jsonl(local_path: str, ctx: ParseContext) -> List[Episode]:
    """JSONL 采集日志（CI/联调模拟源）→ Episode。"""
    from robotloop.pipeline import bag_to_episode

    ep = bag_to_episode(
        local_path,
        camera_topics=[ctx.camera_topic],
        state_topic=ctx.joint_topic,
        task=ctx.task,
        language_instruction=ctx.language_instruction or ctx.task,
        embodiment_tag=ctx.embodiment_tag,
        source=DataSource(ctx.source),
        success=ctx.success,
        tolerance=0.050,
        dataset_name=f"s3://{ctx.bucket}/{ctx.key}",
    )
    return [ep]
