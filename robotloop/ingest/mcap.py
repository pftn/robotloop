"""MCAP 解析器 —— 真实设备数据的入口。

MCAP 是 ROS2 生态主流的日志容器（Foxglove 主推，rosbag2 官方插件支持）。
本模块做两件事：

1. 把 MCAP 里的多 topic 消息解成 ``TopicStream``（时间戳 + 业务值）
2. 调用 ``ingest.align`` 对齐成同步 Step，组装成 ``Episode`` 入库

消息抽取是注入式的（TopicExtractor）：不同机器人消息定义不同，
内置 JointState / Image / CompressedImage 三个常见抽取器，其余自定义。

依赖：``pip install mcap``（解码 protobuf/ros2msg 另需 mcap-protobuf-support /
mcap-ros2-support）。未安装时本模块其他功能不受影响。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from robotloop.ingest.align import AlignedFrame, TopicStream
from robotloop.schema.episode import DataSource, Episode, Step

# topic -> (decoded_msg, timestamp_s) -> value；返回 None 表示丢弃该消息
TopicExtractor = Callable[[Any, float], Optional[Any]]


# ---------------------------------------------------------------------------
# 内置消息抽取器
# ---------------------------------------------------------------------------
def joint_state_extractor(msg: Any, ts: float) -> Optional[List[float]]:
    """sensor_msgs/msg/JointState → 关节位置向量。"""
    pos = getattr(msg, "position", None)
    if pos is None and isinstance(msg, dict):
        pos = msg.get("position")
    if pos is None:
        return None
    return [float(p) for p in pos]


def image_extractor_factory(image_dir: str) -> TopicExtractor:
    """sensor_msgs/msg/(Compressed)Image → 落盘为文件，返回完整路径。

    图像不内嵌进 Episode —— 与 RobotLoop 存储分层一致（对象存储存像素，
    元数据存路径）。返回完整路径（而非裸文件名），入库端（EpisodeSink）
    才能找到文件上传 MinIO 并把 image_paths 重写为 s3:// URI。
    """
    os.makedirs(image_dir, exist_ok=True)
    counter = {"n": 0}

    def _extract(msg: Any, ts: float) -> Optional[str]:
        counter["n"] += 1
        fname = f"frame_{counter['n']:06d}.jpg"
        fpath = os.path.join(image_dir, fname)
        data = (
            getattr(msg, "data", None) if not isinstance(msg, dict) else msg.get("data")
        )
        if data is None:
            return None
        if isinstance(data, str):  # base64（JSONL 模拟源）
            import base64

            data = base64.b64decode(data)
        with open(fpath, "wb") as f:
            f.write(bytes(data))
        return fpath

    return _extract


# ---------------------------------------------------------------------------
# MCAP 读取
# ---------------------------------------------------------------------------
def read_mcap(
    path: str,
    extractors: Dict[str, TopicExtractor],
    use_log_time: bool = True,
) -> Dict[str, TopicStream]:
    """把 MCAP 文件解成 {topic: TopicStream}。

    参数：
        path:        .mcap 文件路径
        extractors:  {topic: 抽取器}；不在其中的 topic 直接跳过
        use_log_time: True 用 log_time（采集时刻），False 用 publish_time
    """
    try:
        from mcap.reader import make_reader
    except ImportError as e:
        raise ImportError(
            "需要 mcap: pip install mcap（解码另需 mcap-ros2-support 等）"
        ) from e

    try:  # 可选解码器：ros2msg / protobuf；没有则退回原始 bytes
        from mcap_ros2.decoder import DecoderFactory  # type: ignore

        def _iter(fp):
            with open(fp, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                yield from reader.iter_decoded_messages(topics=list(extractors))

    except ImportError:

        def _iter(fp):
            with open(fp, "rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages(
                    topics=list(extractors)
                ):
                    # 未解码 bytes 放在 decoded 位置，交给自定义 extractor 处理
                    yield schema, channel, message, message.data

    def _to_seconds(t: Any) -> float:
        # Message record 的 log_time/publish_time 是 int 纳秒；保留 datetime 兼容
        return t.timestamp() if hasattr(t, "timestamp") else t / 1e9

    buffers: Dict[str, Dict[str, List[Any]]] = {
        t: {"ts": [], "val": []} for t in extractors
    }
    for _schema, channel, message, decoded in _iter(path):
        topic = channel.topic
        if topic not in extractors:
            continue
        ts = _to_seconds(message.log_time if use_log_time else message.publish_time)
        val = extractors[topic](decoded, ts)
        if val is None:
            continue
        buffers[topic]["ts"].append(ts)
        buffers[topic]["val"].append(val)

    streams = {}
    for topic, buf in buffers.items():
        if not buf["ts"]:
            continue
        order = np.argsort(buf["ts"])
        streams[topic] = TopicStream(
            name=topic,
            timestamps=np.asarray(buf["ts"], dtype=np.float64)[order],
            values=[buf["val"][i] for i in order],
            kind=_guess_kind(topic),
        )
    return streams


def _guess_kind(topic: str) -> str:
    tl = topic.lower()
    if "image" in tl or "camera" in tl or "rgb" in tl:
        return "image"
    if "joint" in tl and ("state" in tl or "position" in tl):
        return "state"
    if "command" in tl or "action" in tl or "target" in tl:
        return "action"
    if "gripper" in tl or "grasp" in tl:
        return "gripper"
    return "generic"


# ---------------------------------------------------------------------------
# JSONL 模拟源（无真机时验证链路 / CI 用）
# ---------------------------------------------------------------------------
def read_jsonl_log(
    path: str, extractors: Dict[str, TopicExtractor]
) -> Dict[str, TopicStream]:
    """读取行式 JSON 日志（{"topic","timestamp","message"}），接口与 read_mcap 一致。

    用途：(a) 没有真机/MCAP 库时跑通全链路 demo；(b) 单元测试。
    """
    buffers: Dict[str, Dict[str, List[Any]]] = {
        t: {"ts": [], "val": []} for t in extractors
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            topic, ts, msg = rec["topic"], float(rec["timestamp"]), rec["message"]
            if topic not in extractors:
                continue
            val = extractors[topic](msg, ts)
            if val is None:
                continue
            buffers[topic]["ts"].append(ts)
            buffers[topic]["val"].append(val)
    streams = {}
    for topic, buf in buffers.items():
        if not buf["ts"]:
            continue
        order = np.argsort(buf["ts"])
        streams[topic] = TopicStream(
            name=topic,
            timestamps=np.asarray(buf["ts"], dtype=np.float64)[order],
            values=[buf["val"][i] for i in order],
            kind=_guess_kind(topic),
        )
    return streams


# ---------------------------------------------------------------------------
# 对齐帧 → Episode
# ---------------------------------------------------------------------------
def frames_to_episode(
    frames: List[AlignedFrame],
    *,
    task: str,
    language_instruction: str,
    embodiment_tag: str,
    source: DataSource = DataSource.TELEOP,
    success: Optional[bool] = None,
    image_topics: Optional[List[str]] = None,
    state_topic: Optional[str] = None,
    action_topic: Optional[str] = None,
    action_from_state_diff: bool = False,
    dataset_name: str = "",
    fps: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Episode:
    """把对齐后的帧组装成 Episode（训练语义上的 action 在这里确定）。

    action 两种来源：
    - ``action_topic``：包里直接录了控制指令（如 /joint_group_position_controller/commands）
    - ``action_from_state_diff=True``：a_t = s_{t+1} - s_t（遥操作日志的常用近似，
      即"下一步到达的状态"作为这一步的动作标签；最后一帧动作置 0 并标 terminal）
    """
    if not frames:
        raise ValueError("frames 不能为空")
    image_topics = image_topics or []
    n = len(frames)

    def _state_of(i: int) -> List[float]:
        if state_topic is None:
            return []
        v = frames[i].samples.get(state_topic)
        return [float(x) for x in v] if v is not None else []

    steps: List[Step] = []
    for i, fr in enumerate(frames):
        images = {t: fr.samples[t] for t in image_topics if t in fr.samples}
        state = _state_of(i)

        if action_topic:
            raw = fr.samples.get(action_topic)
            action = [float(x) for x in raw] if raw is not None else [0.0] * len(state)
        elif action_from_state_diff:
            if i + 1 < n:
                nxt = _state_of(i + 1)
                action = (
                    [b - a for a, b in zip(state, nxt)] if nxt else [0.0] * len(state)
                )
            else:
                action = [0.0] * len(state)
        else:
            action = [0.0] * len(state)

        steps.append(
            Step(
                frame_index=i,
                timestamp=fr.timestamp,
                observation={"images": images, "state": state},
                action=action,
                is_terminal=(i == n - 1),
                language_instruction=language_instruction,
            ).validate()
        )

    t0, t1 = frames[0].timestamp, frames[-1].timestamp
    return Episode(
        task=task,
        language_instruction=language_instruction,
        embodiment_tag=embodiment_tag,
        source=source,
        success=success,
        duration=float(t1 - t0),
        dataset_name=dataset_name,
        fps=fps or (n / (t1 - t0) if t1 > t0 else 0.0),
        steps=steps,
        metadata=metadata or {},
    ).validate()
