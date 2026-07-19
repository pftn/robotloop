"""rosbag2 解析器 —— 与 MCAP 解析器共用 TopicStream / 对齐 / Episode 组装链路。

用纯 Python 的 ``rosbags`` 库读取（无需 ROS2 运行时）：
``pip install rosbags``

读取产出与 ``ingest.mcap.read_mcap`` 完全相同的 ``{topic: TopicStream}``，
之后的对齐与 Episode 组装代码零差异 —— 这是刻意的接口统一：
采集格式是工程细节，领域模型只认 TopicStream。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from robotloop.ingest.align import TopicStream
from robotloop.ingest.mcap import TopicExtractor, _guess_kind


def read_rosbag2(
    path: str,
    extractors: Dict[str, TopicExtractor],
) -> Dict[str, TopicStream]:
    """把 rosbag2 目录解成 {topic: TopicStream}。

    参数：
        path:       rosbag2 目录（含 metadata.yaml）
        extractors: {topic: 抽取器}；收到的是 rosbags 反序列化后的消息对象
    """
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ImportError("需要 rosbags: pip install rosbags") from e

    from pathlib import Path

    buffers: Dict[str, Dict[str, list]] = {t: {"ts": [], "val": []} for t in extractors}
    with AnyReader([Path(path)]) as reader:
        conns = [c for c in reader.connections if c.topic in extractors]
        for conn, timestamp, rawdata in reader.messages(connections=conns):
            ts = timestamp / 1e9
            msg = reader.deserialize(rawdata, conn.msgtype)
            val = extractors[conn.topic](msg, ts)
            if val is None:
                continue
            buffers[conn.topic]["ts"].append(ts)
            buffers[conn.topic]["val"].append(val)

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


def bag_to_streams(path: str, extractors: Dict[str, TopicExtractor]) -> Dict[str, TopicStream]:
    """按后缀自动分流 .mcap / rosbag2 目录。"""
    if path.endswith(".mcap"):
        from robotloop.ingest.mcap import read_mcap

        return read_mcap(path, extractors)
    return read_rosbag2(path, extractors)
