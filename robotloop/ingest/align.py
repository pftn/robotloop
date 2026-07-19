"""多 topic 时间戳对齐 —— 机器人采集侧最痛的工程问题。

真实设备上各传感器各自为政地打时间戳：

- 相机 30Hz（周期 ~33.3ms，且有抖动/掉帧）
- 关节状态 500Hz（周期 2ms）
- 夹爪/力觉 100Hz...

训练要的是"同一逻辑时刻"的 (image, state, action) 三元组。本模块把
异步多速率的 topic 流对齐成一条同步时间轴上的 Step 序列。

对齐策略：
- ``nearest``       取时间戳最近的样本，偏差超过 ``tolerance`` 则该帧判为缺失
- ``latest_before`` 取不晚于参考时刻的最近样本（rosbag 回放/控制常用的 asof 语义）

核心匹配用 ``pandas.merge_asof``，容差窗默认 50ms，超窗丢帧。

参考时间轴两种取法：
- 以某个 topic 为基准（``reference_topic``，通常选相机 —— 帧是训练的天然节拍，
  相机帧为锚点 + 关节流最近邻对齐）
- 按目标 fps 重采样（``target_fps``，在各 topic 时间范围的交集上等距生成）

时钟源注意（真实场景的常见问题）：rosbag 若开 use_sim_time，
消息 header.stamp 与 /clock 同源，直接对齐即可；若 header 是墙钟而包内
时间是仿真钟，必须在解析层（ingest.mcap / ingest.rosbag2）先统一时钟源
再送进本模块 —— 对齐层假设所有 stream 已在同一时钟域。

每个对齐结果附 AlignmentReport：各 topic 的匹配率、平均/最大偏差、丢弃帧数 ——
对齐质量可观测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class TopicStream:
    """一条 topic 的时间序列（已从 MCAP/rosbag2 解出）。"""

    name: str
    timestamps: np.ndarray  # 单调递增，单位秒
    values: Sequence[Any]  # 与 timestamps 等长；图像帧可以是路径/bytes
    kind: str = "generic"  # image | state | action | gripper | generic

    def __post_init__(self):
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if len(self.timestamps) != len(self.values):
            raise ValueError(
                f"topic {self.name}: timestamps({len(self.timestamps)}) 与 values({len(self.values)}) 不等长"
            )
        if len(self.timestamps) > 1 and np.any(np.diff(self.timestamps) < 0):
            raise ValueError(f"topic {self.name}: 时间戳必须单调递增")

    @property
    def nominal_hz(self) -> float:
        """名义频率（周期中位数的倒数），用于报告与频率异常检测。"""
        if len(self.timestamps) < 2:
            return 0.0
        dt = np.median(np.diff(self.timestamps))
        return float(1.0 / dt) if dt > 0 else 0.0


@dataclass
class TopicAlignStat:
    topic: str
    matched: int = 0
    missing: int = 0
    mean_offset_ms: float = 0.0
    max_offset_ms: float = 0.0
    nominal_hz: float = 0.0


@dataclass
class AlignmentReport:
    reference: str
    num_frames: int
    stats: Dict[str, TopicAlignStat] = field(default_factory=dict)
    dropped_frames: int = 0  # 有任一必需 topic 缺失而被丢弃的参考帧数

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "num_frames": self.num_frames,
            "dropped_frames": self.dropped_frames,
            "stats": {
                k: {
                    "matched": v.matched,
                    "missing": v.missing,
                    "mean_offset_ms": round(v.mean_offset_ms, 3),
                    "max_offset_ms": round(v.max_offset_ms, 3),
                    "nominal_hz": round(v.nominal_hz, 2),
                }
                for k, v in self.stats.items()
            },
        }


@dataclass
class AlignedFrame:
    """对齐后的一帧（Step 的雏形）。"""

    frame_index: int
    timestamp: float  # 统一时刻（秒）
    samples: Dict[str, Any]  # topic -> 对齐到的样本值
    offsets_ms: Dict[str, float]  # topic -> 样本与统一时刻的偏差（ms）
    missing: List[str] = field(default_factory=list)  # 该帧缺失的 topic


def _match_indices(
    ref_ts: np.ndarray,
    topic_ts: np.ndarray,
    strategy: str,
    tolerance: float,
):
    """对每条参考时间戳，在 topic 时间轴上找匹配样本下标。

    用 pandas.merge_asof 实现：
    - nearest       -> direction="nearest"
    - latest_before -> direction="backward"
    tolerance 直接传给 merge_asof，超窗的行匹配结果为 NaN。

    返回 (indices, offsets)：indices[i] = -1 表示第 i 个参考时刻无可用样本。
    """
    import pandas as pd

    idx = np.full(len(ref_ts), -1, dtype=np.int64)
    off = np.full(len(ref_ts), np.inf, dtype=np.float64)
    if len(topic_ts) == 0 or len(ref_ts) == 0:
        return idx, off

    direction = {"nearest": "nearest", "latest_before": "backward"}.get(strategy)
    if direction is None:
        raise ValueError(f"未知对齐策略: {strategy}（可选 nearest / latest_before）")

    ref_df = pd.DataFrame({"t": ref_ts})
    top_df = pd.DataFrame(
        {"t": topic_ts, "top_idx": np.arange(len(topic_ts), dtype=np.int64)}
    )
    merged = pd.merge_asof(
        ref_df, top_df, on="t", direction=direction, tolerance=tolerance
    )

    hit = merged["top_idx"].notna().to_numpy()
    j = merged["top_idx"].to_numpy(dtype=np.float64)
    idx[hit] = j[hit].astype(np.int64)
    off[hit] = np.abs(ref_ts[hit] - topic_ts[idx[hit]])
    return idx, off


def align_streams(
    streams: Sequence[TopicStream],
    reference_topic: Optional[str] = None,
    target_fps: Optional[float] = None,
    strategy: str = "nearest",
    tolerance: float = 0.050,  # 50ms 容差窗，超窗丢帧
    required: Optional[Sequence[str]] = None,
    drop_incomplete: bool = True,
):
    """把多条异步 topic 流对齐成同步帧序列。

    参数：
        reference_topic: 以该 topic 的时间戳为参考轴（与 target_fps 二选一）
        target_fps:      按目标帧率在所有流时间交集上等距生成参考轴
        strategy:        nearest | latest_before
        tolerance:       最大允许偏差（秒），超过则该 topic 该帧记为缺失
        required:        必需 topic；缺失即丢帧（drop_incomplete=True 时）
        drop_incomplete: 是否丢弃不完备的帧（否则保留并在 frame.missing 标注）
    """
    if not streams:
        raise ValueError("streams 不能为空")
    by_name = {s.name: s for s in streams}
    if (reference_topic is None) == (target_fps is None):
        raise ValueError("reference_topic 与 target_fps 必须且只能提供一个")

    if reference_topic is not None:
        if reference_topic not in by_name:
            raise ValueError(f"参考 topic 不存在: {reference_topic}")
        ref_ts = by_name[reference_topic].timestamps
        ref_name = reference_topic
    else:
        t0 = max(s.timestamps[0] for s in streams if len(s.timestamps))
        t1 = min(s.timestamps[-1] for s in streams if len(s.timestamps))
        if t1 <= t0:  # pragma: no cover
            raise ValueError("各 topic 时间范围无交集，无法重采样")
        ref_ts = np.arange(t0, t1, 1.0 / target_fps)
        ref_name = f"resample@{target_fps}Hz"

    required = list(required) if required else [s.name for s in streams]
    frames: List[AlignedFrame] = []
    report = AlignmentReport(reference=ref_name, num_frames=0)
    for s in streams:
        report.stats[s.name] = TopicAlignStat(topic=s.name, nominal_hz=s.nominal_hz)

    # 每个 topic 一次性向量化匹配
    matches: Dict[str, tuple] = {}
    for s in streams:
        if s.name == reference_topic:
            idx = np.arange(len(ref_ts))
            off = np.zeros(len(ref_ts))
        else:
            idx, off = _match_indices(ref_ts, s.timestamps, strategy, tolerance)
        matches[s.name] = (idx, off)

    for i, t in enumerate(ref_ts):
        samples: Dict[str, Any] = {}
        offsets: Dict[str, float] = {}
        missing: List[str] = []
        for s in streams:
            idx, off = matches[s.name]
            j, o = idx[i], off[i]
            stat = report.stats[s.name]
            if j < 0:
                missing.append(s.name)
                stat.missing += 1
            else:
                samples[s.name] = s.values[j]
                offsets[s.name] = float(o * 1000.0)
                stat.matched += 1
                m = o * 1000.0
                stat.max_offset_ms = max(stat.max_offset_ms, m)
                stat.mean_offset_ms += (m - stat.mean_offset_ms) / stat.matched

        is_complete = not any(r in missing for r in required)
        if is_complete or not drop_incomplete:
            frames.append(
                AlignedFrame(
                    frame_index=len(frames),
                    timestamp=float(t),
                    samples=samples,
                    offsets_ms=offsets,
                    missing=missing,
                )
            )
        else:
            report.dropped_frames += 1

    report.num_frames = len(frames)
    return frames, report
