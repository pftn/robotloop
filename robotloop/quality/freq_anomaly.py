"""topic 频率异常检测 —— 采集链路体检。

真实采集最常见的三类病：
- **掉帧/间隙**：某段 dt 远大于正常周期（USB 带宽抢占、磁盘抖动）
- **频率漂移**：整体频率偏离名义值（驱动配置错了，30Hz 相机跑成 22Hz）
- **抖动过大**：dt 方差异常（系统负载高，对齐后 residual 变大）

输出逐 topic 报告 + 异常时间区间列表 —— 能直接回答"这条 bag 哪一秒出的事"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FreqAnomaly:
    kind: str                 # gap | drift | jitter
    start_s: float
    end_s: float
    detail: str


@dataclass
class TopicFreqReport:
    topic: str
    num_samples: int
    nominal_hz: float           # 中位数周期对应的频率
    mean_hz: float
    std_jitter_ms: float        # dt 的标准差（毫秒）
    max_gap_ms: float
    anomalies: List[FreqAnomaly] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "num_samples": self.num_samples,
            "nominal_hz": round(self.nominal_hz, 2),
            "mean_hz": round(self.mean_hz, 2),
            "std_jitter_ms": round(self.std_jitter_ms, 3),
            "max_gap_ms": round(self.max_gap_ms, 2),
            "anomalies": [
                {"kind": a.kind, "start_s": round(a.start_s, 4), "end_s": round(a.end_s, 4), "detail": a.detail}
                for a in self.anomalies
            ],
        }


def detect_freq_anomalies(
    timestamps: np.ndarray,
    topic: str = "topic",
    expected_hz: Optional[float] = None,
    gap_factor: float = 3.0,        # dt > gap_factor × 中位周期 → 间隙
    drift_tol: float = 0.15,        # 实测频率偏离 expected_hz 超过此比例 → 漂移
    jitter_factor: float = 5.0,     # jitter > jitter_factor × 中位周期 → 抖动异常
) -> TopicFreqReport:
    """对单条 topic 的时间戳做频率体检。"""
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) < 3:
        return TopicFreqReport(topic=topic, num_samples=len(ts), nominal_hz=0.0,
                               mean_hz=0.0, std_jitter_ms=0.0, max_gap_ms=0.0)

    dt = np.diff(ts)
    med = float(np.median(dt))
    nominal_hz = 1.0 / med if med > 0 else 0.0
    mean_hz = float(len(ts) - 1) / float(ts[-1] - ts[0]) if ts[-1] > ts[0] else 0.0
    jitter_ms = float(np.std(dt - med) * 1000.0)
    report = TopicFreqReport(
        topic=topic,
        num_samples=len(ts),
        nominal_hz=nominal_hz,
        mean_hz=mean_hz,
        std_jitter_ms=jitter_ms,
        max_gap_ms=float(dt.max() * 1000.0),
    )

    # 1) 间隙：逐段报告
    gap_idx = np.where(dt > gap_factor * med)[0]
    for gi in gap_idx:
        report.anomalies.append(FreqAnomaly(
            kind="gap",
            start_s=float(ts[gi]),
            end_s=float(ts[gi + 1]),
            detail=f"间隔 {dt[gi]*1000:.1f}ms 超过 {gap_factor}× 中位周期 ({med*1000:.2f}ms)，疑似掉帧",
        ))

    # 2) 漂移：整体频率偏离标称值
    if expected_hz and nominal_hz > 0:
        dev = abs(mean_hz - expected_hz) / expected_hz
        if dev > drift_tol:
            report.anomalies.append(FreqAnomaly(
                kind="drift",
                start_s=float(ts[0]),
                end_s=float(ts[-1]),
                detail=f"实测 {mean_hz:.2f}Hz 偏离标称 {expected_hz:.2f}Hz {dev*100:.0f}%",
            ))

    # 3) 抖动
    if med > 0 and jitter_ms / 1000.0 > jitter_factor * med:
        report.anomalies.append(FreqAnomaly(
            kind="jitter",
            start_s=float(ts[0]),
            end_s=float(ts[-1]),
            detail=f"dt 抖动 {jitter_ms:.2f}ms，超过 {jitter_factor}× 中位周期",
        ))
    return report


def check_streams(streams: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, Any]]:
    """对 {topic: TopicStream} 批量体检（ingest 链路直接可用）。"""
    out = {}
    for name, s in streams.items():
        ts = s.timestamps if hasattr(s, "timestamps") else np.asarray(s)
        out[name] = detect_freq_anomalies(ts, topic=name, **kwargs).as_dict()
    return out
