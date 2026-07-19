"""失败轨迹过滤 —— 质检流水线的第一道闸。

训练只该见到"干净的成功轨迹"。过滤分两类：
- 显式：success 标记（True 保留 / False 剔除 / None 可配置策略）
- 启发式：截断（帧数过少）、时长异常、缺语言指令、动作全零

每条被剔的轨迹都带 reason —— 质检结论必须可解释，不是黑箱。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class FilterResult:
    kept: List[Any] = field(default_factory=list)
    removed: List[Dict[str, Any]] = field(default_factory=list)  # {episode_id, reasons}

    @property
    def summary(self) -> Dict[str, Any]:
        reason_count: Dict[str, int] = {}
        for r in self.removed:
            for reason in r["reasons"]:
                reason_count[reason] = reason_count.get(reason, 0) + 1
        return {
            "total": len(self.kept) + len(self.removed),
            "kept": len(self.kept),
            "removed": len(self.removed),
            "reason_breakdown": reason_count,
        }


def filter_episodes(
    episodes: Sequence[Any],
    *,
    keep_unlabeled: bool = False,
    min_frames: int = 5,
    max_duration: float = 600.0,
    min_duration: float = 0.2,
    require_instruction: bool = True,
    zero_action_eps: float = 1e-8,
) -> FilterResult:
    """过滤失败/异常轨迹。

    参数：
        keep_unlabeled:  success=None（未标注）是否保留
        min_frames:      少于此帧数判为截断
        max/min_duration: 时长异常区间
        require_instruction: 缺语言指令则剔除
        zero_action_eps: 动作全零判定阈值（传感器失联的典型症状）
    """
    result = FilterResult()
    for ep in episodes:
        reasons: List[str] = []

        success = getattr(ep, "success", None)
        if success is False:
            reasons.append("success=false")
        elif success is None and not keep_unlabeled:
            reasons.append("success 未标注")

        n = getattr(ep, "num_frames", 0)
        if n < min_frames:
            reasons.append(f"帧数过少({n}<{min_frames})，疑似截断")

        dur = getattr(ep, "duration", 0.0)
        if dur < min_duration:
            reasons.append(f"时长过短({dur:.2f}s)")
        if dur > max_duration:
            reasons.append(f"时长过长({dur:.1f}s)，疑似挂机/未停止录制")

        if require_instruction and not (getattr(ep, "language_instruction", "") or "").strip():
            reasons.append("缺少 language_instruction")

        steps = getattr(ep, "steps", None)
        if steps:
            actions = np.asarray([s.action for s in steps], dtype=np.float64)
            if actions.size and float(np.abs(actions).max()) < zero_action_eps:
                reasons.append("动作序列全零，疑似控制/采集链路断开")

        if reasons:
            result.removed.append({"episode_id": getattr(ep, "episode_id", "?"), "reasons": reasons})
        else:
            result.kept.append(ep)
    return result
