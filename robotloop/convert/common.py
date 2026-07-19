"""格式互转共享工具：统计量计算、任务表、JSONL 读写。

转换架构是 hub-and-spoke：所有格式 ↔ Episode 领域模型 ↔ 所有格式，
不写 N×N 的两两转换器。LeRobot v2.1 ↔ v3.0 也走 Episode 中转，
保证语义无损（成功标记、来源、本体标签都不丢）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

import numpy as np

from robotloop.schema.episode import Episode


def build_task_table(episodes: Sequence[Episode]) -> Dict[str, int]:
    """task 文本 → task_index（按首次出现顺序，确定性）。"""
    table: Dict[str, int] = {}
    for ep in episodes:
        if ep.task not in table:
            table[ep.task] = len(table)
    return table


def feature_stats(arrays: List[np.ndarray]) -> Dict[str, list]:
    """一组等宽向量的逐维统计（LeRobot episodes_stats 的格式）。"""
    if not arrays:
        return {"mean": [], "std": [], "min": [], "max": [], "count": [0]}
    m = np.stack([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    return {
        "mean": m.mean(axis=0).tolist(),
        "std": m.std(axis=0).tolist(),
        "min": m.min(axis=0).tolist(),
        "max": m.max(axis=0).tolist(),
        "count": [int(m.shape[0])],
    }


def merge_stats(per_episode: List[Dict[str, list]]) -> Dict[str, list]:
    """把多条 episode 的统计合并为全局统计（v3.0 的 meta/stats.json）。"""
    eps = [s for s in per_episode if s.get("count", [0])[0] > 0]
    if not eps:
        return {"mean": [], "std": [], "min": [], "max": [], "count": [0]}
    counts = np.array([s["count"][0] for s in eps], dtype=np.float64)
    means = np.array([s["mean"] for s in eps])
    stds = np.array([s["std"] for s in eps])
    total = counts.sum()
    gmean = (means * counts[:, None]).sum(axis=0) / total
    # 合并方差：E[x²] = std² + mean²
    gvar = ((stds**2 + means**2) * counts[:, None]).sum(axis=0) / total - gmean**2
    return {
        "mean": gmean.tolist(),
        "std": np.sqrt(np.maximum(gvar, 0)).tolist(),
        "min": np.array([s["min"] for s in eps]).min(axis=0).tolist(),
        "max": np.array([s["max"] for s in eps]).max(axis=0).tolist(),
        "count": [int(total)],
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def episode_action_array(ep: Episode) -> np.ndarray:
    return np.asarray([s.action for s in ep.steps], dtype=np.float32)


def episode_state_array(ep: Episode) -> np.ndarray:
    return np.asarray(
        [s.observation.get("state", []) or [] for s in ep.steps], dtype=np.float32
    )
