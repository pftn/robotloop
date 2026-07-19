"""RLDS ↔ RobotLoop Episode 转换。

RLDS（Reinforcement Learning Datasets）是 Open X-Embodiment、DROID 等
的发布格式，物理上是 TFDS/TFRecord，逻辑结构是::

    {
      "episode_metadata": {...},
      "steps": [
        {"observation": {...}, "action": [...], "reward": r,
         "is_first": b, "is_last": b, "is_terminal": b,
         "language_instruction": "...", "discount": 1.0},
        ...
      ]
    }

方向约定（范围决策：RLDS 只做单向读取）：

- **唯一受支持的生产路径**是 ``load_tfds_rlds`` / ``rlds_episode_to_episode``：
  Open X / DROID 等 RLDS 数据集 → Episode（单向读取）。TF/TFDS 依赖重，
  在单独 Docker 里跑（见 docs/format_convert.md），不污染主环境。
- ``episode_to_rlds_dict`` 仅为单元测试保留的对称参考实现（Episode → 规范
  dict，不落 TFRecord），**不做"LeRobot → RLDS"反向转换，不维护、勿用于
  生产**（没需求且工作量翻倍）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from robotloop.schema.episode import DataSource, Episode, Step


def _to_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.ravel().tolist()]
    return [float(v) for v in x]


def rlds_episode_to_episode(
    rlds_ep: Dict[str, Any],
    *,
    task: str = "",
    embodiment_tag: str = "unknown",
    source: DataSource = DataSource.TELEOP,
    dataset_name: str = "",
    episode_index: int = 0,
    fps: float = 0.0,
    state_keys: Optional[List[str]] = None,
) -> Episode:
    """RLDS episode dict → Episode。

    参数：
        state_keys: observation 里哪些键拼成 state（默认取 "state" 键；
                    Open X 各数据集键名不一，如 "proprio" / "cartesian_position"）
    """
    steps_raw = list(rlds_ep.get("steps", []))
    if not steps_raw:
        raise ValueError("RLDS episode 没有 steps")

    rlds_meta = rlds_ep.get("episode_metadata", {}) or {}
    instr = ""
    for s in steps_raw:
        li = s.get("language_instruction")
        if isinstance(li, bytes):
            li = li.decode("utf-8", errors="ignore")
        if li:
            instr = str(li)
            break
    task_name = task or instr or f"episode_{episode_index}"

    steps: List[Step] = []
    n = len(steps_raw)
    for i, s in enumerate(steps_raw):
        obs = s.get("observation", {}) or {}
        images = {k: v for k, v in obs.items() if "image" in k.lower() or "rgb" in k.lower()}
        if state_keys:
            state = (
                np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in state_keys if k in obs]).tolist()
                if any(k in obs for k in state_keys)
                else []
            )
        else:
            state = _to_list(obs.get("state"))
        steps.append(
            Step(
                frame_index=i,
                timestamp=float(i) / fps if fps else float(i),
                observation={"images": images, "state": state},
                action=_to_list(s.get("action")),
                reward=float(s["reward"]) if s.get("reward") is not None else None,
                is_terminal=bool(s.get("is_terminal", i == n - 1)),
                language_instruction=instr,
            )
        )

    success = rlds_meta.get("success")
    if success is None and steps_raw:
        # Open X 部分数据集用末帧 reward>0 近似成功标记
        last_r = steps_raw[-1].get("reward")
        if last_r is not None:
            success = bool(float(last_r) > 0)

    return Episode(
        task=task_name,
        language_instruction=instr or task_name,
        embodiment_tag=embodiment_tag,
        source=source,
        success=success,
        episode_id=str(rlds_meta.get("episode_id") or f"rlds_{episode_index:06d}"),
        duration=(n / fps) if fps else 0.0,
        dataset_name=dataset_name,
        episode_index=episode_index,
        fps=fps,
        steps=steps,
        metadata={"rlds": {k: v for k, v in rlds_meta.items() if isinstance(v, (str, int, float, bool))}},
    ).validate()


def episode_to_rlds_dict(ep: Episode) -> Dict[str, Any]:
    """Episode → RLDS 规范 dict（可交给 rlds/TFDS builder 写 TFRecord）。"""
    n = ep.num_frames
    steps = []
    for i, s in enumerate(ep.steps):
        obs: Dict[str, Any] = {}
        if s.observation.get("state"):
            obs["state"] = np.asarray(s.observation["state"], dtype=np.float32)
        for k, v in s.observation.get("images", {}).items():
            obs[k] = v  # ndarray 或路径；写 TFRecord 前由调用方编码为 bytes
        steps.append(
            {
                "observation": obs,
                "action": np.asarray(s.action, dtype=np.float32),
                "reward": float(s.reward) if s.reward is not None else (1.0 if (ep.success and i == n - 1) else 0.0),
                "discount": 1.0,
                "is_first": i == 0,
                "is_last": i == n - 1,
                "is_terminal": bool(s.is_terminal),
                "language_instruction": s.language_instruction or ep.language_instruction,
            }
        )
    return {
        "episode_metadata": {
            "episode_id": ep.episode_id,
            "success": ep.success,
            "embodiment_tag": ep.embodiment_tag,
            "source": ep.source.value,
            "dataset_name": ep.dataset_name,
        },
        "steps": steps,
    }


def load_tfds_rlds(
    tfds_name: str,
    split: str = "train",
    max_episodes: Optional[int] = None,
    **kwargs,
) -> List[Episode]:
    """从 TFDS 直接加载 Open X 子集为 Episode 列表（需要 tensorflow_datasets）。

    例：``load_tfds_rlds("bridge_v2/0.1.0", split="train[:100]", embodiment_tag="widowx")``
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise ImportError(
            "加载 Open X/RLDS 数据集需要 tensorflow_datasets: pip install tensorflow_datasets"
        ) from e

    ds = tfds.load(tfds_name, split=split, shuffle_files=False)
    episodes: List[Episode] = []
    for i, rlds_ep in enumerate(ds):
        if max_episodes is not None and i >= max_episodes:
            break
        episodes.append(
            rlds_episode_to_episode(
                rlds_ep,
                episode_index=i,
                dataset_name=tfds_name.split("/")[0],
                **kwargs,
            )
        )
    return episodes
