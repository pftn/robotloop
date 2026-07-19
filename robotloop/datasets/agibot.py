"""AgiBot World（智元）公开数据集加载。

两条通路：
1. **LeRobot 镜像**（推荐）：社区已将 AgiBot World 转为 LeRobot 格式发布
   （HF 上有多个 v2.1/v3.0 镜像），直接走 ``load_lerobot_hub`` 通路，
   灌库时打上 ``embodiment_tag='agibot_g1'``。
2. **原生布局**：官方发布的 task 目录结构（meta json + 逐 episode 视频/参数）。
   原生解析器实现元信息侧解析；布局不符时给出明确报错而非静默出错。
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

from robotloop.schema.episode import DataSource, Episode


def load_agibot_lerobot_mirror(
    repo_id: str,
    local_dir: Optional[str] = None,
) -> List[Episode]:
    """加载 HF 上 LeRobot 格式的 AgiBot World 镜像。"""
    from robotloop.datasets.lerobot_hub import load_lerobot_hub

    episodes = load_lerobot_hub(
        repo_id, local_dir=local_dir, embodiment_tag="agibot_g1", source=DataSource.TELEOP
    )
    for ep in episodes:
        ep.dataset_name = f"agibot_world/{repo_id.split('/')[-1]}"
    return episodes


def load_agibot_native(root: str, max_episodes: Optional[int] = None) -> List[Episode]:
    """解析 AgiBot World 原生发布目录（元信息侧：task/指令/时长/成败）。

    帧级动作序列的逐字段对齐建议经 LeRobot 镜像通路入库（社区转换工具
    更成熟）；原生解析用于快速浏览数据集构成与登记元数据。
    """
    meta_files = sorted(
        glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
        + glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
    )
    if not meta_files:
        raise FileNotFoundError(
            f"{root} 下未找到 episode 元信息 json/jsonl。"
            "若不是 AgiBot World 原生布局，请改用 load_agibot_lerobot_mirror()。"
        )

    episodes: List[Episode] = []
    for i, mf in enumerate(meta_files):
        if max_episodes is not None and i >= max_episodes:
            break
        try:
            with open(mf, encoding="utf-8") as f:
                if mf.endswith(".jsonl"):
                    recs = [json.loads(l) for l in f if l.strip()]
                    rec = recs[0] if recs else {}
                else:
                    rec = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        task = rec.get("task_name") or rec.get("task") or os.path.basename(os.path.dirname(mf))
        instr = rec.get("language_instruction") or rec.get("instruction") or task
        episodes.append(
            Episode(
                task=str(task),
                language_instruction=str(instr),
                embodiment_tag="agibot_g1",
                source=DataSource.TELEOP,
                success=rec.get("success"),
                episode_id=str(rec.get("episode_id") or f"agibot_{i:06d}"),
                duration=float(rec.get("duration", 0.0)),
                dataset_name="agibot_world",
                episode_index=i,
                fps=float(rec.get("fps", 30.0)),
                steps=[],  # 原生帧级数据建议经 LeRobot 镜像通路入库
                metadata={"native_meta_path": mf},
            ).validate()
        )
    return episodes
