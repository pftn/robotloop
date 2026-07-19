"""Episode 领域模型 —— RobotLoop 数据闭环的统一数据结构。

一条 Episode 是一次完整任务执行的轨迹：

    Episode ── 1:N ──► Step（按帧索引 frame_index 有序）

- Episode 承载"这条轨迹是谁、在什么本体上、做什么任务、成功没有、数据从哪来"，
  是检索、过滤、统计的最小业务单元；
- Step 承载逐帧的观测（observation）与动作（action），是训练数据的最小单元。

与外部格式的对应关系见 ``robotloop.schema.mapping``（RLDS ↔ LeRobot ↔ RobotLoop）。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DataSource(str, Enum):
    """数据采集来源。取值与 Iceberg episodes.source 列的约束一致。"""

    TELEOP = "teleop"  # 真机遥操作
    SIM = "sim"  # 仿真（LIBERO / SimplerEnv / Isaac Lab ...）
    REAL = "real"  # 真机自主执行回放/在线采集


class Embodiment(str, Enum):
    """常见机器人本体标签，与 GR00T embodiment tag / Open X 命名习惯对齐。

    不在枚举内的本体可以直接用字符串，Iceberg 侧不做枚举约束。
    """

    FRANKA = "franka"  # Franka Emika Panda（LIBERO / 大量 Open X 子集）
    WIDOWX = "widowx"  # BridgeData V2
    GOOGLE = "google_robot"  # RT-1 / RT-2
    XARM = "xarm"
    UR5 = "ur5"
    ALOHA = "aloha"
    SO100 = "so100"
    AGIBOT_G1 = "agibot_g1"  # AgiBot World（智元）
    UNITREE_G1 = "unitree_g1"
    LIBERO_PANDA = "libero_panda"  # GR00T 预注册 embodiment tag


@dataclass
class Step:
    """一帧同步后的机器人数据（训练最小单元）。

    所有多模态信号在同一逻辑时刻对齐后写入一个 Step ——
    对齐工作由 ``robotloop.ingest.align`` 在入库前完成。
    """

    frame_index: (
        int  # 帧序号（episode 内 0..N-1，即 RLDS step 序号 / LeRobot frame_index）
    )
    timestamp: float  # 对齐后的统一时间戳（秒，Unix epoch 或 episode 相对时间）
    observation: Dict[str, Any]  # {"images": {cam: path|dict}, "state": [...], ...}
    action: List[float]  # 动作向量（关节/EEF 增量等）
    reward: Optional[float] = None  # RLDS 有 reward；LeRobot 无（可选）
    is_terminal: bool = (
        False  # 是否该 episode 最后一帧（对齐 LeRobot next.done / RLDS is_terminal）
    )
    language_instruction: str = ""  # 逐帧语言指令（通常与 episode 级一致）

    def validate(self) -> "Step":
        if self.frame_index < 0:
            raise ValueError(f"frame_index 必须非负, got {self.frame_index}")
        if not self.action:
            raise ValueError("action 不能为空")
        return self


@dataclass
class Episode:
    """一条完整任务轨迹（检索/过滤/统计最小单元）。

    字段与 Iceberg ``robotloop.episodes`` 表一一对应（见 schema/iceberg.py）。
    """

    task: str  # 任务名（结构化，如 "pick_red_cube"）
    language_instruction: str  # 自然语言指令（语义检索的文本对象）
    embodiment_tag: str  # 本体标签（aloha / widowx / agibot_g1 ...）
    source: DataSource = DataSource.TELEOP
    success: Optional[bool] = None  # None = 未知/未标注
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    duration: float = 0.0  # 秒
    dataset_name: str = ""  # 来源数据集（openx/bridge_v2, agibot_world, 自采 ...）
    episode_index: int = 0  # 在来源数据集中的序号（对应 LeRobot episode_index）
    fps: float = 0.0  # 对齐后的目标帧率
    robot_type: str = ""  # 细粒度机型（"panda" / "widowx250s" ...）
    steps: List[Step] = field(default_factory=list)
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # 扩展信息（相机配置、标定、采集员 ...）
    parquet_path: str = (
        ""  # 帧数据 Parquet 路径（入库时由 frame_store 填充；Iceberg 只存这个指针）
    )
    created_at: float = field(default_factory=time.time)

    # 向量在入库时由 retrieval.encoder 计算后写入特征表，
    # 不放在领域模型里，避免领域模型依赖编码器。

    def validate(self) -> "Episode":
        if not self.task:
            raise ValueError("task 不能为空")
        if not self.embodiment_tag:
            raise ValueError("embodiment_tag 不能为空")
        if not isinstance(self.source, DataSource):
            self.source = DataSource(self.source)
        for i, s in enumerate(self.steps):
            if s.frame_index != i:
                raise ValueError(
                    f"steps 必须按 frame_index 连续排列: 第 {i} 个 step 的 frame_index={s.frame_index}"
                )
        if self.steps and self.duration <= 0 and self.fps > 0:
            self.duration = len(self.steps) / self.fps
        return self

    # ---------- 便捷属性 ----------
    @property
    def num_frames(self) -> int:
        return len(self.steps)

    @property
    def action_dim(self) -> int:
        return len(self.steps[0].action) if self.steps else 0

    def step_dicts(self) -> List[Dict[str, Any]]:
        """展开为帧级行（每行一帧），写入 MinIO 的 frames/{episode_id}.parquet。

        字段命名对齐 LeRobot v2.1/v3.0 与 RLDS：is_first / is_last / is_terminal
        三者齐全（Episode 模型是两者的超集）。
        """
        n = len(self.steps)
        rows = []
        for s in self.steps:
            rows.append(
                {
                    "episode_id": self.episode_id,
                    "frame_index": s.frame_index,
                    "timestamp": s.timestamp,
                    "action": [float(a) for a in s.action],
                    "state": _as_float_list(s.observation.get("state")),
                    "image_paths": {
                        k: (v if isinstance(v, str) else v.get("path", ""))
                        for k, v in s.observation.get("images", {}).items()
                    },
                    "reward": s.reward,
                    "is_terminal": s.is_terminal,
                    "is_first": s.frame_index == 0,
                    "is_last": s.frame_index == n - 1,
                    "language_instruction": s.language_instruction
                    or self.language_instruction,
                }
            )
        return rows

    def meta_dict(self) -> Dict[str, Any]:
        """episodes 表的一行。"""
        return {
            "episode_id": self.episode_id,
            "dataset_name": self.dataset_name,
            "episode_index": self.episode_index,
            "task": self.task,
            "language_instruction": self.language_instruction,
            "embodiment_tag": str(self.embodiment_tag),
            "source": self.source.value,
            "success": self.success,
            "duration": float(self.duration),
            "fps": float(self.fps),
            "num_frames": self.num_frames,
            "robot_type": self.robot_type,
            "parquet_path": self.parquet_path,
            "created_at": self.created_at,
        }


def check_consistent_dims(episodes: List["Episode"]) -> None:
    """校验一组 Episode 的 action/state 维度一致。

    一个 LeRobot 数据集只有一套 features 定义，action/state 维度必须全数据集
    统一（fixed-size list 列）。混本体导出（如 franka 7 维 + agibot 14 维）
    必然写不出，这里提前拦截并给出可操作提示，而不是让 pyarrow 在写出
    中途抛 ArrowInvalid。
    """
    dims: Dict[tuple, List["Episode"]] = {}
    for ep in episodes:
        if not ep.steps:
            continue
        a = len(ep.steps[0].action)
        s = len(_as_float_list(ep.steps[0].observation.get("state")))
        dims.setdefault((a, s), []).append(ep)
    if len(dims) <= 1:
        return
    detail = "; ".join(
        f"action={a}/state={s}: {len(eps)} 条（{', '.join(sorted({e.embodiment_tag for e in eps}))}）"
        for (a, s), eps in sorted(dims.items())
    )
    raise ValueError(
        f"LeRobot 数据集要求 action/state 维度统一，当前混入 {len(dims)} 种维度：{detail}。"
        f"请按本体过滤后导出，例如 filters={{'embodiment_tag': 'aloha'}}"
    )


def check_consistent_fps(episodes: List["Episode"], rtol: float = 0.05) -> None:
    """校验一组 Episode 的 fps 一致。

    一个 LeRobot 数据集的 info.json 只有一个 fps 值，而各 episode 的
    timestamp 是采集时的真实值：混帧率导出（如自采 30Hz MCAP + 灌库
    10Hz Open X）会让 lerobot 按单一 fps 校验所有 episode 的时间戳，
    低帧率段必然超容差报 timestamps violate the tolerance。这里提前
    拦截并给出可操作提示。rtol 容差防 29.97 vs 30 的浮点抖动。
    """
    groups: List[List["Episode"]] = []
    centroids: List[float] = []
    for ep in episodes:
        if ep.fps <= 0:
            continue
        for i, c in enumerate(centroids):
            if abs(ep.fps - c) <= c * rtol:
                groups[i].append(ep)
                break
        else:
            centroids.append(ep.fps)
            groups.append([ep])
    if len(groups) <= 1:
        return
    detail = "; ".join(
        f"fps={c:g}: {len(eps)} 条（{', '.join(sorted({e.embodiment_tag for e in eps}))}）"
        for c, eps in zip(centroids, groups)
    )
    raise ValueError(
        f"LeRobot 数据集要求全库统一 fps（info.json 单值），当前混入 {len(groups)} 种：{detail}。"
        "请按来源/本体过滤后导出，使数据集 fps 单一，例如 "
        "filters={'source': 'openx'} 或 filters={'embodiment_tag': 'aloha', 'source': 'sim'}"
    )


def _as_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return [float(v) for v in x.ravel().tolist()]
    except ImportError:  # pragma: no cover
        pass
    return [float(v) for v in x]
