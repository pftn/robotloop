"""合成演示数据集 —— 不下载任何外部数据即可端到端跑通灌库 + 检索 demo。

模仿真实公开数据集的成分构成：多本体、多任务、成功/失败混合、
sim/teleop 混合。确定性随机（固定 seed），CI 可复现。

每个 step 带一路合成相机图像（/cam_head/image）：像素落盘为 jpg、
observation["images"] 存路径 —— 与真实 MCAP 链路的分层一致（对象存储
存像素，元数据存路径）。有图像特征，demo 导出的数据集才能过 ACT 等
策略的 validate_features（至少一路图像输入，observation.state 不算数）。
"""

from __future__ import annotations

import colorsys
import os
import tempfile
from typing import List

import numpy as np

from robotloop.schema.episode import DataSource, Episode, Step

# (task, language_instruction, embodiment, source, 成功率)
_DEMO_RECIPE = [
    ("pick_red_cube", "抓取红色方块并放到蓝色盘子里", "aloha", DataSource.SIM, 0.7),
    (
        "pick_red_cube",
        "pick up the red cube and place it on the plate",
        "aloha",
        DataSource.TELEOP,
        0.6,
    ),
    ("stack_blocks", "把绿色方块叠到红色方块上", "aloha", DataSource.SIM, 0.5),
    ("open_drawer", "拉开抽屉", "aloha", DataSource.SIM, 0.8),
    ("put_bowl_in_sink", "把碗放进水槽", "widowx", DataSource.TELEOP, 0.6),
    ("pick_carrot", "抓起胡萝卜放到盘子里", "widowx", DataSource.TELEOP, 0.5),
    ("pick_can", "pick the coke can", "google_robot", DataSource.TELEOP, 0.7),
    ("fold_cloth", "折叠毛巾", "agibot_g1", DataSource.TELEOP, 0.4),
    ("handover_cup", "把水杯递给人", "agibot_g1", DataSource.REAL, 0.5),
    ("press_button", "按下电梯按钮", "agibot_g1", DataSource.SIM, 0.9),
]

_ACTION_DIM = {"aloha": 14, "widowx": 7, "google_robot": 7, "agibot_g1": 14}
_ROBOT_TYPE = {
    "aloha": "aloha",
    "widowx": "widowx250s",
    "google_robot": "google",
    "agibot_g1": "genie1",
}

_DEMO_CAM = "/top"  # 对齐 gym_aloha 观测键：导出即 observation.images.top
_IMG_H, _IMG_W = (
    480,
    640,
)  # 对齐 gym_aloha 渲染分辨率（demo 数据可直接配 aloha env 评测）


def _demo_frames_dir(dataset_name: str, seed: int, episode_id: str) -> str:
    """合成图像落盘目录。按 seed 分目录，不同 seed 重生成互不污染；
    同 seed 重复调用幂等复用（测试/notebook 反复灌库零额外成本）。"""
    d = os.path.join(
        tempfile.gettempdir(),
        "robotloop_demo_frames",
        f"{dataset_name}-seed{seed}-{_IMG_W}x{_IMG_H}",
        episode_id,
    )
    os.makedirs(d, exist_ok=True)
    return d


def _render_demo_frame(
    img_rng: np.random.Generator, task: str, action_row: np.ndarray, path: str
) -> None:
    """渲染一帧合成图：底色随 task 取色（同任务同色系），亮块位置随
    action 前两维移动 —— "末端执行器"的视觉代理，让图像与状态/动作相关。
    幂等：文件已存在直接跳过。
    """
    if os.path.exists(path):
        return
    from PIL import Image

    r, g, b = colorsys.hsv_to_rgb((hash(task) % 100) / 100.0, 0.45, 0.55)
    img = np.empty((_IMG_H, _IMG_W, 3), dtype=np.uint8)
    img[:, :] = [int(r * 255), int(g * 255), int(b * 255)]
    ax = float(np.clip(action_row[0] if len(action_row) > 0 else 0.0, -1.5, 1.5))
    ay = float(np.clip(action_row[1] if len(action_row) > 1 else 0.0, -1.5, 1.5))
    cx = int((ax / 3.0 + 0.5) * (_IMG_W - 1))
    cy = int((ay / 3.0 + 0.5) * (_IMG_H - 1))
    s = 30
    img[max(0, cy - s) : cy + s, max(0, cx - s) : cx + s] = [255, 255, 255]
    img = (
        (img.astype(np.float32) + img_rng.normal(0, 6, img.shape))
        .clip(0, 255)
        .astype(np.uint8)
    )
    Image.fromarray(img).save(path, format="JPEG", quality=85)


def make_demo_episodes(
    n: int = 60,
    seed: int = 42,
    dataset_name: str = "robotloop_demo",
    min_len: int = 15,
    max_len: int = 40,
) -> List[Episode]:
    """生成 n 条合成 episode。轨迹动作 = 本体基线 + 任务模式 + 噪声（让去重/统计有意义）。"""
    rng = np.random.default_rng(seed)
    img_rng = np.random.default_rng(seed + 777)  # 图像噪声独立随机源，不扰动动作序列
    episodes: List[Episode] = []
    for i in range(n):
        task, instr, emb, source, p_succ = _DEMO_RECIPE[i % len(_DEMO_RECIPE)]
        success = bool(rng.random() < p_succ)
        length = int(rng.integers(min_len, max_len + 1))
        dim = _ACTION_DIM[emb]
        fps = 10.0 if source != DataSource.TELEOP else 30.0

        phase = (hash(task) % 100) / 100.0 * 2 * np.pi
        freq = 1.0 + (hash(task) % 5)
        t = np.linspace(0, 2 * np.pi, length)
        pattern = np.sin(freq * t + phase)
        base = rng.normal(0, 0.05, size=dim)
        actions = (
            pattern[:, None] * 0.5
            + base[None, :]
            + rng.normal(0, 0.02, size=(length, dim))
        )
        if not success:
            actions += rng.normal(0, 0.15, size=(length, dim))  # 失败轨迹叠加异常抖动

        frames_dir = _demo_frames_dir(dataset_name, seed, f"demo_{i:05d}")
        frame_paths = []
        for j in range(length):
            p = os.path.join(frames_dir, f"frame_{j:06d}.jpg")
            _render_demo_frame(img_rng, task, actions[j], p)
            frame_paths.append(p)

        steps = [
            Step(
                frame_index=j,
                timestamp=j / fps,
                observation={
                    "images": {_DEMO_CAM: frame_paths[j]},
                    "state": actions[j].tolist(),
                },
                action=actions[j].astype(np.float64).tolist(),
                is_terminal=(j == length - 1),
                language_instruction=instr,
            )
            for j in range(length)
        ]
        episodes.append(
            Episode(
                task=task,
                language_instruction=instr,
                embodiment_tag=emb,
                source=source,
                success=success,
                episode_id=f"demo_{i:05d}",
                duration=length / fps,
                dataset_name=dataset_name,
                episode_index=i,
                fps=fps,
                robot_type=_ROBOT_TYPE[emb],
                steps=steps,
            ).validate()
        )
    return episodes
