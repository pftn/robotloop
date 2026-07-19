"""GR00T fine-tune 衔接 —— 进阶路径，非主推。

训练主推 LeRobot 自带 ACT / Diffusion Policy（见
``robotloop.export.act_train``），GR00T/OpenVLA 微调"架构上支持，留作进阶"
—— 本模块即为该进阶路径的预留接口：

1. ``write_modality_json``：给导出的 v2.1 数据集补 GR00T 要求的
   ``meta/modality.json``（state/action 各维度段语义声明 ——
   "维度语义放元数据不写死 schema"的统一做法也复用它）
2. ``render_finetune_script``：GR00T N1.5 fine-tune 脚本生成（7B 级模型，
   租卡成本与调试时间会失控，演示闭环请勿使用）

参考：NVIDIA Isaac-GR00T 仓库 demo_data/cube_to_bowl_5/meta/modality.json
与 examples/LIBERO/finetune_libero_10.sh。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence


def write_modality_json(
    lerobot_dir: str,
    state_groups: Dict[str, Sequence[int]],
    action_groups: Dict[str, Sequence[int]],
    video_keys: Optional[List[str]] = None,
    annotation_key: str = "human.task_description",
) -> str:
    """在已导出的 LeRobot v2.1 数据集里写入 GR00T 要求的 meta/modality.json。

    参数：
        state_groups:   {"arm": (0, 6), "gripper": (6, 7)} —— observation.state 各段语义
        action_groups:  同上，对应 action 向量
        video_keys:     视频键名列表（如 ["observation.images.ego_view"]）
    """
    modality = {
        "state": {k: {"start": v[0], "end": v[1]} for k, v in state_groups.items()},
        "action": {k: {"start": v[0], "end": v[1]} for k, v in action_groups.items()},
        "video": {vk: {} for vk in (video_keys or [])},
        "annotation": {annotation_key: {}},
    }
    path = os.path.join(lerobot_dir, "meta", "modality.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(modality, f, indent=2, ensure_ascii=False)
    return path


_FINETUNE_TEMPLATE = """#!/bin/bash
# ============================================================
# RobotLoop × GR00T {groot_version} fine-tune（AutoDL 按小时租卡）
# 数据: {dataset_path}
# 预算参考: 单卡 4090(24G) 约 ¥2/h, {max_steps} steps 约 2-4h → 几十元跑通闭环
# ============================================================
set -e

# ---------- 0. 环境（AutoDL 选 PyTorch 2.x + CUDA 12.x 镜像）----------
if [ ! -d "Isaac-GR00T" ]; then
  git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
fi
cd Isaac-GR00T
pip install -q uv
uv sync --python 3.10
uv pip install -e .

# ---------- 1. 数据 ----------
# 方式 A（自定义数据）: RobotLoop 导出的 v2.1 数据集已 scp 到 {dataset_path}
#   robotloop export --store ./lake --filters '{{"embodiment_tag":"aloha"}}' \\
#       --out {dataset_path} --version v2.1 --success-only --with-modality
# 方式 B（官方 LIBERO）:
#   hf download --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \\
#       --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/
#   cp examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/

# ---------- 2. fine-tune ----------
CUDA_VISIBLE_DEVICES=0 uv run python scripts/gr00t_finetune.py \\
  --dataset-path {dataset_path} \\
  --num-gpus 1 \\
  --output-dir {output_dir} \\
  --max-steps {max_steps} \\
  --data-config {data_config} \\
  --video-backend torchvision_av
# 显存不足（<25G）时追加: --no-tune_diffusion_model

# ---------- 3. 评测 + 录屏（LIBERO 仿真） ----------
# uv run bash examples/LIBERO/eval_libero.sh --model-path {output_dir}
# 录屏文件位于 eval 输出目录, 取一条放回 RobotLoop README 即完成闭环证据链
echo "✔ fine-tune 完成, checkpoint: {output_dir}"
"""


def render_finetune_script(
    dataset_path: str,
    output_dir: str = "./groot-checkpoints",
    groot_version: str = "N1.5",
    data_config: str = "libero",
    max_steps: int = 10000,
    out_path: Optional[str] = None,
) -> str:
    """生成 AutoDL 上一键执行的 fine-tune 脚本。"""
    script = _FINETUNE_TEMPLATE.format(
        groot_version=groot_version,
        dataset_path=dataset_path,
        output_dir=output_dir,
        data_config=data_config,
        max_steps=max_steps,
    )
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(script)
        os.chmod(out_path, 0o755)
    return script
