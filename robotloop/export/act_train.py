"""训练闭环主推路径：LeRobot 自带 ACT / Diffusion Policy。

AutoDL 租 4090，ACT 在 PushT / aloha sim 上几小时出结果，全程一百元以内。
不碰 OpenVLA/GR00T 微调 —— 7B 级模型租卡成本和调试时间会失控
（架构上支持 GR00T/OpenVLA 微调接入，留作进阶，见 export/gr00t.py）。

本模块生成 AutoDL 上一键跑的训练 + 仿真评测 + 录屏脚本。
"""

from __future__ import annotations

import os
from typing import Optional

_ACT_TEMPLATE = """#!/bin/bash
# ============================================================
# RobotLoop 训练闭环：LeRobot {policy} × {env}（AutoDL 按小时租卡）
# 数据: {dataset}
# 预算参考: 单卡 4090(24G) 约 ¥2/h，ACT+PushT {steps} steps 约 2-3h → 全程一百元以内
# 训练用 ACT/Diffusion Policy，不碰 GR00T/OpenVLA 微调（留作进阶）
# ============================================================
set -e

# ---------- 0. 环境（AutoDL 选 PyTorch 2.x + CUDA 12.x 镜像）----------
# lerobot pin 到 v0.3.3：main 分支（0.8.x）训练入口已改为 lerobot-train
# entry point，python -m lerobot.scripts.train 模块路径不存在
LEROBOT_VERSION=${{LEROBOT_VERSION:-v0.3.3}}
if [ ! -d "lerobot/.git" ]; then
  git clone --branch ${{LEROBOT_VERSION}} --depth 1 https://github.com/huggingface/lerobot.git
fi
cd lerobot
# 已有 clone 不在 pin 版本时切过去（比如之前 clone 过 main）
if [ "$(git describe --tags 2>/dev/null)" != "${{LEROBOT_VERSION}}" ]; then
  git fetch --depth 1 origin tag ${{LEROBOT_VERSION}} && git checkout ${{LEROBOT_VERSION}}
fi
pip install -e .
# PushT 仿真环境依赖
pip install -e ".[pusht]" || pip install gym-pusht

# torchcodec（lerobot 视频解码后端）两个前置：
# 1. 系统 FFmpeg 动态库（支持 4-7 任一；ubuntu 22.04 源装 4.4 即满足）
apt-get update && apt-get install -y ffmpeg
# 2. torchcodec 与 torch 版本严格对应：torch 2.7.x -> torchcodec 0.4.x
#    （pip 默认装最新会配错；其他 torch 版本查官方兼容表）
pip install "torchcodec==0.4.*"
# wandb：训练曲线 + eval 录屏自动上传的跟踪面板
pip install wandb

# mujoco/dm_control 仿真渲染（aloha 等 env 的 eval 录屏）：
# 无显示器服务器必须把 GL 后端切到离屏，否则 gladLoadGL / GLFW 报错。
# AutoDL 有 N 卡 -> egl（GPU 渲染，快）；不行改 MUJOCO_GL=osmesa（CPU 渲染，慢但通用）
apt-get install -y libegl1 libgles2 libglfw3 libglew2.2 libosmesa6 libgl1-mesa-glx
export MUJOCO_GL=${{MUJOCO_GL:-egl}}
export PYOPENGL_PLATFORM=${{PYOPENGL_PLATFORM:-egl}}
# 若报 GLIBCXX 版本错（conda 的 libstdc++ 与系统库冲突），追加两行再跑：
#   unset LD_LIBRARY_PATH
#   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# ---------- 1. 数据 ----------
# 方式 A（RobotLoop 导出，推荐 —— 走通自家闭环）:
#   本地执行: robotloop export --store ./lake \\
#       --filters '{{"embodiment_tag":"aloha","success":true}}' \\
#       --out ./ft_data --version v2.1 --success-only
#   再 scp 到 AutoDL: scp -r ./ft_data root@<autodl-host>:/root/lerobot/data/robotloop_ft
# 方式 B（官方示例，先验证环境）:
#   DATASET=lerobot/pusht （HF_ENDPOINT=https://hf-mirror.com 走镜像）
export HF_ENDPOINT=https://hf-mirror.com
DATASET={dataset}

# ---------- 2. 训练 + 仿真评测 + 录屏（一条命令全包） ----------
python -m lerobot.scripts.train \\
  --policy.type={policy} \\
  --dataset.repo_id=${{DATASET}} \\
{env_line}  --output_dir=outputs/{policy}_{env_name} \\
  --job_name=robotloop_{policy}_{env_name} \\
  --steps={steps} \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --eval_freq={eval_freq} \\
  --save_freq={save_freq} \\
  --eval.batch_size=10 \\
  --eval.n_episodes={n_eval_episodes} \\
  --wandb.enable=${{WANDB_ENABLE:-true}} \\
  --wandb.project=${{WANDB_PROJECT:-robotloop-{policy}-{env}}} \\
  --wandb.mode=${{WANDB_MODE:-offline}}

# wandb 参数说明：
#   enable  开关；false 时下面参数全部无效
#   project wandb 项目名（曲线与录屏归到这个面板）
#   mode    offline 先落本地 ./wandb（AutoDL 直连 wandb.ai 常超时），
#           跑完 wandb sync wandb/offline-run-* 再上传；网络可用改 online
# 录屏无需参数：--env.type 非空时，train 在每个 eval_freq 节点自动仿真
# 评测并录 mp4 到 outputs/{policy}_{env_name}/videos/（online 模式同步到 wandb）。
# 未传 --env.type（自家 franka 数据：lerobot 内置仿真环境只有
# pusht/aloha/xarm/gym_manipulator/hil，无 franka，维度不匹配会在 eval 报
# size mismatch）则只训练不做仿真评测，checkpoint 正常产出。
# 取一条 eval 录屏放回 RobotLoop README 顶部，
# 即完成「入库 → 检索 → 导出 → 训练 → 评测 → 视频」全链路证据。

echo "✔ 训练完成, checkpoint: outputs/{policy}_{env_name}"
"""


def render_act_train_script(
    dataset: str = "lerobot/pusht",
    policy: str = "act",
    env: str = "pusht",
    steps: int = 20000,
    eval_freq: int = 5000,
    save_freq: int = 10000,
    n_eval_episodes: int = 10,
    out_path: Optional[str] = None,
) -> str:
    """生成 AutoDL 一键训练脚本（ACT/Diffusion Policy × PushT/aloha sim）。

    参数：
        dataset: LeRobot 数据集 repo_id 或本地路径（RobotLoop 导出产物）
        policy:  act | diffusion（lerobot 自带策略）
        env:     pusht | aloha（仿真评测环境）；传 "" 不做仿真评测——
                 自家 franka 数据必须传 ""（lerobot 无 franka 仿真环境，
                 维度不匹配会在 eval 报 size mismatch）
        steps:   训练步数（4090 上 ACT+PushT 2 万步约 2-3h）
    """
    if policy not in ("act", "diffusion"):
        raise ValueError(f"policy 仅支持 lerobot 自带的 act/diffusion，got {policy}")
    # env 为空 = 不做仿真评测（自家 franka 数据：lerobot 无匹配仿真环境，
    # 强配 env 会因观测/动作维度不一致在 eval 报 size mismatch）
    env_name = env or "dataset"
    env_line = f"  --env.type={env} \\\n" if env else ""
    script = _ACT_TEMPLATE.format(
        policy=policy,
        env=env,
        env_name=env_name,
        env_line=env_line,
        dataset=dataset,
        steps=steps,
        eval_freq=eval_freq,
        save_freq=save_freq,
        n_eval_episodes=n_eval_episodes,
    )
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(script)
        os.chmod(out_path, 0o755)
    return script
