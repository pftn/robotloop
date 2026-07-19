#!/usr/bin/env python3
"""生成示例 MCAP 数据包 —— 无真机时演示真实数据链路（upload_mcap.py 的上游）。

写一个合成 .mcap：
- /top          30Hz  sensor_msgs/msg/CompressedImage（jpeg 帧，480x640）
- /joint_states 500Hz sensor_msgs/msg/JointState（14 维 ALOHA 双臂平滑轨迹，
                模拟"接近 → 抓取 → 收回"的 pick 动作轮廓）

14 维对齐 lerobot aloha 仿真环境的观测/动作空间，相机 topic 名 /top 与
480x640 分辨率也对齐 gym_aloha 的观测（observation.images.top, 480x640）：
自采数据导出训练后可直接用 --env.type=aloha 做仿真评测，eval 阶段不会因
相机键名/分辨率错配而报错。

生成的包可直接被 RobotLoop 解析端（robotloop.ingest.mcap / pipeline.bag_to_episode）
读回 —— 生成与解析两端自洽。

用法：
    python make_sample_bag.py --out-dir ./sample_bags [--duration 8] [--joints 14]
"""

from __future__ import annotations

import argparse
import math
import os

# 相机帧：480x640 合成 JPEG —— 亮块横向位置随 pick 轨迹的 reach 进度移动
# （视觉与动作相关；分辨率对齐 gym_aloha 的 observation.images.top，
# 导出训练后可直接做 aloha 仿真评测，无需导出侧 resize）。
_IMG_W, _IMG_H = 640, 480


def _render_frame(reach: float) -> bytes:
    """渲染一帧：灰色工作台背景 + 随 reach 移动的亮色方块（末端执行器视觉代理）。"""
    import io

    from PIL import Image

    img = Image.new("RGB", (_IMG_W, _IMG_H), (90, 86, 80))
    cx = int(40 + reach * (_IMG_W - 120))
    cy = _IMG_H // 2
    for dy in range(-30, 31):
        for dx in range(-30, 31):
            img.putpixel((cx + dx, cy + dy), (230, 200, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


_COMPRESSED_IMAGE_DEF = """std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""

_JOINT_STATE_DEF = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def _pick_trajectory(t: float, duration: float, joints: int) -> list:
    """平滑 pick 轨迹：慢速接近 → 下压抓取 → 抬升收回。

    joints=14 按 ALOHA 双臂布局：前 7 维右臂主动作，后 7 维左臂
    镜像辅助（半幅度 + 相位偏移）。
    """
    phase = t / duration  # 0..1
    if phase < 0.4:  # 接近段：缓动
        p = phase / 0.4
        reach = p * p * (3 - 2 * p)
        lift = 0.0
    elif phase < 0.6:  # 下压抓取段
        reach = 1.0
        lift = -0.15 * math.sin((phase - 0.4) / 0.2 * math.pi)
    else:  # 收回段
        p = (phase - 0.6) / 0.4
        reach = 1.0 - p * p * (3 - 2 * p)
        lift = 0.2 * math.sin(p * math.pi)

    def _arm(i: int, scale: float, phase_shift: float) -> float:
        return (
            0.8 * reach * math.cos(i + phase_shift) * scale
            + 0.05 * math.sin(2 * math.pi * t + i)
            + lift * (i % 3 == 1) * scale
        )

    if joints == 14:
        return [_arm(i, 1.0, 0.0) for i in range(7)] + [
            _arm(i, 0.5, math.pi / 4) for i in range(7)
        ]
    return [_arm(i, 1.0, 0.0) for i in range(joints)]


def _joint_names(joints: int) -> list:
    if joints == 14:  # ALOHA 双臂：右臂 7 + 左臂 7
        return [f"right_arm_joint{i + 1}" for i in range(7)] + [
            f"left_arm_joint{i + 1}" for i in range(7)
        ]
    return [f"joint{i + 1}" for i in range(joints)]


def make_sample_bag(
    out_path: str,
    duration: float = 8.0,
    camera_hz: float = 30.0,
    joint_hz: float = 500.0,
    joints: int = 14,
    t0_ns: int = 1_700_000_000_000_000_000,
) -> dict:
    """写一个合成 MCAP，返回统计信息。"""
    try:
        from mcap_ros2.writer import Writer as Ros2Writer
    except ImportError as e:
        raise ImportError(
            "需要 mcap-ros2-support: pip install mcap mcap-ros2-support"
        ) from e

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n_cam, n_joint = 0, 0

    with open(out_path, "wb") as f:
        writer = Ros2Writer(f)
        img_schema = writer.register_msgdef(
            "sensor_msgs/msg/CompressedImage", _COMPRESSED_IMAGE_DEF
        )
        joint_schema = writer.register_msgdef(
            "sensor_msgs/msg/JointState", _JOINT_STATE_DEF
        )

        # 关节流 500Hz：全时段
        n_joint_samples = int(duration * joint_hz)
        for k in range(n_joint_samples):
            t = k / joint_hz
            ts = t0_ns + int(t * 1e9)
            writer.write_message(
                topic="/joint_states",
                schema=joint_schema,
                message={
                    "header": {
                        "stamp": {"sec": ts // 10**9, "nanosec": ts % 10**9},
                        "frame_id": "base_link",
                    },
                    "name": _joint_names(joints),
                    "position": _pick_trajectory(t, duration, joints),
                    "velocity": [],
                    "effort": [],
                },
                log_time=ts,
                publish_time=ts,
            )
            n_joint += 1

        # 相机流 30Hz：同一时轴；帧内容与轨迹同步（亮块 = reach 进度）
        n_cam_samples = int(duration * camera_hz)
        for k in range(n_cam_samples):
            t = k / camera_hz
            ts = t0_ns + int(t * 1e9)
            reach = _pick_trajectory(t, duration, 14)[0]  # 右臂 joint1 作为进度代理
            writer.write_message(
                topic="/top",
                schema=img_schema,
                message={
                    "header": {
                        "stamp": {"sec": ts // 10**9, "nanosec": ts % 10**9},
                        "frame_id": "camera_link",
                    },
                    "format": "jpeg",
                    "data": _render_frame(min(max((reach + 0.8) / 1.6, 0.0), 1.0)),
                },
                log_time=ts,
                publish_time=ts,
            )
            n_cam += 1

        writer.finish()

    stats = {
        "path": out_path,
        "size_kb": round(os.path.getsize(out_path) / 1024, 1),
        "camera_frames": n_cam,
        "joint_samples": n_joint,
        "duration_s": duration,
    }
    print(f"✔ sample bag: {stats}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="生成示例 MCAP（无真机演示道具）")
    ap.add_argument("--out-dir", default="./sample_bags")
    ap.add_argument("--name", default="demo_pick_red_cube_001.mcap")
    ap.add_argument("--duration", type=float, default=8.0, help="秒")
    ap.add_argument("--camera-hz", type=float, default=30.0)
    ap.add_argument("--joint-hz", type=float, default=500.0)
    ap.add_argument(
        "--joints",
        type=int,
        default=14,
        help="关节维度，默认 14（ALOHA 双臂，可配 lerobot aloha 仿真评测）",
    )
    args = ap.parse_args()
    make_sample_bag(
        os.path.join(args.out_dir, args.name),
        duration=args.duration,
        camera_hz=args.camera_hz,
        joint_hz=args.joint_hz,
        joints=args.joints,
    )
    print(
        f"下一步: python upload_mcap.py --dir {args.out_dir} "
        f'--embodiment aloha --task pick_red_cube --instruction "pick up the red cube" --success true'
    )


if __name__ == "__main__":
    main()
