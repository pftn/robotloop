"""格式互转 CLI::

    # v2.1 ↔ v3.0（纯本地，无重依赖）
    robotloop-convert v21-to-v30 --in ./ds_v21 --out ./ds_v30
    robotloop-convert v30-to-v21 --in ./ds_v30 --out ./ds_v21

    # RLDS/Open X → LeRobot（需要 tensorflow_datasets）
    robotloop-convert rlds-to-lerobot --tfds bridge_v2/0.1.0 --split "train[:50]" \
        --embodiment widowx --out ./bridge_lerobot --version v2.1

    # MCAP/rosbag2 → LeRobot（采集侧直接产训练数据）
    robotloop-convert mcap-to-lerobot --bag demo.mcap --camera /cam/image_raw \
        --state /joint_states --task pick_red_cube --embodiment aloha --out ./mcap_ds

    # 查看任意 LeRobot 数据集版本与摘要
    robotloop-convert info --path ./ds_v30
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from robotloop.convert.lerobot_v21 import read_lerobot_v21, write_lerobot_v21
from robotloop.convert.lerobot_v30 import read_lerobot_v30, write_lerobot_v30


def detect_version(path: str) -> str:
    info_path = os.path.join(path, "meta", "info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"未找到 {info_path}，不是 LeRobot 数据集目录")
    with open(info_path, encoding="utf-8") as f:
        return json.load(f).get("codebase_version", "unknown")


def _write_any(episodes, root: str, version: str, **kwargs):
    if version in ("v2.1", "2.1", "v21"):
        return write_lerobot_v21(episodes, root, **kwargs)
    if version in ("v3.0", "3.0", "v30"):
        return write_lerobot_v30(episodes, root, **kwargs)
    raise ValueError(f"未知目标版本: {version}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="robotloop-convert", description="RLDS ↔ LeRobot v2.1 ↔ v3.0 互转"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_io(p):
        p.add_argument("--in", dest="src", required=True, help="源数据集目录")
        p.add_argument("--out", dest="dst", required=True, help="目标数据集目录")

    p = sub.add_parser("v21-to-v30", help="LeRobot v2.1 → v3.0")
    _add_io(p)
    p = sub.add_parser("v30-to-v21", help="LeRobot v3.0 → v2.1")
    _add_io(p)

    p = sub.add_parser("rlds-to-lerobot", help="RLDS/TFDS（Open X 子集）→ LeRobot")
    p.add_argument("--tfds", required=True, help="TFDS 名，如 bridge_v2/0.1.0")
    p.add_argument("--split", default="train", help="TFDS split，如 'train[:50]'")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--embodiment", required=True, help="本体标签，如 aloha / widowx")
    p.add_argument(
        "--task", default="", help="覆盖任务名（默认取 language_instruction）"
    )
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--version", default="v2.1", choices=["v2.1", "v3.0"])

    p = sub.add_parser("mcap-to-lerobot", help="MCAP/rosbag2 → LeRobot")
    p.add_argument("--bag", required=True, help=".mcap 文件或 rosbag2 目录")
    p.add_argument("--camera", action="append", default=[], help="相机 topic（可多个）")
    p.add_argument("--state", required=True, help="关节状态 topic")
    p.add_argument(
        "--action-topic", default=None, help="控制指令 topic（缺省用 state 差分近似）"
    )
    p.add_argument("--task", required=True)
    p.add_argument("--instruction", default="")
    p.add_argument("--embodiment", required=True)
    p.add_argument("--fps", type=float, default=30.0, help="对齐目标帧率")
    p.add_argument("--tolerance-ms", type=float, default=20.0)
    p.add_argument("--success", choices=["true", "false", "unknown"], default="unknown")
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--version", default="v2.1", choices=["v2.1", "v3.0"])

    p = sub.add_parser("info", help="查看 LeRobot 数据集摘要")
    p.add_argument("--path", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "v21-to-v30":
        eps = read_lerobot_v21(args.src)
        info = write_lerobot_v30(eps, args.dst)
        print(
            f"✔ v2.1 → v3.0 完成: {info['total_episodes']} episodes / {info['total_frames']} frames → {args.dst}"
        )
    elif args.cmd == "v30-to-v21":
        eps = read_lerobot_v30(args.src)
        info = write_lerobot_v21(eps, args.dst)
        print(
            f"✔ v3.0 → v2.1 完成: {info['total_episodes']} episodes / {info['total_frames']} frames → {args.dst}"
        )
    elif args.cmd == "rlds-to-lerobot":
        from robotloop.convert.rlds import load_tfds_rlds

        eps = load_tfds_rlds(
            args.tfds,
            split=args.split,
            max_episodes=args.max_episodes,
            task=args.task,
            embodiment_tag=args.embodiment,
            fps=args.fps,
        )
        info = _write_any(eps, args.dst, args.version, fps=args.fps)
        print(f"✔ RLDS → {args.version} 完成: {len(eps)} episodes → {args.dst}")
    elif args.cmd == "mcap-to-lerobot":
        from robotloop.pipeline import bag_to_episode

        episode = bag_to_episode(
            args.bag,
            camera_topics=args.camera,
            state_topic=args.state,
            action_topic=args.action_topic,
            task=args.task,
            language_instruction=args.instruction or args.task,
            embodiment_tag=args.embodiment,
            target_fps=args.fps,
            tolerance=args.tolerance_ms / 1000.0,
            success={"true": True, "false": False, "unknown": None}[args.success],
        )
        info = _write_any([episode], args.dst, args.version, fps=args.fps)
        print(f"✔ MCAP → {args.version} 完成: {episode.num_frames} frames → {args.dst}")
    elif args.cmd == "info":
        ver = detect_version(args.path)
        with open(os.path.join(args.path, "meta", "info.json"), encoding="utf-8") as f:
            info = json.load(f)
        print(
            json.dumps(
                {
                    "codebase_version": ver,
                    "robot_type": info.get("robot_type"),
                    "total_episodes": info.get("total_episodes"),
                    "total_frames": info.get("total_frames"),
                    "fps": info.get("fps"),
                    "features": list(info.get("features", {}).keys()),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
