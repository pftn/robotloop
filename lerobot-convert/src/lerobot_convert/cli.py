"""lerobot-convert CLI。

    lerobot-convert v21-to-v30 SRC DST [--data-file-size-mb 100]
    lerobot-convert v30-to-v21 SRC DST
    lerobot-convert validate DATASET_DIR
    lerobot-convert batch-v21-to-v30 SRC_DIR DST_DIR   # 目录级批量

退出码：validate 发现 error 时返回 1（CI 可挂）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from lerobot_convert.convert import (
    convert_v21_to_v30,
    convert_v30_to_v21,
    detect_version,
)
from lerobot_convert.validate import validate_dataset


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        prog="lerobot-convert",
        description="LeRobot v2.1 <-> v3.0 batch converter with validation "
                    "(part of the RobotLoop data closed-loop)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("v21-to-v30", help="v2.1 目录 -> v3.0 目录")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--data-file-size-mb", type=int, default=100)
    p.add_argument("--no-videos", action="store_true")

    p = sub.add_parser("v30-to-v21", help="v3.0 目录 -> v2.1 目录")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--no-videos", action="store_true")

    p = sub.add_parser("batch-v21-to-v30", help="批量：SRC_DIR 下所有 v2.1 数据集 -> DST_DIR")
    p.add_argument("src_dir")
    p.add_argument("dst_dir")
    p.add_argument("--data-file-size-mb", type=int, default=100)

    p = sub.add_parser("validate", help="校验数据集（v2.1/v3.0 自适应）")
    p.add_argument("root")
    p.add_argument("--skip-frames", action="store_true", help="跳过逐帧 frame_index 检查（大数据集加速）")

    args = ap.parse_args(argv)

    if args.cmd == "v21-to-v30":
        info = convert_v21_to_v30(args.src, args.dst,
                                  data_file_size_mb=args.data_file_size_mb,
                                  copy_videos=not args.no_videos)
        rep = validate_dataset(args.dst)
        print(json.dumps({"info_codebase_version": info["codebase_version"],
                          "validation": rep.as_dict()}, ensure_ascii=False, indent=2))
        return 0 if rep.ok else 1

    if args.cmd == "v30-to-v21":
        info = convert_v30_to_v21(args.src, args.dst, copy_videos=not args.no_videos)
        rep = validate_dataset(args.dst)
        print(json.dumps({"info_codebase_version": info["codebase_version"],
                          "validation": rep.as_dict()}, ensure_ascii=False, indent=2))
        return 0 if rep.ok else 1

    if args.cmd == "batch-v21-to-v30":
        results = {}
        for name in sorted(os.listdir(args.src_dir)):
            src = os.path.join(args.src_dir, name)
            if not os.path.isdir(src):
                continue
            try:
                if detect_version(src) != "v2.1":
                    results[name] = "skip(not v2.1)"
                    continue
                convert_v21_to_v30(src, os.path.join(args.dst_dir, name),
                                   data_file_size_mb=args.data_file_size_mb)
                rep = validate_dataset(os.path.join(args.dst_dir, name))
                results[name] = "ok" if rep.ok else f"INVALID: {rep.errors}"
            except Exception as e:
                results[name] = f"FAILED: {e}"
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0 if all(v == "ok" for v in results.values()) else 1

    if args.cmd == "validate":
        rep = validate_dataset(args.root, check_frames=not args.skip_frames)
        print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
        return 0 if rep.ok else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
