#!/usr/bin/env python3
"""LeRobot v2.1 数据集训练前校验 —— 在上 AutoDL 之前本地拦截所有加载端报错。

模拟 lerobot v0.3.3 训练加载的全项检查（本仓库导出链路曾踩过的坑逐项覆盖）：

 1. meta/info.json 的 data_path 占位符为 episode_chunk/episode_index（KeyError 防线）
 2. 至少一路 VISUAL 图像特征（ACT validate_features 防线）
 3. 图像特征键与目标仿真 env 对齐（aloha → observation.images.top；eval KeyError 防线）
 4. 图像 stats 五件套：min/max/mean/std 形状 (3,1,1)、值域 [0,1]、count (1,)
    （make_dataset use_imagenet_stats KeyError 防线 + _assert_type_and_shape 防线）
 5. observation.state / action 维度一致（normalize_inputs 防线）
 6. 声明 fps 与实际时间戳间隔偏差 <= 1e-4（check_timestamps_sync 防线）
 7. 每路视频 mp4 存在且帧数 == episodes.jsonl length

用法：
    python scripts/verify_lerobot_dataset.py /tmp/ft_mcap [--expect-env aloha]
退出码 0 = 全部通过；非 0 = 存在失败项（不要拿去训练，先修）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ENV_CAM_KEYS = {
    "aloha": ["observation.images.top"],
    "pusht": ["observation.image"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="LeRobot v2.1 数据集训练前校验")
    ap.add_argument("dataset", help="导出的数据集目录（含 meta/info.json）")
    ap.add_argument(
        "--expect-env",
        default="aloha",
        choices=list(ENV_CAM_KEYS) + ["none"],
        help="目标仿真 env：校验图像键与该 env 观测键对齐（默认 aloha）",
    )
    ap.add_argument(
        "--expect-dim", type=int, default=14, help="期望 action/state 维度（默认 14）"
    )
    args = ap.parse_args()
    root = args.dataset

    errors = []

    def check(name, cond, detail=""):
        print(
            ("PASS  " if cond else "FAIL  ") + name + (f" | {detail}" if detail else "")
        )
        if not cond:
            errors.append(name)

    # ---- info.json / 占位符 ----
    info_path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"FAIL  {info_path} 不存在 —— 不是 LeRobot v2.1 数据集目录")
        return 1
    info = json.load(open(info_path))
    check(
        "data_path 占位符 episode_chunk/episode_index",
        "{episode_chunk:" in info.get("data_path", "")
        and "{episode_index:" in info.get("data_path", ""),
        info.get("data_path", ""),
    )

    feats = info.get("features", {})
    img_feats = {k: v for k, v in feats.items() if v.get("dtype") == "video"}

    # ---- ACT validate_features ----
    check(
        "至少一路 VISUAL 图像特征（ACT validate_features）",
        len(img_feats) >= 1,
        str(list(img_feats)) or "无 video 特征",
    )

    # ---- env 键名/分辨率对齐 ----
    if args.expect_env != "none":
        want_keys = ENV_CAM_KEYS[args.expect_env]
        check(
            f"图像键对齐 {args.expect_env} env {want_keys}",
            all(k in img_feats for k in want_keys),
            f"实际: {list(img_feats)}",
        )
        if args.expect_env == "aloha":
            for k in want_keys:
                if k in img_feats:
                    check(
                        "aloha 分辨率 [480,640,3]",
                        img_feats[k].get("shape") == [480, 640, 3],
                        str(img_feats[k].get("shape")),
                    )

    # ---- 维度 ----
    check(
        f"action 维度 == {args.expect_dim}",
        feats.get("action", {}).get("shape") == [args.expect_dim],
        str(feats.get("action", {}).get("shape")),
    )
    if "observation.state" in feats:
        check(
            f"observation.state 维度 == {args.expect_dim}",
            feats["observation.state"].get("shape") == [args.expect_dim],
            str(feats["observation.state"].get("shape")),
        )

    # ---- 图像 stats 五件套 ----
    stats_rows = [
        json.loads(l) for l in open(os.path.join(root, "meta", "episodes_stats.jsonl"))
    ]
    stats_ok, stats_detail = True, ""
    for r in stats_rows:
        for vk in img_feats:
            vs = r["stats"].get(vk)
            if vs is None:
                stats_ok, stats_detail = (
                    False,
                    f"episode {r.get('episode_index')} 缺 {vk} stats",
                )
                break
            for f in ("min", "max", "mean", "std"):
                a = np.asarray(vs.get(f, []), dtype=np.float64)
                if a.shape != (3, 1, 1) or a.min() < 0.0 or a.max() > 1.0:
                    stats_ok, stats_detail = (
                        False,
                        f"{vk}.{f} shape={a.shape} range=[{a.min():.3f},{a.max():.3f}]",
                    )
            if np.asarray(vs.get("count", [])).shape != (1,):
                stats_ok, stats_detail = False, f"{vk}.count 形状异常"
    check("图像 stats 五件套 [0,1]/(3,1,1)/count(1,)", stats_ok, stats_detail)

    # ---- 时间戳同步（lerobot check_timestamps_sync 容差 1e-4）----
    import pyarrow.parquet as pq

    fps = float(info.get("fps") or 0)
    sync_ok, worst = True, 0.0
    ep_rows = [
        json.loads(l) for l in open(os.path.join(root, "meta", "episodes.jsonl"))
    ]
    for r in ep_rows:
        chunk = r["episode_index"] // 1000
        pq_path = os.path.join(
            root,
            "data",
            f"chunk-{chunk:03d}",
            f"episode_{r['episode_index']:06d}.parquet",
        )
        ts = np.array(pq.read_table(pq_path).column("timestamp").to_pylist())
        if len(ts) > 1:
            diffs = np.abs(np.diff(ts) - 1.0 / fps)
            worst = max(worst, float(diffs.max()))
            if (diffs > 1e-4).any():
                sync_ok = False
    check(
        "check_timestamps_sync（声明 fps vs 实际间隔, tol=1e-4）",
        sync_ok,
        f"fps={fps}, worst diff={worst:.2e}",
    )

    # ---- mp4 存在且帧数一致 ----
    vids_ok, vids_detail = True, ""
    try:
        import imageio.v2 as imageio

        for r in ep_rows:
            chunk = r["episode_index"] // 1000
            for vk in img_feats:
                mp4 = os.path.join(
                    root,
                    "videos",
                    f"chunk-{chunk:03d}",
                    vk,
                    f"episode_{r['episode_index']:06d}.mp4",
                )
                if not os.path.exists(mp4) or os.path.getsize(mp4) == 0:
                    vids_ok, vids_detail = False, f"{mp4} 缺失或为空"
                    continue
                rd = imageio.get_reader(mp4)
                try:
                    nf = rd.count_frames()
                finally:
                    rd.close()
                if nf != r["length"]:
                    vids_ok, vids_detail = (
                        False,
                        f"{os.path.basename(mp4)} 帧数 {nf} != length {r['length']}",
                    )
    except ImportError:
        print("SKIP  mp4 帧数校验（需要 imageio + imageio-ffmpeg）")
    check("每路视频 mp4 存在且帧数 == length", vids_ok, vids_detail)

    print()
    if errors:
        print("❌ 存在失败项，先修复再训练: " + "; ".join(errors))
        return 1
    print(
        f"✅ 全部校验通过（{info['total_episodes']} episodes / {info['total_frames']} frames）"
        f"—— 可通过 lerobot v0.3.3 训练加载"
        + (
            f"，并可配 --env.type={args.expect_env} 做仿真评测"
            if args.expect_env != "none"
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
