#!/usr/bin/env python3
"""模拟机器人多模态数据生成器"""

import os
import json
import time
import random
import shutil
import hashlib
import logging
from datetime import datetime, timedelta
import numpy as np
from PIL import Image

OUTPUT_DIR = "/tmp/robot_data"
SAMPLE_IMG = os.path.join(os.path.dirname(__file__), "sample.jpg")
NUM_SCENES = 1000

SCENE_TYPES = ["kitchen", "living_room", "warehouse", "corridor", "lab"]
WEATHERS = [
    "indoor",
    "indoor",
    "indoor",
    "outdoor_clear",
    "outdoor_rain",
    "outdoor_fog",
]
DESCRIPTIONS = {
    "kitchen": [
        "杂乱厨房，操作台上有碗碟和锅，机械臂尝试抓取玻璃杯但中途滑脱，手爪提前回缩",
        "整洁厨房，机械臂成功抓取马克杯并平稳移动",
        "杂乱厨房台面，机械臂成功抓取玻璃杯但略微抖动",
        "厨房水槽边，机械臂清洗盘子后放置晾干架",
        "厨房角落，机械臂从冰箱取出鸡蛋并轻放在台面上",
    ],
    "living_room": [
        "客厅茶几上散落书籍，机器人避障清扫地面",
        "客厅沙发区，机器人语音交互并递送遥控器",
        "昏暗客厅，机器人开启落地灯并整理抱枕",
    ],
    "warehouse": [
        "仓库货架间，叉车机器人搬运重型箱子",
        "狭窄通道，机器人扫描条形码并更新库存",
        "高位货架，机器人伸展臂取放轻量包裹",
    ],
    "corridor": [
        "长走廊，配送机器人避让行人后继续前进",
        "医院走廊，消毒机器人喷雾消毒地面",
        "办公楼走廊，巡逻机器人检测异常噪音",
    ],
    "lab": [
        "实验室工作台，移液机器人精确转移液体",
        "生物安全柜旁，机械臂夹取试管并离心",
        "电子实验室，焊接机器人修复电路板",
    ],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ensure_sample_image():
    if not os.path.exists(SAMPLE_IMG):
        img = Image.new("RGB", (640, 480), color=(150, 150, 150))
        os.makedirs(os.path.dirname(SAMPLE_IMG), exist_ok=True)
        img.save(SAMPLE_IMG)
        logging.info("sample.jpg generated")


def build_manifest():
    manifest = {}
    for root, _, files in os.walk(OUTPUT_DIR):
        for name in files:
            full = os.path.join(root, name)
            with open(full, "rb") as f:
                manifest[name] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_sample_image()

    scenes = []
    base_time = datetime(2026, 7, 9, 8, 0, 0)

    for i in range(NUM_SCENES):
        scene_id = f"scene_{i:04d}"
        scene_type = random.choice(SCENE_TYPES)
        weather = random.choice(WEATHERS)
        desc = random.choice(DESCRIPTIONS[scene_type])
        offset_seconds = random.randint(0, 12 * 3600)
        ts = base_time + timedelta(seconds=offset_seconds)
        timestamp_start = ts.strftime("%Y-%m-%d %H:%M:%S")

        bag_file = f"s3://rosbags/{scene_type}_{weather}_{i % 10:02d}.bag"
        start_t = random.randint(0, 3000)
        end_t = start_t + random.randint(300, 600)
        file_path = f"{bag_file}#t={start_t},{end_t}"

        quality = round(random.uniform(0.6, 1.0), 2)

        frame_files = []
        for f_idx in range(random.randint(2, 5)):
            frame_name = f"{scene_id}_frame_{f_idx:03d}.png"
            shutil.copy(SAMPLE_IMG, os.path.join(OUTPUT_DIR, frame_name))
            frame_files.append(frame_name)

        scenes.append(
            {
                "scene_id": scene_id,
                "timestamp_start": timestamp_start,
                "file_path": file_path,
                "scene_type": scene_type,
                "weather": weather,
                "quality_score": quality,
                "scene_description": desc,
                "frame_files": frame_files,
            }
        )

    metadata_path = os.path.join(OUTPUT_DIR, "scene_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(scenes, f, indent=2)
    logging.info(f"Metadata written to {metadata_path}")

    manifest = build_manifest()
    logging.info(f"Manifest generated with {len(manifest)} files")


if __name__ == "__main__":
    generate_dataset()
