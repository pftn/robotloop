import os
import json
import time
import random
import hashlib
import logging
import shutil
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

OUTPUT_DIR = "/tmp/robot_data"


def generate_robot_data(output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    logs = []
    for i in range(100):
        logs.append({
            "timestamp": timestamp + i,
            "device_id": "robot-001",
            "state_json": {
                "battery": 100 - i * 0.8,
                "imu": {
                    "accel": {"x": random.uniform(-0.1, 0.1), "y": random.uniform(-0.1, 0.1), "z": 9.8},
                    "gyro": {"x": random.uniform(-0.01, 0.01), "y": random.uniform(-0.01, 0.01),
                             "z": random.uniform(-0.01, 0.01)}
                },
                "position": {
                    "lat": 22.5431 + random.uniform(-0.001, 0.001),
                    "lng": 113.9295 + random.uniform(-0.001, 0.001),
                    "alt": 0.0
                },
                "joint_angles": [random.uniform(0, 180) for _ in range(6)],
                "camera": "rgb_01",
                "lidar": "lidar_01"
            },
            "event": "normal" if i < 90 else "low_battery"
        })
    with open(os.path.join(output_dir, "robot_state.json"), "w") as f:
        json.dump(logs, f, indent=2)
    logging.info("机器人状态日志已生成")

    # 生成模拟媒体文件
    sample_path = os.path.join(os.path.dirname(__file__), "sample.jpg")
    for i in range(10):
        dst = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        shutil.copy(sample_path, dst)
    for i in range(2):
        with open(os.path.join(output_dir, f"lidar_{i:04d}.pcd"), "wb") as f:
            f.write(os.urandom(50 * 1024 * 1024))  # 50MB
    logging.info("媒体文件已生成")

    # 生成模拟 ROS 消息序列（JSON 格式）
    ros_messages = []
    for i in range(50):  # 50条示例消息
        t = timestamp + i
        # IMU 消息
        ros_messages.append({
            "topic": "/imu",
            "timestamp": t,
            "msg_type": "sensor_msgs/Imu",
            "data": {
                "linear_acceleration": {"x": random.uniform(-0.1, 0.1), "y": random.uniform(-0.1, 0.1), "z": 9.8},
                "angular_velocity": {"x": random.uniform(-0.01, 0.01), "y": random.uniform(-0.01, 0.01),
                                     "z": random.uniform(-0.01, 0.01)}
            }
        })
        # GPS 消息
        ros_messages.append({
            "topic": "/gps",
            "timestamp": t + 0.1,
            "msg_type": "sensor_msgs/NavSatFix",
            "data": {
                "latitude": 22.5431 + random.uniform(-0.001, 0.001),
                "longitude": 113.9295 + random.uniform(-0.001, 0.001),
                "altitude": 0.0
            }
        })
        # 相机消息（仅元数据，实际图片已生成）
        ros_messages.append({
            "topic": "/camera/rgb",
            "timestamp": t + 0.2,
            "msg_type": "sensor_msgs/Image",
            "data": {"file": f"frame_{i % 10:04d}.jpg"}
        })

    with open(os.path.join(output_dir, "ros_data.json"), "w") as f:
        json.dump(ros_messages, f, indent=2)
    logging.info("ROS 数据已生成（模拟）")

    # 生成 manifest
    manifest = {}
    for root, _, files in os.walk(output_dir):
        for file in files:
            full = os.path.join(root, file)
            with open(full, "rb") as f:
                manifest[file] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Manifest 已生成")
    return output_dir


def main():
    logging.info("开始生成模拟机器人数据...")
    generate_robot_data()
    logging.info("模拟机器人数据生成完毕")


if __name__ == "__main__":
    main()
