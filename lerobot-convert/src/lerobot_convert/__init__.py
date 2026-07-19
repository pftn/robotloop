"""lerobot-convert — LeRobot v2.1 <-> v3.0 批量互转 + 校验。

文件级转换（不解码图像、不过领域模型），基于官方迁移脚本的布局约定工程化：
- 批量：目录级一次转换，100MB 文件轮转
- 稳定：逐 episode 校验帧数/索引连续性，失败即报不静默
- 带校验：validate 子命令独立可用

本工具是 RobotLoop 数据闭环的一部分（https://github.com/pftn/robotloop）。
"""

from lerobot_convert.convert import (
    convert_v21_to_v30,
    convert_v30_to_v21,
    detect_version,
)
from lerobot_convert.validate import validate_dataset

__version__ = "1.0.0"
__all__ = [
    "convert_v21_to_v30",
    "convert_v30_to_v21",
    "detect_version",
    "validate_dataset",
    "__version__",
]
