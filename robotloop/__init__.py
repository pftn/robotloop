"""RobotLoop 具身智能数据闭环核心包。

在既有 RobotLoop 平台（MinIO/Kafka/Ray/LanceDB/Milvus/Iceberg）之上，
新增面向机器人学习（VLA）数据流的六组能力：

1. ``robotloop.schema``    —— Episode 领域模型、Iceberg episodes 表（帧数据走 MinIO Parquet+指针）
2. ``robotloop.ingest``    —— MCAP / rosbag2 解析（rosbags 纯 Python）+ merge_asof 对齐 + 解析器注册表
3. ``robotloop.convert``   —— LeRobot v2.1/v3.0 读写 + RLDS 单向读取
4. ``robotloop.datasets``  —— Open X / LeRobot Hub / AgiBot World 公开数据集统一灌库
5. ``robotloop.retrieval`` —— CLIP 语义检索 × Iceberg 结构化过滤的混合检索
6. ``robotloop.export``    —— 一键导出 LeRobot 数据集 + ACT 训练脚本（GR00T 留作进阶）
7. ``robotloop.quality``   —— 数据质检流水线（失败过滤 / 频率异常 / 相似去重 / 分布统计）
"""

__version__ = "0.3.0"
