"""RLDS ↔ LeRobot ↔ RobotLoop 概念映射。

这是理解整个数据闭环的领域模型层的钥匙：
三套生态用三套词汇描述同一件事，RobotLoop 的 Episode/Step 模型是它们的公共超集。

- RLDS（TFDS/Reverb 系）：Open X-Embodiment、DROID 原始发布格式
- LeRobot（Hugging Face 系）：v2.1（π0/OpenPI、GR00T 当前要求）与 v3.0（新版官方工具链）
- RobotLoop：Iceberg 结构化元数据 + 向量特征库 + 对象存储图像/视频

``render_markdown()`` 生成 README 用的对照表。
"""

from __future__ import annotations

from typing import Dict, List

CONCEPT_MAPPING: List[Dict[str, str]] = [
    {
        "concept": "数据集",
        "rlds": "TFDS DatasetBuilder（如 `bridge_v2/0.1.0`）",
        "lerobot": "LeRobotDataset（HF repo，含 data/ videos/ meta/）",
        "robotloop": "Iceberg namespace `robotloop` + 特征表",
        "note": "`episodes.dataset_name` 记录来源数据集",
    },
    {
        "concept": "一条轨迹",
        "rlds": "Episode（steps 的 Sequence）",
        "lerobot": "一个 episode（parquet 行段 + mp4 片段）",
        "robotloop": "`episodes` 表一行",
        "note": "主键 `episode_id`；LeRobot 的 `episode_index` 保留为来源序号",
    },
    {
        "concept": "一帧",
        "rlds": "Step",
        "lerobot": "一帧（data 一行 + 各相机视频帧）",
        "robotloop": "frames/{episode_id}.parquet 中一行（Iceberg 只存 parquet_path 指针）",
        "note": "`frame_index` 三边一致，跨格式对齐的锚点",
    },
    {
        "concept": "任务/语言指令",
        "rlds": "`language_instruction`（step 级字符串 feature）",
        "lerobot": "`meta/tasks.jsonl` + 帧级 `task_index`",
        "robotloop": "`episodes.task`（结构化）+ `language_instruction`（文本）",
        "note": "文本是 CLIP 语义检索的编码对象",
    },
    {
        "concept": "观测",
        "rlds": "`observation`（嵌套 tensor dict）",
        "lerobot": "`observation.state` / `observation.images.*` 特征列",
        "robotloop": "`steps.state` + `steps.image_paths`",
        "note": "图像不内嵌，落 MinIO 对象存储，Iceberg 只存路径",
    },
    {
        "concept": "动作",
        "rlds": "`action`",
        "lerobot": "`action` 列（float32 数组）",
        "robotloop": "`steps.action`（list<double>）",
        "note": "维度随本体不同（7/8/14...），schema 不约束",
    },
    {
        "concept": "奖励",
        "rlds": "`reward`（RL 语义）",
        "lerobot": "无标准字段（可选 `next.reward`）",
        "robotloop": "`steps.reward`（可空）",
        "note": "LeRobot 面向模仿学习，通常无 reward",
    },
    {
        "concept": "终止标记",
        "rlds": "`is_terminal` / `is_last`",
        "lerobot": "无（靠 episode 边界 + frame_index）",
        "robotloop": "`steps.is_terminal`",
        "note": "导出 LeRobot 时由 frame_index == length-1 推导",
    },
    {
        "concept": "成功标记",
        "rlds": "无标准（少数数据集自定义，如 `success`）",
        "lerobot": "无标准",
        "robotloop": "`episodes.success`（bool，可空）",
        "note": "RobotLoop 显式建模 —— 失败轨迹过滤的第一字段",
    },
    {
        "concept": "机器人本体",
        "rlds": "无标准（隐含在 dataset 元信息）",
        "lerobot": "`meta/info.json` 的 `robot_type`",
        "robotloop": "`episodes.embodiment_tag` + `robot_type`",
        "note": "取值与 GR00T embodiment tag / Open X 命名对齐",
    },
    {
        "concept": "采集来源",
        "rlds": "无",
        "lerobot": "无",
        "robotloop": "`episodes.source`（teleop | sim | real）",
        "note": "RobotLoop 扩展维度：区分遥操作/仿真/真机自主",
    },
    {
        "concept": "时间基准",
        "rlds": "step 序号（离散，无墙钟时间）",
        "lerobot": "`timestamp` 列（秒，按 fps 等距）",
        "robotloop": "`steps.timestamp`（对齐后统一时刻）",
        "note": "多 topic 原始时间戳在入库前由 ingest.align 对齐",
    },
    {
        "concept": "帧率",
        "rlds": "无显式（隐含于采集频率）",
        "lerobot": "`info.json` 的 `fps`",
        "robotloop": "`episodes.fps`（对齐目标帧率）",
        "note": "30Hz 相机 + 500Hz 关节流 → 统一 fps",
    },
    {
        "concept": "图像/视频",
        "rlds": "内嵌 image tensor",
        "lerobot": "`videos/` 下按相机分目录的 mp4",
        "robotloop": "MinIO 对象 + `image_paths` map",
        "note": "导出 LeRobot 时按需重编码为 mp4",
    },
    {
        "concept": "格式版本",
        "rlds": "TFDS version（如 0.1.0）",
        "lerobot": "`codebase_version`（v2.0 / v2.1 / v3.0）",
        "robotloop": "—（由 convert 层双版本兼容）",
        "note": "π0/OpenPI 要 v2.1；新版 LeRobot 工具链要 v3.0",
    },
]


def render_markdown() -> str:
    """生成 README 用的概念映射表（Markdown）。"""
    lines = [
        "| 概念 | RLDS（TFDS/Open X） | LeRobot（HF） | RobotLoop | 备注 |",
        "|---|---|---|---|---|",
    ]
    for row in CONCEPT_MAPPING:
        lines.append(
            f"| {row['concept']} | {row['rlds']} | {row['lerobot']} | {row['robotloop']} | {row['note']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(render_markdown())
