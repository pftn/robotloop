# Episode 领域模型 × Iceberg schema

## 领域模型

```
Episode（轨迹：检索/过滤/统计最小单元）
  ├── episode_id / dataset_name / episode_index
  ├── task / language_instruction
  ├── embodiment_tag / robot_type
  ├── source (teleop|sim|real) / success / duration / fps
  └── steps: List[Step]（帧：训练最小单元，frame_index 连续）
        ├── frame_index / timestamp
        ├── observation: {images: {cam: path}, state: [float]}
        └── action / reward / is_terminal
```

关键设计：

- **success 显式建模**：RLDS/LeRobot 都没有标准的成功标记字段，但"失败轨迹过滤"
  是数据闭环的高频操作，RobotLoop 把它提为 first-class 字段（可空 = 未标注）。
- **source 三值**：teleop（遥操作）/ sim（仿真）/ real（真机自主）——
  sim2real 配比分析、采集成本核算都依赖这一列。
- **图像不内嵌**：像素落 MinIO 对象存储，Iceberg 只存路径 map。
  与 LeRobot 的 `videos/` 外置 mp4 同一哲学。

## Iceberg 物理落地（设计约束：Iceberg 严禁存帧级行）

| 存储 | 内容 | 布局/分区 | 服务 |
|---|---|---|---|
| Iceberg `robotloop.episodes` | 轨迹级元数据（含 `parquet_path` 指针） | `embodiment_tag` identity 分区 | 检索/过滤/统计 |
| MinIO `frames/{episode_id}.parquet` | 帧级数据（一个 episode 一个文件） | zstd 压缩 | 训练导出按指针直读 |

- **没有也不得有帧级 Iceberg 表**：帧数据走 Parquet 文件 + 指针
  （LeRobot v3 文件块思路），`robotloop/schema/frame_store.py` 是读写入口，
  `robotloop/schema/sink.py` 的 EpisodeSink 是一体化生产写入端（REST catalog）。
- 向量（文本 embedding、轨迹 embedding）**不进 Iceberg**，进 Milvus/LanceDB ——
  结构化与 ANN 分层，与 RobotLoop v1.0 平台一致。
- pyiceberg 为可选依赖；无 Iceberg 环境时用 `LocalStore`（本地镜像，同样
  parquet_path 指针布局）开发测试，schema 由同一份 pyarrow 定义转换，保证不漂移。

## RLDS ↔ LeRobot ↔ RobotLoop 概念映射

见 README 的对照表（由 `robotloop.schema.mapping.render_markdown()` 生成，
改映射只需改一处数据）。
