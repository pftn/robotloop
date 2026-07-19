# RLDS ↔ LeRobot v2.1 ↔ v3.0 互转

v2.1/v3.0 并存是当下真实痛点：**π0/OpenPI、GR00T 要求 v2.1；新版 LeRobot
官方工具链（lerobot ≥ 0.4.0）是 v3.0**。

## 架构：hub-and-spoke

不写 N×N 两两转换器。所有格式 ↔ **Episode 领域模型** ↔ 所有格式，
v2.1 ↔ v3.0 也走 Episode 中转——成功标记、来源、本体标签全部无损
（挂在 info.json 的 `robotloop` 扩展字段，不破坏官方规范，训练侧忽略即可）。

## 两个版本的物理差异

| | v2.1 | v3.0 |
|---|---|---|
| 数据 | `data/chunk-XXX/episode_YYYYYY.parquet`（一 episode 一文件） | `data/chunk-XXX/file-YYY.parquet`（多 episode 聚合，默认 100MB/文件） |
| 视频 | 一 episode 一 mp4 | 多 episode 聚合 mp4，边界靠 `videos/<key>/from_timestamp` 定位 |
| 任务表 | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| episode 元数据 | `meta/episodes.jsonl` | `meta/episodes/chunk-XXX/file-YYY.parquet`（含 `data/chunk_index`、`dataset_from/to_index`） |
| 统计 | `meta/episodes_stats.jsonl`（逐 episode） | `meta/stats.json`（全局合并） |
| 识别 | `info.json: codebase_version=v2.1` | `=v3.0` |

## RLDS 方向

- `rlds_episode_to_episode()`：TFDS 迭代出的 dict → Episode（Open X 风格兼容：
  step 级 `language_instruction`、末帧 reward>0 近似 success）
- `episode_to_rlds_dict()`：Episode → RLDS 规范 dict（交给 rlds/TFDS builder 落 TFRecord）
- `load_tfds_rlds()`：`pip install tensorflow_datasets` 后直接拉 Open X 子集

## CLI

```bash
robotloop-convert v21-to-v30 --in ./ds_v21 --out ./ds_v30
robotloop-convert v30-to-v21 --in ./ds_v30 --out ./ds_v21
robotloop-convert rlds-to-lerobot --tfds bridge_v2/0.1.0 --split "train[:50]" \
    --embodiment widowx --out ./bridge_lerobot --version v2.1
robotloop-convert mcap-to-lerobot --bag demo.mcap --camera /cam/image_raw \
    --state /joint_states --task pick_red_cube --embodiment aloha --out ./mcap_ds
robotloop-convert info --path ./ds_v30
```

测试覆盖：v2.1/v3.0 各自往返无损、跨版本往返无损、文件大小轮转、RLDS 双向。
