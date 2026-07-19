# lerobot-convert

LeRobot **v2.1 ↔ v3.0 批量互转 + 校验**，`pip install` 即用。

> 本工具是 [RobotLoop](https://gitee.com/pftnzc/robotloop) 数据闭环的一部分
> （设备数据 → 数据湖 → 混合检索 → 训练导出）。单独发布是因为格式转换
> 这件事本身值得一个干净的小工具。GitHub topics: `lerobot` `rlds`
> `embodied-ai` `robotics`。

## 为什么需要它

LeRobot 生态现在双版本并存，而下游框架各自站队：

| 格式 | 布局 | 谁在用 |
|---|---|---|
| v2.1 | 每 episode 一个 parquet + JSONL 元数据 | π0 / OpenPI、GR00T 等要求 |
| v3.0 | 多 episode 聚合 file-XXX.parquet + Parquet 元数据 + 100MB 文件轮转 | lerobot ≥ 0.4.0 官方默认 |

官方仓库有一次性迁移脚本，本工具的增量是**工程化**：批量、稳定、带校验、
失败不静默。转换是文件级的（不过领域模型、不解码图像），大数据集也能跑。

## 安装

```bash
pip install .
# 或开发模式
pip install -e .
```

## 使用

```bash
# v2.1 -> v3.0（转换后自动校验，校验失败退出码 1）
lerobot-convert v21-to-v30 ./my_ds_v21 ./my_ds_v30 --data-file-size-mb 100

# v3.0 -> v2.1
lerobot-convert v30-to-v21 ./my_ds_v30 ./my_ds_v21

# 目录级批量（SRC_DIR 下每个子目录是一个数据集）
lerobot-convert batch-v21-to-v30 ./all_v21 ./all_v30

# 独立校验（v2.1/v3.0 自适应）
lerobot-convert validate ./my_ds_v30
```

## before / after

```
$ lerobot-convert v21-to-v30 ./demo_v21 ./demo_v30
2026-07-17 22:55 INFO wrote ./demo_v30/data/chunk-000/file-000.parquet (1673 rows)
2026-07-17 22:55 INFO v2.1 -> v3.0 done: 60 episodes -> ./demo_v30
{
  "info_codebase_version": "v3.0",
  "validation": {
    "version": "v3.0", "total_episodes": 60, "total_frames": 1673,
    "ok": true, "errors": [], "warnings": []
  }
}
```

## 校验项

1. `meta/info.json` 存在且 `codebase_version` 合法
2. episode 数量：元数据记录数 == 数据文件覆盖的 episode 数
3. `episode_index` 不重不漏（0..N-1 连续）
4. 每 episode 内 `frame_index` 连续
5. 总帧数 == 元数据 length 之和
6. v3.0 专属：meta 里的 chunk/file 指针对应真实文件

## 已知取舍（如实说明）

- **v3.0 → v2.1 的 per-episode 统计不可还原**：v3.0 只存全局聚合统计，
  拆回 v2.1 时 `episodes_stats.jsonl` 每条的 stats 从全局复制并加
  `note` 字段注明。帧数据与元数据本身无损。
- 视频文件原样复制（v3.0 的视频聚合编码是独立步骤，不阻塞数据转换）。
