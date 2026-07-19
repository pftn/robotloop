# 公开数据集灌库

## 数据源与指定方式

| 来源 | 指定方式 | 示例 |
|---|---|---|
| LeRobot（HF Hub） | `--lerobot <repo_id>` | `--lerobot lerobot/pusht` |
| Open X（RLDS/TFDS） | `--openx <tfds_name>` | `--openx viola/0.1.0` |
| 内置合成数据 | `--demo` | 60 条，CI 用 |
| 预设清单 | `--preset <name>` | 定义于 `config/presets.yaml` |

## 推荐数据集清单（`--preset demo`）

| 数据集 | 来源 | episodes | 大小（GCS 元数据实测） | 用途 |
|---|---|---|---|---|
| `lerobot/pusht` | LeRobot | 206 / 25,650 帧 | ~1.5 GB | 主力演示，ACT 训练配套 |
| `lerobot/pusht_keypoints` | LeRobot | 206 | 2.9 MB（无视频） | CI 秒级验证（`--preset ci`） |
| `lerobot/aloha_sim_insertion_human` | LeRobot | ~50 | ~1 GB | 双臂本体多样性 |
| `nyu_door_opening_surprising_effectiveness` | Open X | 484 | 7.65 GB | Stretch 本体（preset demo） |
| `berkeley_fanuc_manipulation` | Open X | 415 | 9.5 GB | Fanuc 工业臂（preset demo） |
| `viola` | Open X | 150 | 11.17 GB | 超 10GB 线，需 `--yes` |
| `jaco_play` | Open X | 1085 | 9.92 GB | Jaco 本体，接近确认线 |
| `toto` | Open X | — | 137 GB | 不推荐：整包 prepare 不现实 |

## 资源占用估算

- 下载前脚本自动执行 size preflight，打印每个数据源的预估大小与 episode 数
  （Open X 查 GCS 对象元数据，LeRobot 查 HF API）；预估超过 10 GB 需 `--yes` 确认。
- TFDS 必须整包 prepare 到本地才能逐条迭代，无增量下载；下载时长按带宽估算：
  100 Mbps 下 2 GB ≈ 3 分钟，10 GB ≈ 15 分钟。
- 向量侧：3000 条 episode 级 CLIP 文本向量（512d float32）约 6 MB；
  帧级动作摘要向量在 LanceDB 本地，不占用 Milvus/Zilliz 额度。

## 灌库命令

```bash
# 预设清单（config/presets.yaml）
python scripts/ingest_public_datasets.py --preset demo --backend production

# 单数据集 + 条数上限
python scripts/ingest_public_datasets.py \
    --lerobot lerobot/pusht --max-episodes 206 --backend production

# Open X 子集（TF/TFDS 单独 Docker 隔离）
python scripts/ingest_public_datasets.py --openx viola toto --max-episodes 500

# CI（合成数据，本地镜像后端）
python scripts/ingest_public_datasets.py --demo --backend local --store-path ./lake
```

## 验收命令

```bash
curl http://localhost:8000/episodes/stats        # 分布统计（本体/任务/来源/成功率）
curl -X POST "http://localhost:8000/episodes/search" \
    -G --data-urlencode "text_query=抓取红色方块" \
    --data-urlencode "embodiment=aloha" --data-urlencode "success=true"
robotloop quality --store ./lake --report ./quality.html   # 质检报告
```
