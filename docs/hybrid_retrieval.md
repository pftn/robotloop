# 公开数据集灌库 × 跨库混合语义检索

## 统一灌库

```
Open X 子集（TFDS/RLDS）          ┐
LeRobot Hub 数据集（v2.1/v3.0）   ├──► Episode ──► ingest_episodes ──► Iceberg（结构化）
AgiBot World（LeRobot 镜像/原生）  ┘                     │            └ Milvus/LanceDB（向量）
自采 MCAP/rosbag2                 ┘               文本 embedding + 轨迹 embedding
```

入库即归一：不管来自哪个数据集，`embodiment_tag / task / success / source`
都落在同一套 schema 上，检索零差异。Open X 常见子集的本体映射内置
（`OPENX_EMBODIMENT_MAP`），灌库不用手工查表。

## 混合检索

demo 场景：**"找所有 ALOHA 双臂成功抓取红色方块的轨迹"**

```
自然语言 ──parse_nl_query──► 结构化谓词: embodiment_tag='aloha' AND success=true
                          └ 残余语义文本: "抓取红色方块"
结构化谓词 ──► Iceberg row_filter（谓词下推 + embodiment 分区裁剪）
残余文本   ──► CLIP encode ──► Milvus ANN
两侧取交集，按语义相似度排序
```

- 编码器：sentence-transformers `clip-ViT-B-32`（与平台 Ray 流水线同款，
  向量空间同源）；无 GPU/网络时自动降级 HashEncoder，链路不断。
- NL 解析是 demo 翻译层（关键词表），生产可换 LLM function-calling，接口不变。
- 存储后端双实现：`LocalStore`（parquet 镜像，离线 demo/CI）与
  `MilvusIcebergStore`（生产，复用平台环境变量）。

## 离线跑通 demo（无需下载任何数据）

```bash
robotloop ingest-demo --store ./lake -n 60
robotloop search --store ./lake --query "找所有 ALOHA 双臂成功抓取红色方块的轨迹"
# structured_filters: {"embodiment_tag":"aloha","success":true}
# 结果按"抓取红色方块"语义排序，pick_red_cube 轨迹置顶
```

## 真实数据集灌库

```python
from robotloop.datasets import load_openx_subset, load_lerobot_hub, load_agibot_lerobot_mirror, ingest_episodes
from robotloop.retrieval.store import LocalStore

store = LocalStore("./lake")
ingest_episodes(load_openx_subset("bridge_v2/0.1.0", split="train[:200]"), store)
ingest_episodes(load_lerobot_hub("lerobot/aloha_mobile_cabinet"), store)
ingest_episodes(load_agibot_lerobot_mirror("<hf-mirror-repo>"), store)
```

## API 集成

在 `cloud/api/main.py` 末尾挂路由，复用平台部署：

```python
from robotloop.retrieval.api import make_episode_router
from robotloop.retrieval.store import MilvusIcebergStore
app.include_router(make_episode_router(MilvusIcebergStore()))
# POST /episodes/search  {"text_query": "找所有 ALOHA 双臂成功抓取红色方块的轨迹"}
# GET  /episodes/stats   任务分布统计（质检 dashboard 数据面）
```

## CI / 离线环境实测数字（HashEncoder 降级模式）

以下数字在**无 GPU、无 sentence-transformers** 的 CI 环境跑出，编码器自动
降级为 HashEncoder（MD5 哈希词袋）——链路完整可用，语义质量低于 CLIP。
生产模式（CLIP + Milvus + Iceberg REST）数字以集群实测为准。

| 项目 | 实测值（CI 环境） |
|---|---|
| 混合检索延迟（60 episodes，本机） | 19.2ms |
| NL 解析 | 「ALOHA 双臂成功抓取红色方块的轨迹」→ `{"embodiment_tag":"aloha","success":true}` + 语义残余「双臂 抓取红色方块」 |
| 消融对比 | 仅语义+embodiment 召回 5 条中 3 条是失败轨迹；加 `success=true` 后 0/5 |
| 库内统计 | 60 episodes：aloha 24 / agibot_g1 18 / widowx 12 / google_robot 6，标注成功率 56.67% |

复现：`jupyter/hybrid_search_demo.ipynb`（默认 `MODE=local`，输出为实际执行结果）。

## Zilliz 免费版运维笔记

Zilliz Cloud 免费版有 CU 与容量上限（serverless 约 5GB）。灌公开数据集时
先用 `--max-episodes 50` 灌子集验证链路，确认无误再放量。`episode_vectors` 集合只存 CLIP 512d 文本向量（轨迹向量维度随
本体变化，留在 LanceDB 做去重/相似度分析），向量体积小，子集验证足够。
