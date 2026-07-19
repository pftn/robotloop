# 数据质检流水线

质检在 **Ray 流水线入库前自动执行**（`cloud/ray_worker/ray_pipeline.py` 的
`parse_bag_to_episodes`），被过滤的轨迹计数进 Prometheus
（`robotloop_episodes_filtered_total{reason}`）并在 Grafana 出图。

> 方法论：**先见真实数据再定阈值**。下表"阈值怎么定的"一列如实写明
> 每个阈值的来源 —— 工程经验初始值 + 校准方法。没有伪装的精确。

## 1. 失败轨迹过滤（`quality/failure_filter.py`）

| 规则 | 检什么 | 为什么 | 阈值怎么定的 |
|---|---|---|---|
| success 标记 | False 剔除 | 模仿学习训失败轨迹会学到失败模式 | 无阈值，标签语义本身 |
| 未标注（None） | 可配置：设备自采保留 / 公开数据集导出训练时剔除 | 自采数据多数没标 success，一刀切会丢光 | 策略开关而非阈值，按用途定 |
| 截断 | 帧数 < 5 | 遥操作误触发产生的 1~2 帧碎片，无训练价值 | Open X 子集 episode 长度分布的 P1 分位量级；初始值 5，灌真实数据后按分布图收紧 |
| 时长过短 | < 0.2s | 同上，误触发 | 30Hz 下 6 帧 ≈ 0.2s，与截断规则互相印证 |
| 时长过长 | > 600s | 采集员忘停录的"挂机轨迹"，含大量静止帧 | 桌面操作任务（Open X/Bridge 类）P99 在 2~5 分钟量级，600s 留出 2 倍余量 |
| 缺语言指令 | language_instruction 为空 | VLA 训练必需条件信号，缺了这帧无法进 batch | 无阈值，硬约束 |
| 动作全零 | max\|action\| < 1e-8 | 控制链路断开/话题没录上的典型症状，全零动作会教模型"不动" | 浮点零判定的工程惯例 eps；真实数据上该规则只触发于采集事故 |

每条被剔轨迹带 **reason**（`FilterResult.removed`），质检结论可解释，不是黑箱。

## 2. topic 频率异常检测（`quality/freq_anomaly.py`）

直接回答"这条 bag 哪一秒出的事"：

| 规则 | 检什么 | 为什么 | 阈值怎么定的 |
|---|---|---|---|
| gap 掉帧 | dt > 3× 中位周期 | USB 带宽抢占、磁盘写抖动导致丢帧，对齐时这些区间会超 50ms 容差窗被丢 | 3× 周期 = 连丢 2 帧；对齐容差 50ms 与 30Hz 周期（33ms）的交叉验证 |
| drift 漂移 | 实测频率偏离标称 > 15% | 驱动配置错（30Hz 相机跑成 22Hz），对齐后动作-观测系统性错位 | 相机驱动实测波动通常 <5%，15% 留出 3 倍余量 |
| jitter 抖动 | dt 标准差 > 5× 中位周期 | 系统负载毛刺，影响 merge_asof 匹配质量 | 正常采集 jitter 在 0.1~0.5× 周期，5× 是明显异常 |

## 3. 轨迹相似度去重（`quality/dedup.py`）

| 规则 | 检什么 | 为什么 | 阈值怎么定的 |
|---|---|---|---|
| 余弦相似度 > 0.98 | 重复采集（同一遥操作员反复录同一动作） | 训练分布被少数模式主导，泛化变差 | 初始值 0.98 偏保守（只抓近乎复制的）；0.95 更激进。正确做法是先在真实数据集上画出两两相似度直方图，取长尾拐点 |

轨迹向量用动作序列统计摘要（mean/std/min/max），union-find 聚类，
每簇保留最早入库的一条。

## 4. 任务分布统计（`quality/dashboard.py`）

Grafana 看系统吞吐，这里看**数据本身**：任务配比、本体配比、来源配比
（teleop/sim/real）、成功率、时长/帧数分布 —— 分布失衡（某任务占 80%）
本身就是质检结论。

```bash
robotloop quality --store ./lake --report ./quality.html
# 终端输出 JSON 统计 + 生成单文件 HTML 报告（无外部依赖，可直接发群）
```

## 组合用法：导出前的完整质检

```python
from robotloop.quality import filter_episodes, dedup_by_similarity
from robotloop.export import episodes_from_store, export_to_lerobot

eps = episodes_from_store(store, filters={"embodiment_tag": "aloha"})
kept = filter_episodes(eps, keep_unlabeled=False).kept          # 第一道闸
ids, _, traj = store.embeddings()
keep_ids = set(dedup_by_similarity(ids, traj, 0.98).kept_ids)   # 第二道闸
export_to_lerobot(store, "./ft_data",
                  episode_ids=[e.episode_id for e in kept if e.episode_id in keep_ids],
                  version="v2.1")
```
