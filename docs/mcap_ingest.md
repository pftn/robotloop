# MCAP / rosbag2 解析 × 多 topic 时间戳对齐

采集侧最痛的工程问题：相机 30Hz、关节 500Hz、夹爪 100Hz，各自打时间戳，
训练要的却是同一逻辑时刻的 `(image, state, action)` 三元组。

## 链路

```
demo.mcap / rosbag2/ ──► {topic: TopicStream} ──► align_streams ──► AlignedFrame ──► Episode
   read_mcap()            (timestamps, values)      对齐引擎           同步帧         领域模型
```

- **read_mcap()**：`pip install mcap`（解码另需 `mcap-ros2-support`）。
  消息抽取是注入式的 `TopicExtractor`，内置 JointState / (Compressed)Image。
- **read_rosbag2()**：`pip install rosbags`（纯 Python，无需 ROS2 运行时）。
  产出与 MCAP 完全相同的 TopicStream —— 采集格式是工程细节，领域模型只认 TopicStream。
- **read_jsonl_log()**：JSONL 模拟源，接口一致，用于无真机验证与 CI。

## 对齐引擎（`ingest/align.py`）

| 参数 | 说明 |
|---|---|
| `reference_topic` | 以该 topic 时间戳为参考轴（通常选相机——帧是训练的天然节拍） |
| `target_fps` | 或在各流时间交集上按目标帧率重采样（二选一） |
| `strategy` | `nearest`（最近邻）/ `latest_before`（asof 语义） |
| `tolerance` | 最大允许偏差，超过判缺失（默认 20ms ≈ 30Hz 相机半周期） |
| `required` | 必需 topic 缺失即丢帧 |

每次对齐产出 **AlignmentReport**：各 topic 匹配率、平均/最大偏差、丢帧数。

实测（tests/test_align.py）：30Hz 相机（±3ms 抖动 + 掉帧）× 500Hz 关节流，
89/90 帧命中，关节流最大偏差 < 2ms（一个周期内）。

## action 语义（`frames_to_episode`）

- 包里录了控制指令 → `action_topic` 直接用
- 只录了状态 → `action_from_state_diff=True`：a_t = s_{t+1} − s_t
  （遥操作日志的常用近似；末帧动作置 0 并标 is_terminal）

## CLI

```bash
robotloop ingest-mcap --bag demo.mcap \
    --camera /cam/image_raw --state /joint_states \
    --task pick_red_cube --instruction "抓取红色方块" \
    --embodiment aloha --source teleop --success true \
    --store ./lake
```
