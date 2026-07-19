"""多 topic 时间戳对齐测试：30Hz 相机 + 500Hz 关节流的经典场景。"""

import numpy as np
import pytest

from robotloop.ingest.align import TopicStream, align_streams


@pytest.fixture
def cam_joint_streams():
    t0 = 1_700_000_000.0
    rng = np.random.default_rng(7)
    cam_ts = t0 + np.arange(0, 3.0, 1 / 30) + rng.uniform(-0.003, 0.003, 90)
    joint_ts = t0 + np.arange(0, 3.0, 1 / 500)
    cam = TopicStream("/cam/image", cam_ts, [f"img_{i}" for i in range(len(cam_ts))], kind="image")
    joint = TopicStream("/joint_states", joint_ts,
                        [[float(i)] * 7 for i in range(len(joint_ts))], kind="state")
    return cam, joint


def test_nearest_align_reference(cam_joint_streams):
    cam, joint = cam_joint_streams
    frames, report = align_streams([cam, joint], reference_topic="/cam/image", tolerance=0.020)
    assert len(frames) == 90                      # 相机每帧都对上
    assert report.dropped_frames == 0
    stat = report.stats["/joint_states"]
    assert stat.matched == 90
    assert stat.max_offset_ms < 2.0               # 500Hz 流最大偏差不超过 2ms（一个周期）
    assert stat.nominal_hz == pytest.approx(500.0, rel=0.01)


def test_camera_dropped_frame_tolerance(cam_joint_streams):
    cam, joint = cam_joint_streams
    cam_ts = np.delete(cam.timestamps, 40)        # 掉一帧
    cam2 = TopicStream("/cam/image", cam_ts, cam.values[: len(cam_ts)], kind="image")
    frames, report = align_streams([cam2, joint], reference_topic="/cam/image", tolerance=0.020)
    assert len(frames) == 89
    assert report.stats["/joint_states"].matched == 89


def test_tolerance_marks_missing():
    t0 = 1.0
    ref = TopicStream("/ref", np.array([t0 + i * 0.1 for i in range(10)]), list(range(10)))
    # 稀疏流：中间空 0.5s
    sparse_ts = np.concatenate([np.array([t0 + i * 0.1 for i in range(3)]),
                                np.array([t0 + 0.8, t0 + 0.9])])
    sparse = TopicStream("/sparse", sparse_ts, list(range(len(sparse_ts))))
    frames, report = align_streams([ref, sparse], reference_topic="/ref",
                                   tolerance=0.020, drop_incomplete=False)
    assert report.stats["/sparse"].missing == 5   # 0.3~0.7 五帧无样本
    missing_frames = [f.frame_index for f in frames if "/sparse" in f.missing]
    assert missing_frames == [3, 4, 5, 6, 7]


def test_latest_before_strategy():
    t0 = 1.0
    ref = TopicStream("/ref", np.array([t0 + 0.105]), ["x"])
    slow = TopicStream("/slow", np.array([t0 + 0.10, t0 + 0.20]), ["a", "b"])
    frames, _ = align_streams([ref, slow], reference_topic="/ref",
                              strategy="latest_before", tolerance=0.5)
    assert frames[0].samples["/slow"] == "a"      # 不晚于参考时刻的最近样本


def test_target_fps_resample(cam_joint_streams):
    cam, joint = cam_joint_streams
    frames, report = align_streams([cam, joint], target_fps=20.0, tolerance=0.030)
    assert report.reference == "resample@20.0Hz"
    assert 55 <= len(frames) <= 60                # 3 秒 @ 20Hz
    ts = [f.timestamp for f in frames]
    assert np.allclose(np.diff(ts), 0.05, atol=1e-9)


def test_invalid_config_raises(cam_joint_streams):
    cam, joint = cam_joint_streams
    with pytest.raises(ValueError):
        align_streams([cam, joint])                                    # 两个都没给
    with pytest.raises(ValueError):
        align_streams([cam, joint], reference_topic="/cam/image", target_fps=30.0)  # 二选一
    with pytest.raises(ValueError):
        align_streams([cam, joint], reference_topic="/nonexistent")


def test_default_tolerance_50ms_merge_asof(cam_joint_streams):
    """默认 50ms 容差窗 + pandas merge_asof 实现，超窗丢帧。"""
    cam, joint = cam_joint_streams
    # 不传 tolerance：默认 0.050（50ms），500Hz 关节流（周期 2ms）全部命中
    frames, report = align_streams([cam, joint], reference_topic="/cam/image")
    assert report.stats["/joint_states"].matched == len(frames)

    # 关节流中间掉帧 200ms（1.0s~1.2s 无数据，模拟真实采集丢包）：
    # 该窗口内的相机帧最近关节样本距离 ~100ms，超 50ms 窗 -> 判缺失并丢帧
    import numpy as np
    from robotloop.ingest.align import TopicStream

    mask = (joint.timestamps < 1_700_000_001.0) | (joint.timestamps > 1_700_000_001.2)
    sparse = TopicStream(
        name="/joint_states",
        timestamps=joint.timestamps[mask],
        values=[v for v, m in zip(joint.values, mask) if m],
    )
    frames2, report2 = align_streams([cam, sparse], reference_topic="/cam/image")
    stat = report2.stats["/joint_states"]
    assert stat.missing > 0
    assert report2.dropped_frames == stat.missing
    assert len(frames2) + report2.dropped_frames == len(cam.timestamps)
    # 掉帧窗口内的最大偏差应显著超过 50ms 窗（否则不会判缺失）
    in_gap = (cam.timestamps > 1_700_000_001.0 + 0.05) & (cam.timestamps < 1_700_000_001.2 - 0.05)
    assert in_gap.sum() > 0
    assert in_gap.sum() <= stat.missing
