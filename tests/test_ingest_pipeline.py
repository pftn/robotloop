"""MCAP/rosbag2 灌库链路测试（JSONL 模拟源 + GR00T 衔接产物）。"""

import base64
import json
import os

import numpy as np
import pytest

from robotloop.export.gr00t import render_finetune_script, write_modality_json
from robotloop.pipeline import bag_to_episode


@pytest.fixture
def mock_bag(tmp_path):
    """造一份 30Hz 相机 + 500Hz 关节流的 JSONL 模拟日志。"""
    t0 = 1_700_000_000.0
    lines = []
    jpg = base64.b64encode(b"\xff\xd8\xff\xe0fakejpeg").decode()
    for i in range(60):  # 2s @ 30Hz
        lines.append(
            {
                "topic": "/cam/image_raw",
                "timestamp": t0 + i / 30,
                "message": {"data": jpg},
            }
        )
    for i in range(1000):  # 2s @ 500Hz
        lines.append(
            {
                "topic": "/joint_states",
                "timestamp": t0 + i / 500,
                "message": {"position": [0.001 * i] * 7},
            }
        )
    path = tmp_path / "demo.jsonl"
    with open(path, "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    return str(path)


def test_bag_to_episode_full_chain(mock_bag, tmp_path):
    ep = bag_to_episode(
        mock_bag,
        camera_topics=["/cam/image_raw"],
        state_topic="/joint_states",
        task="pick_red_cube",
        language_instruction="抓取红色方块",
        embodiment_tag="franka",
        success=True,
        image_dir=str(tmp_path / "imgs"),
    )
    assert ep.num_frames == 60  # 以相机为参考轴
    assert ep.success is True
    assert ep.metadata["alignment_report"]["stats"]["/joint_states"]["matched"] == 60
    # 图像落盘，且 observation["images"] 存的是完整路径（sink 端要靠它找到文件上传）
    assert len(os.listdir(tmp_path / "imgs" / "cam_image_raw")) == 60
    img_ref = ep.steps[0].observation["images"]["/cam/image_raw"]
    assert os.path.isabs(img_ref) and os.path.exists(img_ref)
    # action 来自 state 差分：非首末帧非零，末帧为零
    assert any(abs(a) > 0 for a in ep.steps[10].action)
    assert all(a == 0.0 for a in ep.steps[-1].action)
    assert ep.steps[-1].is_terminal


def test_stage_images_to_s3(tmp_path):
    """入库端图像持久化：本地帧图上传 MinIO，image_paths 就地重写为 s3:// URI；
    文件缺失保留原值不炸（导出端会给带上下文的报错）。"""
    from robotloop.schema.episode import DataSource, Episode, Step
    from robotloop.schema.sink import stage_images_to_s3

    # 造两帧本地图像 + 一帧指向不存在文件
    img1 = tmp_path / "f1.jpg"
    img1.write_bytes(b"\xff\xd8\xff\xe0fake1")
    img2 = tmp_path / "f2.jpg"
    img2.write_bytes(b"\xff\xd8\xff\xe0fake2")
    steps = [
        Step(
            frame_index=0,
            timestamp=0.0,
            observation={"images": {"/cam/image": str(img1)}, "state": [0.1]},
            action=[0.1],
        ),
        Step(
            frame_index=1,
            timestamp=0.1,
            observation={"images": {"/cam/image": str(img2)}, "state": [0.2]},
            action=[0.2],
        ),
        Step(
            frame_index=2,
            timestamp=0.2,
            observation={
                "images": {"/cam/image": str(tmp_path / "gone.jpg")},
                "state": [0.3],
            },
            action=[0.3],
        ),
    ]
    ep = Episode(
        task="t",
        language_instruction="t",
        embodiment_tag="aloha",
        source=DataSource.TELEOP,
        success=True,
        episode_id="ep_img",
        duration=0.3,
        fps=10.0,
        steps=steps,
    ).validate()

    class _FakeS3:
        def __init__(self):
            self.calls = []

        def put_object(self, Bucket, Key, Body):
            self.calls.append((Bucket, Key, Body))

    s3 = _FakeS3()
    n = stage_images_to_s3(ep, s3, "robotloop-data")
    assert n == 2 and len(s3.calls) == 2
    refs = [s.observation["images"]["/cam/image"] for s in ep.steps]
    assert refs[0] == "s3://robotloop-data/frames/ep_img/images/cam_image/f1.jpg"
    assert refs[1].startswith("s3://robotloop-data/frames/ep_img/images/cam_image/")
    assert refs[2].endswith("gone.jpg") and not refs[2].startswith(
        "s3://"
    )  # 缺失保留原值
    # 幂等：已是 s3:// 的不再上传
    n2 = stage_images_to_s3(ep, s3, "robotloop-data")
    assert n2 == 0 and len(s3.calls) == 2


def test_missing_topic_raises(mock_bag):
    with pytest.raises(ValueError, match="缺少必需 topic"):
        bag_to_episode(
            mock_bag,
            camera_topics=["/cam/image_raw"],
            state_topic="/nonexistent",
            task="t",
            language_instruction="t",
            embodiment_tag="franka",
        )


def test_modality_json(tmp_path):
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    path = write_modality_json(
        str(tmp_path / "ds"),
        state_groups={"arm": (0, 6), "gripper": (6, 7)},
        action_groups={"arm": (0, 6), "gripper": (6, 7)},
        video_keys=["observation.images.ego_view"],
    )
    m = json.load(open(path))
    assert m["state"]["arm"] == {"start": 0, "end": 6}
    assert m["action"]["gripper"] == {"start": 6, "end": 7}
    assert "observation.images.ego_view" in m["video"]
    assert "human.task_description" in m["annotation"]


def test_finetune_script_render(tmp_path):
    out = str(tmp_path / "run.sh")
    script = render_finetune_script("./ft_data", max_steps=5000, out_path=out)
    assert os.path.exists(out)
    assert os.access(out, os.X_OK)
    assert "gr00t_finetune.py" in script
    assert "--max-steps 5000" in script
    assert "libero" in script


def test_real_mcap_end_to_end(tmp_path):
    """真实 MCAP 端到端：make_sample_bag 生成 -> bag_to_episode 解析自洽。

    读取端走 make_reader + mcap_ros2 DecoderFactory 官方解码路径。
    """
    import pytest
    import sys
    import os

    pytest.importorskip("mcap")
    pytest.importorskip("mcap_ros2")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "device"))
    from make_sample_bag import make_sample_bag

    from robotloop.pipeline import bag_to_episode

    bag = str(tmp_path / "sample.mcap")
    stats = make_sample_bag(bag, duration=4.0)
    assert stats["camera_frames"] == 120 and stats["joint_samples"] == 2000

    ep = bag_to_episode(
        bag,
        camera_topics=["/top"],
        state_topic="/joint_states",
        task="pick_red_cube",
        language_instruction="pick up the red cube",
        embodiment_tag="aloha",
        tolerance=0.050,
    )
    assert ep.num_frames == 120
    assert abs(ep.fps - 30.0) < 1.0
    assert len(ep.steps[0].action) == 14
    assert abs(ep.duration - 4.0) < 0.2
    # 关节 diff 推导的 action 非全零（质检"动作全零"规则不应误伤）
    assert any(abs(a) > 1e-8 for a in ep.steps[1].action)


def test_real_mcap_to_training_dataset(tmp_path):
    """核心叙事链路端到端：MCAP 造包 → 解析 → 灌库 → 导出 v2.1 训练数据集
    （14 维 action + 图像 video 特征 + mp4 帧数一致 —— ACT validate_features 能过）。"""
    import json
    import os
    import sys

    import pytest

    pytest.importorskip("mcap")
    pytest.importorskip("mcap_ros2")
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "device"))
    from make_sample_bag import make_sample_bag

    from robotloop.datasets.load import ingest_episodes
    from robotloop.export.lerobot_export import export_to_lerobot
    from robotloop.pipeline import bag_to_episode
    from robotloop.retrieval.store import LocalStore

    bag = str(tmp_path / "sample.mcap")
    make_sample_bag(bag, duration=2.0)
    ep = bag_to_episode(
        bag,
        camera_topics=["/top"],
        state_topic="/joint_states",
        task="pick_red_cube",
        language_instruction="pick up the red cube",
        embodiment_tag="aloha",
        tolerance=0.050,
        image_dir=str(tmp_path / "imgs"),
    )
    store = LocalStore(str(tmp_path / "lake"))
    ingest_episodes([ep], store)

    out = str(tmp_path / "ft")
    info = export_to_lerobot(store, out, fps=30.0)
    assert info["features"]["action"]["shape"] == [14]
    vk = "observation.images.top"
    assert info["features"][vk]["dtype"] == "video"
    assert info["features"][vk]["shape"] == [480, 640, 3]
    mp4 = os.path.join(out, "videos", "chunk-000", vk, "episode_000000.mp4")
    assert os.path.exists(mp4)
    reader = imageio.get_reader(mp4)
    try:
        assert reader.count_frames() == ep.num_frames
    finally:
        reader.close()
    # 相机特征的归一化统计（lerobot make_dataset 的 KeyError 防线）
    ep_stats = json.loads(
        open(os.path.join(out, "meta", "episodes_stats.jsonl")).readline()
    )
    vs = ep_stats["stats"][vk]
    assert np.asarray(vs["mean"]).shape == (3, 1, 1)
    assert vs["count"] == [ep.num_frames]
