"""RLDS ↔ LeRobot v2.1 ↔ v3.0 互转测试（本地 parquet，无外部依赖）。"""

import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytest

from robotloop.convert.cli import detect_version
from robotloop.convert.lerobot_v21 import read_lerobot_v21, write_lerobot_v21
from robotloop.convert.lerobot_v30 import read_lerobot_v30, write_lerobot_v30
from robotloop.convert.rlds import episode_to_rlds_dict, rlds_episode_to_episode
from robotloop.schema.episode import DataSource


@pytest.fixture
def episodes(make_episode):
    return [make_episode(idx=i, n=10 + i, task="pick_red_cube" if i % 2 == 0 else "stack_blocks",
                         success=(i % 2 == 0)) for i in range(5)]


def test_v21_write_layout(episodes, tmp_path):
    root = str(tmp_path / "v21")
    info = write_lerobot_v21(episodes, root, fps=10.0)
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 5
    assert info["total_frames"] == sum(10 + i for i in range(5))
    assert info["total_tasks"] == 2
    assert os.path.exists(os.path.join(root, "data", "chunk-000", "episode_000000.parquet"))
    assert os.path.exists(os.path.join(root, "meta", "tasks.jsonl"))
    assert os.path.exists(os.path.join(root, "meta", "episodes_stats.jsonl"))
    assert detect_version(root) == "v2.1"


def test_v30_write_layout(episodes, tmp_path):
    root = str(tmp_path / "v30")
    info = write_lerobot_v30(episodes, root, fps=10.0)
    assert info["codebase_version"] == "v3.0"
    assert os.path.exists(os.path.join(root, "data", "chunk-000", "file-000.parquet"))
    assert os.path.exists(os.path.join(root, "meta", "tasks.parquet"))
    assert os.path.exists(os.path.join(root, "meta", "stats.json"))
    assert os.path.exists(os.path.join(root, "meta", "episodes", "chunk-000", "file-000.parquet"))
    assert detect_version(root) == "v3.0"


def test_v21_roundtrip_lossless(episodes, tmp_path):
    root = str(tmp_path / "v21")
    write_lerobot_v21(episodes, root, fps=10.0)
    back = read_lerobot_v21(root)
    assert len(back) == 5
    for orig, rt in zip(episodes, back):
        assert rt.episode_id == orig.episode_id
        assert rt.task == orig.task
        assert rt.success == orig.success
        assert rt.num_frames == orig.num_frames
        assert np.allclose(rt.steps[2].action, orig.steps[2].action, atol=1e-6)


def test_v30_roundtrip_lossless(episodes, tmp_path):
    root = str(tmp_path / "v30")
    write_lerobot_v30(episodes, root, fps=10.0)
    back = read_lerobot_v30(root)
    assert len(back) == 5
    for orig, rt in zip(episodes, back):
        assert rt.episode_id == orig.episode_id
        assert rt.success == orig.success
        assert rt.source == orig.source
        assert rt.num_frames == orig.num_frames
        assert np.allclose(rt.steps[-1].action, orig.steps[-1].action, atol=1e-6)


def test_v21_v30_cross_conversion(episodes, tmp_path):
    """v2.1 → Episode → v3.0 → Episode → v2.1 全程语义无损。"""
    v21a = str(tmp_path / "v21a")
    v30 = str(tmp_path / "v30")
    v21b = str(tmp_path / "v21b")
    write_lerobot_v21(episodes, v21a, fps=10.0)
    mid = read_lerobot_v21(v21a)
    write_lerobot_v30(mid, v30, fps=10.0)
    back = read_lerobot_v30(v30)
    write_lerobot_v21(back, v21b, fps=10.0)
    final = read_lerobot_v21(v21b)
    assert [e.episode_id for e in final] == [e.episode_id for e in episodes]
    assert [e.success for e in final] == [e.success for e in episodes]
    assert np.allclose(final[3].steps[4].action, episodes[3].steps[4].action, atol=1e-6)


def test_v30_file_rotation(episodes, tmp_path):
    """数据文件按目标大小轮转到 file-001。"""
    root = str(tmp_path / "v30rot")
    info = write_lerobot_v30(episodes, root, fps=10.0, data_file_size_mb=0)  # 0MB → 每条一个文件
    files = sorted(os.listdir(os.path.join(root, "data", "chunk-000")))
    assert files == [f"file-{i:03d}.parquet" for i in range(5)]
    back = read_lerobot_v30(root)
    assert len(back) == 5  # 多文件读取正确


def test_rlds_roundtrip(make_episode):
    ep = make_episode(idx=7, n=9, success=True)
    rd = episode_to_rlds_dict(ep)
    assert rd["episode_metadata"]["episode_id"] == ep.episode_id
    assert len(rd["steps"]) == 9
    assert rd["steps"][0]["is_first"] is True
    assert rd["steps"][-1]["is_terminal"] is True
    back = rlds_episode_to_episode(rd, embodiment_tag="franka", fps=10.0)
    assert back.num_frames == 9
    assert back.success is True
    assert back.language_instruction == ep.language_instruction


def test_rlds_success_from_final_reward():
    """Open X 风格：无显式 success 时用末帧 reward 近似。"""
    raw = {
        "episode_metadata": {},
        "steps": [
            {"observation": {"state": np.zeros(7)}, "action": np.zeros(7), "reward": 0.0},
            {"observation": {"state": np.ones(7)}, "action": np.ones(7), "reward": 1.0,
             "language_instruction": "pick the cube"},
        ],
    }
    ep = rlds_episode_to_episode(raw, embodiment_tag="franka", fps=10.0)
    assert ep.success is True
    assert ep.language_instruction == "pick the cube"
