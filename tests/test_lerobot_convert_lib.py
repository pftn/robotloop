"""lerobot-convert 独立库测试（批量互转 + 校验）。"""

import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lerobot-convert", "src"))

from lerobot_convert import (  # noqa: E402
    convert_v21_to_v30,
    convert_v30_to_v21,
    detect_version,
    validate_dataset,
)


@pytest.fixture()
def v21_dataset(tmp_path, make_episode):
    from robotloop.convert.lerobot_v21 import write_lerobot_v21

    eps = [make_episode(idx=i, n=10, success=True) for i in range(6)]
    root = str(tmp_path / "ds_v21")
    write_lerobot_v21(eps, root, fps=30.0)
    return root


def test_detect_version(v21_dataset):
    assert detect_version(v21_dataset) == "v2.1"


def test_v21_to_v30_roundtrip(v21_dataset, tmp_path):
    v30 = str(tmp_path / "ds_v30")
    convert_v21_to_v30(v21_dataset, v30)
    assert detect_version(v30) == "v3.0"
    rep = validate_dataset(v30)
    assert rep.ok, rep.errors
    assert rep.total_episodes == 6
    assert rep.total_frames == 60

    # 转回 v2.1
    back = str(tmp_path / "ds_v21_back")
    convert_v30_to_v21(v30, back)
    rep2 = validate_dataset(back)
    assert rep2.ok, rep2.errors
    assert rep2.total_episodes == 6
    assert rep2.total_frames == 60

    # 帧数据无损
    a = pq.read_table(os.path.join(v21_dataset, "data/chunk-000/episode_000000.parquet"))
    b = pq.read_table(os.path.join(back, "data/chunk-000/episode_000000.parquet"))
    assert a.num_rows == b.num_rows
    assert np.allclose(
        np.array(a.column("action").to_pylist(), dtype=float),
        np.array(b.column("action").to_pylist(), dtype=float),
    )


def test_validate_catches_missing_episode(v21_dataset, tmp_path):
    # 删掉一个 episode 文件制造不一致
    os.remove(os.path.join(v21_dataset, "data/chunk-000/episode_000003.parquet"))
    rep = validate_dataset(v21_dataset)
    assert not rep.ok
    assert any("episode" in e for e in rep.errors)


def test_file_rotation_respects_size_limit(v21_dataset, tmp_path):
    # 极小文件上限 -> 多文件轮转
    v30 = str(tmp_path / "ds_v30_rot")
    convert_v21_to_v30(v21_dataset, v30, data_file_size_mb=0)  # 0MB -> 每 episode 一个文件
    data_files = []
    for dirpath, _, names in os.walk(os.path.join(v30, "data")):
        data_files.extend(n for n in names if n.endswith(".parquet"))
    assert len(data_files) >= 2
    rep = validate_dataset(v30)
    assert rep.ok, rep.errors
