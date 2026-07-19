"""openx 数据集接入行为测试"""

import pytest

import robotloop.datasets.openx as openx


def test_hardcoded_blocklist_removed():
    """数据集一律按名称指定，由 preflight 实测大小。"""
    assert not hasattr(openx, "BLOCKED_LARGE_SUBSETS")
    assert not hasattr(openx, "check_subset_allowed")
    assert not hasattr(
        openx, "RECOMMENDED_SMALL_SUBSETS"
    )  # 清单移到 config/presets.yaml


def test_embodiment_map_covers_preset_datasets():
    """presets.yaml 用到的 Open X 子集必须有本体映射。"""
    assert (
        openx.OPENX_EMBODIMENT_MAP["nyu_door_opening_surprising_effectiveness"]
        == "stretch"
    )
    assert openx.OPENX_EMBODIMENT_MAP["berkeley_fanuc_manipulation"] == "fanuc"
    assert openx.OPENX_EMBODIMENT_MAP["viola"] == "franka"
    assert openx.OPENX_EMBODIMENT_MAP["jaco_play"] == "jaco"


def test_embodiment_fallback_unknown(monkeypatch):
    """未映射子集 embodiment 落 unknown，不中断加载。"""
    called = {}

    class FakeEpisode:
        def __init__(self):
            self.dataset_name = ""
            self.embodiment_tag = ""

    def fake_load(tfds_name, **kwargs):
        called.update(kwargs)
        return [FakeEpisode()]

    monkeypatch.setattr("robotloop.convert.rlds.load_tfds_rlds", fake_load)
    eps = openx.load_openx_subset("some_new_subset/0.1.0")
    assert called["embodiment_tag"] == "unknown"
    assert eps[0].dataset_name == "openx/some_new_subset"
