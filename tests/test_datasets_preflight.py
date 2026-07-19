"""size preflight 与 presets.yaml 加载测试（HTTP 全部 mock，不触网）。"""

import io
import json
import os
import sys
import urllib.request

import pytest

from robotloop.datasets.lerobot_hub import preflight_lerobot
from robotloop.datasets.openx import preflight_openx


def _mock_urlopen(payloads):
    """按 URL 子串路由返回预设 JSON 的 urlopen 替身。"""
    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(url, timeout=0):
        for key, payload in payloads.items():
            if key in url:
                return FakeResp(json.dumps(payload).encode())
        raise urllib.error.URLError(f"no mock for {url}")

    return fake


# ---------------------------------------------------------------- Open X
def test_preflight_openx_sums_sizes_and_episodes(monkeypatch):
    payloads = {
        # GCS 对象列表（两个对象共 1.5GB）
        "prefix=datasets/viola/0.1.0/": {
            "items": [{"size": "1000000000"}, {"size": "500000000"}],
        },
        # dataset_info 元数据
        "dataset_info/viola/0.1.0/dataset_info.json": {
            "splits": [{"name": "train", "shardLengths": ["80", "55"]}],
        },
    }
    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen(payloads))
    info = preflight_openx("viola/0.1.0")
    assert info["available"] is True
    assert info["size_gb"] == 1.5
    assert info["episodes"] == 135


def test_preflight_openx_unavailable_is_nonfatal(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen({}))
    info = preflight_openx("nonexistent_dataset/0.1.0")
    assert info["available"] is False
    assert info["size_gb"] is None
    assert info["episodes"] is None  # 不抛异常，由调用方决定


def test_preflight_openx_falls_back_to_gresearch(monkeypatch):
    """tfds-data 无对象时回退 gresearch/robotics bucket（Open X 多数子集实际位置）。"""
    payloads = {
        # tfds-data 对象列表为空
        "b/tfds-data/o?prefix=datasets/big_ds/0.1.0/": {"items": []},
        # gresearch 有对象：11GB（超 10GB 确认线）
        "b/gresearch/o?prefix=robotics/big_ds/0.1.0/": {
            "items": [{"size": "11000000000"}],
        },
    }
    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen(payloads))
    info = preflight_openx("big_ds/0.1.0")
    assert info["available"] is True
    assert info["size_gb"] == 11.0


# ---------------------------------------------------------------- LeRobot
def test_preflight_lerobot_reads_api_and_info_json(monkeypatch):
    payloads = {
        "/api/datasets/lerobot/pusht": {
            "siblings": [{"size": 1_000_000_000}, {"size": 500_000_000}, {"size": None}],
        },
        "/datasets/lerobot/pusht/resolve/main/meta/info.json": {
            "codebase_version": "v2.1",
            "total_episodes": 206,
        },
    }
    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen(payloads))
    info = preflight_lerobot("lerobot/pusht")
    assert info["available"] is True
    assert info["size_gb"] == 1.5
    assert info["episodes"] == 206


# ---------------------------------------------------------------- presets.yaml
def test_presets_yaml_loadable_and_wellformed():
    import yaml

    path = os.path.join(os.path.dirname(__file__), "..", "config", "presets.yaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    presets = cfg["presets"]
    assert "demo" in presets and "ci" in presets
    for name, p in presets.items():
        assert p["max_episodes_per_source"] >= 1
        for d in p.get("lerobot", []):
            assert "/" in d["repo_id"]          # 合法 repo_id
        for d in p.get("openx", []):
            assert d["tfds_name"]                # 非空 tfds 名


def test_load_preset_unknown_exits():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from ingest_public_datasets import load_preset

    preset = load_preset("demo")
    assert [d["repo_id"] for d in preset["lerobot"]] == [
        "lerobot/pusht", "lerobot/aloha_sim_insertion_human"]
    assert [d["tfds_name"] for d in preset["openx"]] == [
        "nyu_door_opening_surprising_effectiveness", "berkeley_fanuc_manipulation"]
    with pytest.raises(SystemExit):
        load_preset("nonexistent")
