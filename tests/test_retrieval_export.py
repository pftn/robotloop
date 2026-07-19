"""混合检索 + 一键导出（闭环出口）测试。"""

import json
import os

import numpy as np
import pytest

from robotloop.datasets.demo import make_demo_episodes
from robotloop.datasets.load import ingest_episodes
from robotloop.export.lerobot_export import episodes_from_store, export_to_lerobot
from robotloop.retrieval.encoder import HashEncoder, encode_trajectory
from robotloop.retrieval.hybrid import HybridRetriever, parse_nl_query
from robotloop.retrieval.store import LocalStore


@pytest.fixture
def store(tmp_path):
    s = LocalStore(str(tmp_path / "lake"))
    eps = make_demo_episodes(n=60, seed=42)
    ingest_episodes(eps, s, encoder=HashEncoder())
    return s


class TestNLParse:
    def test_demo_query(self):
        residual, filters = parse_nl_query("找所有 ALOHA 双臂成功抓取红色方块的轨迹")
        assert filters == {"embodiment_tag": "aloha", "success": True}
        assert "抓取红色方块" in residual

    def test_failure_query(self):
        _, filters = parse_nl_query("widowx 失败的放碗轨迹")
        assert filters["embodiment_tag"] == "widowx"
        assert filters["success"] is False

    def test_source_query(self):
        _, filters = parse_nl_query("智元在仿真里按按钮")
        assert filters["embodiment_tag"] == "agibot_g1"
        assert filters["source"] == "sim"


class TestHybridSearch:
    def test_nl_hybrid_search(self, store):
        rt = HybridRetriever(store, encoder=HashEncoder())
        r = rt.search(
            text="找所有 ALOHA 双臂成功抓取红色方块的轨迹", top_k=5, nl_query=True
        )
        assert r["count"] > 0
        for x in r["results"]:
            assert x["embodiment_tag"] == "aloha"
            assert x["success"] is True
        # 语义排序：pick_red_cube 应排在最前
        assert r["results"][0]["task"] == "pick_red_cube"

    def test_pure_structured(self, store):
        rt = HybridRetriever(store, encoder=HashEncoder())
        r = rt.search(
            filters={"embodiment_tag": "agibot_g1", "source": "sim"}, top_k=10
        )
        assert r["count"] > 0
        for x in r["results"]:
            assert x["embodiment_tag"] == "agibot_g1"
            assert x["source"] == "sim"

    def test_empty_result(self, store):
        rt = HybridRetriever(store, encoder=HashEncoder())
        r = rt.search(filters={"embodiment_tag": "nonexistent_bot"}, top_k=5)
        assert r["count"] == 0


class TestIngestIdempotent:
    def test_reingest_dedups(self, tmp_path):
        s = LocalStore(str(tmp_path / "lake2"))
        eps = make_demo_episodes(n=10, seed=1)
        ingest_episodes(eps, s, encoder=HashEncoder())
        ingest_episodes(eps, s, encoder=HashEncoder())  # 同 id 再灌一遍
        assert s.count() == 10


class TestExport:
    def test_success_only_export(self, store, tmp_path):
        out = str(tmp_path / "ft")
        info = export_to_lerobot(
            store,
            out,
            filters={"embodiment_tag": "aloha", "source": "sim"},
            success_only=True,
            fps=10.0,
        )
        assert info["codebase_version"] == "v2.1"
        assert info["total_episodes"] > 0
        # 读回验证全部 success=True 且本体正确
        from robotloop.convert.lerobot_v21 import read_lerobot_v21

        back = read_lerobot_v21(out)
        assert all(e.success is True for e in back)
        assert all("aloha" in e.embodiment_tag for e in back)

    def test_export_steps_preserved(self, store, tmp_path):
        out = str(tmp_path / "ft30")
        info = export_to_lerobot(
            store,
            out,
            filters={"embodiment_tag": "agibot_g1", "source": "sim"},
            version="v3.0",
            fps=10.0,
        )
        from robotloop.convert.lerobot_v30 import read_lerobot_v30

        back = read_lerobot_v30(out)
        total_steps = sum(e.num_frames for e in back)
        assert info["total_frames"] == total_steps
        assert back[0].steps[0].action  # 帧级动作完整

    def test_export_with_images(self, store, tmp_path):
        """ACT validate_features 回归：导出数据集必须带 video 类型图像特征
        （observation.state 不算数），且 mp4 真实编码、帧数与 episode 一致。"""
        pytest.importorskip("imageio_ffmpeg")
        import imageio.v2 as imageio

        out = str(tmp_path / "ft_img")
        info = export_to_lerobot(
            store, out, filters={"embodiment_tag": "aloha", "source": "sim"}, fps=10.0
        )
        video_feats = {
            k: v for k, v in info["features"].items() if v.get("dtype") == "video"
        }
        assert (
            video_feats
        ), "导出数据集必须含图像特征（ACT validate_features 要求至少一路图像）"
        vk = "observation.images.top"  # topic /top 映射而来（对齐 gym_aloha 观测键）
        assert vk in video_feats
        assert video_feats[vk]["shape"] == [
            480,
            640,
            3,
        ]  # demo 合成图对齐 aloha env 渲染分辨率
        assert info["total_videos"] == info["total_episodes"] * len(video_feats)
        # 每个 episode 的 mp4 真实存在，且帧数 == episodes.jsonl 声明的 length
        lengths = {
            r["episode_index"]: r["length"]
            for r in (
                json.loads(l) for l in open(os.path.join(out, "meta", "episodes.jsonl"))
            )
        }
        for ep_idx, length in lengths.items():
            mp4 = os.path.join(
                out, "videos", "chunk-000", vk, f"episode_{ep_idx:06d}.mp4"
            )
            assert os.path.exists(mp4) and os.path.getsize(mp4) > 0
            reader = imageio.get_reader(mp4)
            try:
                assert reader.count_frames() == length
            finally:
                reader.close()
        # episodes_stats.jsonl 每行都带相机特征的归一化统计（lerobot make_dataset
        # 按相机 key 取 stats，缺失直接 KeyError: 'observation.images.*'；
        # 官方 _assert_type_and_shape 校验图像 stats 形状为 (3,1,1)）
        for r in (
            json.loads(l)
            for l in open(os.path.join(out, "meta", "episodes_stats.jsonl"))
        ):
            vs = r["stats"][vk]
            for f in ("min", "max", "mean", "std"):
                arr = np.asarray(vs[f], dtype=np.float64)
                assert arr.shape == (3, 1, 1)
                assert arr.min() >= 0.0 and arr.max() <= 1.0  # [0,1] 值域约定
            assert np.asarray(vs["mean"]).max() > 0.01  # 非纯黑图
            assert vs["count"] == [lengths[r["episode_index"]]]

    def test_export_camera_map_renames_keys(self, store, tmp_path):
        """camera_map 显式对齐仿真 env 相机键（真实数据任意 topic 名场景，
        否则 eval 时 policy 找不到训练时的相机输入键直接 KeyError）。"""
        pytest.importorskip("imageio_ffmpeg")
        out = str(tmp_path / "ft_cammap")
        info = export_to_lerobot(
            store,
            out,
            filters={"embodiment_tag": "aloha", "source": "sim"},
            fps=10.0,
            camera_map={"/top": "observation.images.wrist"},
        )
        assert "observation.images.wrist" in info["features"]
        assert "observation.images.top" not in info["features"]

    def test_export_episode_ids_subset(self, store, tmp_path):
        rt = HybridRetriever(store, encoder=HashEncoder())
        r = rt.search(filters={"embodiment_tag": "widowx", "success": True}, top_k=3)
        ids = [x["episode_id"] for x in r["results"]]
        if not ids:
            pytest.skip("demo 数据无 widowx 成功样本")
        out = str(tmp_path / "ft_ids")
        info = export_to_lerobot(store, out, episode_ids=ids, fps=10.0)
        assert info["total_episodes"] == len(ids)


def test_encode_trajectory_dim():
    v = encode_trajectory([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    assert v.shape == (8,)  # 4 × action_dim
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_open_store_iceberg_backend(monkeypatch):
    """CLI 的 iceberg:// 前缀分发到生产湖 store（本地路径仍 LocalStore）。"""
    import robotloop.cli as cli

    called = {}

    class _FakeStore:
        def __init__(self, episodes_table="robotloop.episodes"):
            called["table"] = episodes_table

    monkeypatch.setattr("robotloop.retrieval.store.MilvusIcebergStore", _FakeStore)
    s = cli._open_store("iceberg://robotloop.episodes")
    assert isinstance(s, _FakeStore) and called["table"] == "robotloop.episodes"
    cli._open_store("iceberg://")
    assert called["table"] == "robotloop.episodes"  # 缺省表名
    from robotloop.retrieval.store import LocalStore

    assert isinstance(cli._open_store("/tmp/x"), LocalStore)


def test_iceberg_row_filter_empty_is_none():
    """空过滤条件返回 None（不调 row_filter）——pyiceberg parser 只接受
    "列名 op 字面量"，"1=1" 常量表达式会 ParseException（容器内回归）。"""
    from robotloop.retrieval.store import _iceberg_row_filter

    assert _iceberg_row_filter({}) is None
    assert _iceberg_row_filter({"success": None}) is None


def test_iceberg_row_filter_parseable():
    """生成的所有子句形式必须能被 pyiceberg 表达式 parser 接受。"""
    pytest.importorskip("pyiceberg")
    from pyiceberg.expressions.parser import parse

    from robotloop.retrieval.store import _iceberg_row_filter

    cases = [
        {"embodiment_tag": "aloha"},
        {"source": "sim"},
        {"success": True},
        {"success": False},
        {"duration_min": 5, "duration_max": 60},
        {"num_frames_min": 10, "num_frames_max": 100},
        {"embodiment_tag": "aloha", "source": "sim", "success": True},
    ]
    for f in cases:
        rf = _iceberg_row_filter(f)
        assert rf is not None
        assert parse(rf) is not None


def test_milvus_iceberg_store_lazy_milvus(monkeypatch):
    """Milvus 连接惰性化：构造不 import pymilvus、不读 MILVUS_URI ——
    导出场景（只走 Iceberg+MinIO）不被向量库依赖卡住。"""
    import sys
    import types

    import robotloop.retrieval.store as st

    fake_catalog_mod = types.ModuleType("pyiceberg.catalog")
    fake_catalog_mod.load_catalog = lambda *a, **k: object()
    fake_pyiceberg = types.ModuleType("pyiceberg")
    fake_pyiceberg.catalog = fake_catalog_mod
    monkeypatch.setitem(sys.modules, "pyiceberg", fake_pyiceberg)
    monkeypatch.setitem(sys.modules, "pyiceberg.catalog", fake_catalog_mod)
    monkeypatch.setitem(sys.modules, "pymilvus", None)  # import 即 ImportError

    s = st.MilvusIcebergStore()  # 构造不炸 = 没碰 pymilvus / MILVUS_URI
    assert s._milvus_client is None
