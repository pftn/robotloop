"""质量工具集测试：失败过滤 / 频率异常 / 相似去重 / 分布统计。"""

import numpy as np
import pytest

from robotloop.quality.dashboard import render_html_report, task_distribution
from robotloop.quality.dedup import dedup_by_similarity
from robotloop.quality.failure_filter import filter_episodes
from robotloop.quality.freq_anomaly import detect_freq_anomalies
from robotloop.schema.iceberg import episodes_to_arrow


class TestFailureFilter:
    def test_success_filter(self, make_episode):
        eps = [make_episode(idx=i, n=10, success=s) for i, s in enumerate([True, False, None])]
        r = filter_episodes(eps, keep_unlabeled=False)
        assert len(r.kept) == 1
        assert len(r.removed) == 2
        assert "success=false" in r.removed[0]["reasons"]
        assert any("未标注" in x for x in r.removed[1]["reasons"])

    def test_truncation_and_duration(self, make_episode):
        short = make_episode(idx=0, n=2, success=True)
        short.duration = 0.1
        ok = make_episode(idx=1, n=20, success=True)
        r = filter_episodes([short, ok], min_frames=5, keep_unlabeled=True)
        assert [e.episode_id for e in r.kept] == ["ep0001"]
        reasons = r.removed[0]["reasons"]
        assert any("帧数过少" in x for x in reasons)
        assert any("时长过短" in x for x in reasons)

    def test_zero_action_detected(self, make_episode):
        ep = make_episode(idx=0, n=10, success=True)
        for s in ep.steps:
            s.action = [0.0] * 7
        r = filter_episodes([ep], keep_unlabeled=True)
        assert len(r.removed) == 1
        assert any("全零" in x for x in r.removed[0]["reasons"])

    def test_reason_breakdown(self, make_episode):
        eps = [make_episode(idx=i, n=10, success=False) for i in range(3)]
        r = filter_episodes(eps)
        assert r.summary["reason_breakdown"]["success=false"] == 3


class TestFreqAnomaly:
    def test_gap_detected(self):
        ts = np.concatenate([np.arange(0, 1.0, 1 / 30), np.arange(1.2, 2.0, 1 / 30)])
        rep = detect_freq_anomalies(ts, topic="/cam", expected_hz=30.0)
        kinds = [a.kind for a in rep.anomalies]
        assert "gap" in kinds
        assert rep.max_gap_ms == pytest.approx(233.3, abs=0.5)

    def test_drift_detected(self):
        ts = np.arange(0, 10.0, 1 / 22)   # 标称 30Hz 的相机跑成 22Hz
        rep = detect_freq_anomalies(ts, topic="/cam", expected_hz=30.0)
        assert any(a.kind == "drift" for a in rep.anomalies)

    def test_clean_stream_no_anomaly(self):
        ts = np.arange(0, 5.0, 1 / 500) + np.random.default_rng(1).normal(0, 0.0001, 2500)
        rep = detect_freq_anomalies(ts, topic="/joints", expected_hz=500.0)
        assert rep.anomalies == []
        assert rep.nominal_hz == pytest.approx(500.0, rel=0.01)


class TestDedup:
    def test_exact_duplicate(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        w = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ids, vecs = ["a", "b", "c"], np.stack([v, v * 1.0001, w])
        r = dedup_by_similarity(ids, vecs, threshold=0.999)
        assert sorted(r.kept_ids) == ["a", "c"]
        assert r.removed_ids == ["b"]
        assert r.duplicate_clusters == [["a", "b"]]

    def test_no_duplicates(self):
        rng = np.random.default_rng(0)
        vecs = rng.normal(size=(10, 16)).astype(np.float32)
        r = dedup_by_similarity([f"e{i}" for i in range(10)], vecs, threshold=0.999)
        assert len(r.kept_ids) == 10
        assert r.duplicate_clusters == []

    def test_cluster_transitivity(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ids = ["a", "b", "c", "d"]
        vecs = np.stack([v, v * 1.0001, v * 1.0002, np.array([0.0, 1.0, 0.0])])
        r = dedup_by_similarity(ids, vecs, threshold=0.999)
        assert r.kept_ids == ["a", "d"]
        assert sorted(r.removed_ids) == ["b", "c"]


class TestDashboard:
    def test_task_distribution(self, make_episode):
        eps = [make_episode(idx=i, n=10, task="pick" if i < 3 else "place",
                            embodiment="franka" if i < 4 else "widowx",
                            success=(i % 2 == 0)) for i in range(6)]
        tbl = episodes_to_arrow([e.meta_dict() for e in eps])
        stats = task_distribution(tbl)
        assert stats["total_episodes"] == 6
        assert stats["by_task"] == {"pick": 3, "place": 3}
        assert stats["by_embodiment"] == {"franka": 4, "widowx": 2}
        assert stats["success"]["true"] == 3

    def test_html_report(self, make_episode, tmp_path):
        eps = [make_episode(idx=i, n=10) for i in range(3)]
        stats = task_distribution(episodes_to_arrow([e.meta_dict() for e in eps]))
        out = str(tmp_path / "report.html")
        render_html_report(stats, out)
        content = open(out, encoding="utf-8").read()
        assert "pick_red_cube" in content and "Episodes" in content
