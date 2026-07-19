"""Episode 领域模型 + Iceberg schema（pyarrow 层）测试。"""

import pyarrow as pa

from robotloop.schema.episode import DataSource, Episode, Step
from robotloop.schema.iceberg import (
    EPISODES_PA_SCHEMA,
    FRAME_PARQUET_SCHEMA,
    episodes_to_arrow,
    steps_to_arrow,
)
from robotloop.schema.mapping import CONCEPT_MAPPING, render_markdown


def test_episode_validate_ok(make_episode):
    ep = make_episode(idx=0, n=8)
    assert ep.num_frames == 8
    assert ep.action_dim == 7


def test_episode_rejects_frame_gap(make_episode):
    ep = make_episode(idx=1, n=5)
    ep.steps[3].frame_index = 99
    try:
        ep.validate()
        assert False, "应当报 frame_index 不连续"
    except ValueError as e:
        assert "frame_index" in str(e)


def test_source_enum_coercion(make_episode):
    ep = make_episode(idx=2, n=3)
    ep.source = "teleop"
    ep.validate()
    assert ep.source is DataSource.TELEOP


def test_meta_and_steps_arrow_roundtrip(make_episode):
    eps = [make_episode(idx=i, n=6, success=(i % 2 == 0)) for i in range(4)]
    meta_tbl = episodes_to_arrow([e.meta_dict() for e in eps])
    assert meta_tbl.schema == EPISODES_PA_SCHEMA
    assert meta_tbl.num_rows == 4
    assert meta_tbl.column("success").to_pylist() == [True, False, True, False]

    step_rows = [r for e in eps for r in e.step_dicts()]
    steps_tbl = steps_to_arrow(step_rows)
    assert steps_tbl.schema == FRAME_PARQUET_SCHEMA
    assert steps_tbl.num_rows == 24
    first = steps_tbl.to_pylist()[0]
    assert first["episode_id"] == "ep0000"
    assert len(first["action"]) == 7
    assert first["is_terminal"] is False
    assert first["is_first"] is True and first["is_last"] is False
    last = steps_tbl.to_pylist()[5]
    assert last["is_terminal"] is True
    assert last["is_last"] is True and last["is_first"] is False


def test_redline_no_frame_level_iceberg_table():
    """schema 层不得再定义任何帧级 Iceberg 表，帧数据只有 parquet 文件 schema。"""
    import robotloop.schema.iceberg as ice

    assert not hasattr(ice, "EPISODE_STEPS_PA_SCHEMA")
    assert not hasattr(ice, "EPISODE_STEPS_TABLE")
    # episodes 表必须带 parquet_path 指针列
    assert "parquet_path" in EPISODES_PA_SCHEMA.names


def test_frame_store_roundtrip(make_episode, tmp_path):
    """帧数据一个 episode 一个 Parquet，路径写进 episode.parquet_path。"""
    from robotloop.schema.frame_store import FrameStore

    fs = FrameStore(backend="local", base_dir=str(tmp_path))
    ep = make_episode(idx=0, n=6)
    path = fs.write_episode_frames(ep)
    assert ep.parquet_path == path
    assert path.endswith(f"frames/{ep.episode_id}.parquet")

    rows = fs.read_episode_steps(path)
    assert len(rows) == 6
    assert rows[0]["frame_index"] == 0
    assert rows[0]["is_first"] is True
    assert isinstance(rows[0]["image_paths"], dict)


def test_empty_arrow():
    assert episodes_to_arrow([]).num_rows == 0
    assert steps_to_arrow([]).num_rows == 0


def test_concept_mapping_covers_key_concepts():
    concepts = {row["concept"] for row in CONCEPT_MAPPING}
    for must in [
        "一条轨迹",
        "一帧",
        "任务/语言指令",
        "动作",
        "成功标记",
        "机器人本体",
        "格式版本",
    ]:
        assert must in concepts
    md = render_markdown()
    assert md.startswith("| 概念 |")
    assert "v2.1" in md and "v3.0" in md and "RLDS" in md


def test_check_consistent_dims_rejects_mixed_embodiments():
    """混本体导出前置拦截：清晰 ValueError 而非 pyarrow ArrowInvalid。"""
    import pytest
    from robotloop.datasets.demo import make_demo_episodes
    from robotloop.schema.episode import check_consistent_dims

    mixed = make_demo_episodes()[:10]  # 含 aloha(14 维) 与其他本体
    dims = {len(e.steps[0].action) for e in mixed}
    assert len(dims) > 1, "demo 前 10 条应混合多种动作维度"

    with pytest.raises(ValueError, match="embodiment_tag"):
        check_consistent_dims(mixed)

    # 单一本体放行
    aloha_only = [e for e in mixed if e.embodiment_tag == "aloha"]
    check_consistent_dims(aloha_only)


def test_write_lerobot_rejects_mixed_dims_early(tmp_path):
    """两个版本 writer 都在写出前拦截混维度，错误信息提示按本体过滤。"""
    import pyarrow as pa
    import pytest
    from robotloop.convert.lerobot_v21 import write_lerobot_v21
    from robotloop.convert.lerobot_v30 import write_lerobot_v30
    from robotloop.datasets.demo import make_demo_episodes

    mixed = make_demo_episodes()[:10]
    for writer, ver in [(write_lerobot_v21, "v21"), (write_lerobot_v30, "v30")]:
        with pytest.raises(ValueError, match="embodiment_tag"):
            writer(mixed, str(tmp_path / ver), fps=30.0)


def test_check_consistent_fps_rejects_mixed_rates():
    import pytest

    from robotloop.datasets.demo import make_demo_episodes
    from robotloop.schema.episode import check_consistent_fps

    eps = make_demo_episodes()[:6]
    fps_values = {round(e.fps, 1) for e in eps}
    assert len(fps_values) > 1, "demo 前 6 条本身混帧率（SIM 10Hz / TELEOP 30Hz）"
    with pytest.raises(ValueError, match="统一 fps"):
        check_consistent_fps(eps)

    same = [e for e in eps if abs(e.fps - 10.0) < 0.5]
    check_consistent_fps(same)  # 单一帧率放行

    near = list(same)
    near[0].fps = 10.2  # rtol=0.05 容差内，防 29.97 vs 30 式误伤
    check_consistent_fps(near)


def test_export_rejects_mixed_fps_early(tmp_path):
    import pytest

    from robotloop.convert.lerobot_v21 import write_lerobot_v21
    from robotloop.datasets.demo import make_demo_episodes

    mixed = make_demo_episodes()[:4]  # SIM 10Hz + TELEOP 30Hz 混合
    with pytest.raises(ValueError, match="统一 fps"):
        write_lerobot_v21(mixed, str(tmp_path / "ds"))
