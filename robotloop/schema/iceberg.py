"""Iceberg 表结构定义 —— Episode 领域模型的物理落地。

核心设计约束：**Iceberg 严禁存帧级行**。

因此只有一张表：

- ``robotloop.episodes``  轨迹级元数据（检索/过滤/统计走这张表）

帧数据一个 episode 一个 Parquet 文件存 MinIO（LeRobot v3 文件块思路），
文件路径写进 episodes 表的 ``parquet_path`` 列。训练导出时按 parquet_path
直接读文件，不再扫 Iceberg。

设计取舍：
- episodes 按 ``embodiment_tag`` 做 identity 分区 —— "找所有 ALOHA 的轨迹"
  这类查询是最高频过滤条件，分区裁剪直接生效；success/source/task 走列存
  谓词下推。
- action/state 维度随机器人型号变化：帧 parquet 里用嵌套数组（list<double>）
  存向量，维度语义学 GR00T modality.json 的做法放元数据（export/gr00t.py），
  不写死 schema。
- 向量（语言指令 embedding、轨迹 embedding）不进 Iceberg，进 LanceDB/Milvus
  （沿用 RobotLoop 既有特征存储分层：Iceberg 管结构化，向量库管 ANN）。
- catalog 用 v1 既有的 REST catalog（iceberg-rest:8181），warehouse 指向
  MinIO 的 iceberg-warehouse bucket。

pyiceberg 为可选依赖：未安装时仍可用 ``EPISODES_PA_SCHEMA`` 等 pyarrow
schema 做本地开发与单元测试。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pyarrow as pa

EPISODES_TABLE = "robotloop.episodes"
ICEBERG_WAREHOUSE = "s3://iceberg-warehouse"

# ---------------------------------------------------------------------------
# pyarrow schema（单一事实来源；pyiceberg schema 由它转换而来）
# ---------------------------------------------------------------------------
# 8 个核心字段：
#   episode_id / embodiment_tag / task / language_instruction /
#   success / source(teleop|sim|real) / duration / parquet_path
# 其余字段（dataset_name/episode_index/fps/num_frames/robot_type/created_at）
# 是 LeRobot v2.1/v3.0 超集的一部分，服务于血缘追溯与导出。
EPISODES_PA_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("dataset_name", pa.string()),
        pa.field("episode_index", pa.int64()),
        pa.field("task", pa.string(), nullable=False),
        pa.field("language_instruction", pa.string()),
        pa.field("embodiment_tag", pa.string(), nullable=False),
        pa.field("source", pa.string()),  # teleop | sim | real
        pa.field("success", pa.bool_()),  # NULL = 未标注
        pa.field("duration", pa.float64()),  # 秒
        pa.field("fps", pa.float64()),
        pa.field("num_frames", pa.int64()),
        pa.field("robot_type", pa.string()),
        # 帧数据 Parquet 在 MinIO 里的路径（s3://robotloop-data/frames/xxx.parquet）
        # Iceberg 只存这个指针，不存帧级行。
        pa.field("parquet_path", pa.string()),
        pa.field("created_at", pa.timestamp("us")),
    ]
)

# 帧级 Parquet 文件的 schema —— 仅用于 MinIO 对象存储里的
# frames/{episode_id}.parquet，绝不作为 Iceberg 表使用。
FRAME_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64()),
        pa.field("action", pa.list_(pa.float64())),
        pa.field("state", pa.list_(pa.float64())),
        pa.field("image_paths", pa.map_(pa.string(), pa.string())),
        pa.field("reward", pa.float64()),
        pa.field("is_terminal", pa.bool_()),
        pa.field("is_first", pa.bool_()),
        pa.field("is_last", pa.bool_()),
        pa.field("language_instruction", pa.string()),
    ]
)


def _pyiceberg_schema(pa_schema: pa.Schema):
    """由 pyarrow schema 生成 pyiceberg Schema（惰性导入 pyiceberg）。"""
    try:
        from pyiceberg.schema import Schema
        from pyiceberg.types import (
            BooleanType,
            DoubleType,
            ListType,
            LongType,
            MapType,
            NestedField,
            StringType,
            TimestampType,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "需要 pyiceberg 才能创建 Iceberg 表: pip install 'pyiceberg[pyarrow]'"
        ) from e

    def conv(f: pa.Field, fid: int) -> NestedField:
        t = f.type
        if pa.types.is_string(t):
            it = StringType()
        elif pa.types.is_int64(t):
            it = LongType()
        elif pa.types.is_float64(t):
            it = DoubleType()
        elif pa.types.is_boolean(t):
            it = BooleanType()
        elif pa.types.is_timestamp(t):
            it = TimestampType()
        elif pa.types.is_list(t):
            it = ListType(
                element_id=fid * 100 + 1,
                element_type=DoubleType(),
                element_required=False,
            )
        elif pa.types.is_map(t):
            it = MapType(
                key_id=fid * 100 + 1,
                key_type=StringType(),
                value_id=fid * 100 + 2,
                value_type=StringType(),
                value_required=False,
            )
        else:  # pragma: no cover
            raise TypeError(f"未映射的 Arrow 类型: {t}")
        return NestedField(
            field_id=fid, name=f.name, field_type=it, required=not f.nullable
        )

    return Schema(*[conv(f, i + 1) for i, f in enumerate(pa_schema)])


def episodes_iceberg_schema():
    return _pyiceberg_schema(EPISODES_PA_SCHEMA)


def create_tables(catalog, namespace: str = "robotloop"):
    """在给定 catalog 上建 episodes 表（幂等）。返回 episodes_table。

    只建元数据表，没有也不得有帧级 Iceberg 表。
    """
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    try:
        catalog.create_namespace(namespace)
    except Exception:
        pass

    # episodes：按本体分区 —— 结构化检索的第一过滤维度
    ep_spec = PartitionSpec(
        PartitionField(
            source_id=6, field_id=1000, transform=IdentityTransform(), name="embodiment"
        )
    )

    ident = f"{namespace}.episodes"
    try:
        return catalog.load_table(ident)
    except Exception:
        return catalog.create_table(
            identifier=ident,
            schema=_pyiceberg_schema(EPISODES_PA_SCHEMA),
            location=f"{ICEBERG_WAREHOUSE}/{ident.split('.')[-1]}",
            partition_spec=ep_spec,
        )


# ---------------------------------------------------------------------------
# 行数据 <-> Arrow
# ---------------------------------------------------------------------------
def episodes_to_arrow(episodes: Iterable[Dict[str, Any]]) -> pa.Table:
    """episodes 表的行（Episode.meta_dict() 的输出）→ Arrow。"""
    rows: List[Dict[str, Any]] = list(episodes)
    if not rows:
        return pa.table({f.name: pa.array([], type=f.type) for f in EPISODES_PA_SCHEMA})
    import datetime as _dt

    for r in rows:
        ts = r.get("created_at")
        if isinstance(ts, (int, float)):
            r["created_at"] = _dt.datetime.fromtimestamp(
                ts, tz=_dt.timezone.utc
            ).replace(tzinfo=None)
    return pa.Table.from_pylist(rows, schema=EPISODES_PA_SCHEMA)


def steps_to_arrow(steps: Iterable[Dict[str, Any]]) -> pa.Table:
    """帧级行（Episode.step_dicts() 的输出）→ Arrow。

    只用于写 MinIO 里的帧 Parquet 文件（frames/{episode_id}.parquet），
    不得写入任何 Iceberg 表。
    """
    rows: List[Dict[str, Any]] = list(steps)
    if not rows:
        return pa.table(
            {f.name: pa.array([], type=f.type) for f in FRAME_PARQUET_SCHEMA}
        )
    # map 列：pylist 里是 dict，需要转成 (key, value) 对列表
    for r in rows:
        ip = r.get("image_paths") or {}
        r["image_paths"] = list(ip.items()) if isinstance(ip, dict) else list(ip)
    return pa.Table.from_pylist(rows, schema=FRAME_PARQUET_SCHEMA)
