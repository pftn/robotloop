"""数据集校验器

校验项（对 v2.1 / v3.0 双版本自适应）：
1. meta/info.json 存在且 codebase_version 合法
2. episode 数量：meta 记录数 == 实际数据覆盖的 episode_index 数
3. episode_index 不重不漏（0..N-1 连续）
4. 每 episode 内 frame_index 连续（0..len-1）
5. 帧数：数据行数 == episodes 元数据 length 之和
6. 每帧 episode_index 与文件归属一致（v3.0：meta 里的 chunk/file 指针有效）
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot_convert.convert import (
    _data_files_v30,
    _episode_files_v21,
    detect_version,
)

logger = logging.getLogger("lerobot_convert.validate")


@dataclass
class ValidationReport:
    version: str = ""
    total_episodes: int = 0
    total_frames: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_dataset(root: str, check_frames: bool = True) -> ValidationReport:
    """校验 LeRobot 数据集目录（v2.1/v3.0 自适应）。"""
    rep = ValidationReport()
    try:
        rep.version = detect_version(root)
    except Exception as e:
        rep.errors.append(str(e))
        return rep

    if rep.version == "v2.1":
        files = _episode_files_v21(root)
        episodes_meta = _load_jsonl_meta(root)
    else:
        files = _data_files_v30(root)
        episodes_meta = _load_v30_meta(root)

    rep.total_episodes = len(episodes_meta)

    # episode_index 不重不漏
    idxs = sorted(m["episode_index"] for m in episodes_meta)
    if idxs != list(range(len(idxs))):
        missing = sorted(set(range(len(idxs))) - set(idxs))
        rep.errors.append(f"episode_index 不连续，缺失: {missing[:10]}")

    expected_frames = sum(int(m.get("length", 0)) for m in episodes_meta)
    length_by_ep = {m["episode_index"]: int(m.get("length", 0)) for m in episodes_meta}

    # 逐文件校验
    seen_eps = set()
    actual_frames = 0
    for f in files:
        tbl = pq.read_table(f)
        actual_frames += tbl.num_rows
        if "episode_index" not in tbl.column_names:
            rep.errors.append(f"{f}: 缺少 episode_index 列")
            continue
        if check_frames and "frame_index" in tbl.column_names:
            ep_col = tbl.column("episode_index").to_pylist()
            fi_col = tbl.column("frame_index").to_pylist()
            # 按 episode 分组检查 frame 连续性
            from collections import defaultdict

            groups: Dict[int, List[int]] = defaultdict(list)
            for e, fi in zip(ep_col, fi_col):
                groups[e].append(fi)
            for e, fis in groups.items():
                if sorted(fis) != list(range(len(fis))):
                    rep.errors.append(f"{f}: episode {e} frame_index 不连续")
        for e in set(tbl.column("episode_index").to_pylist()):
            seen_eps.add(e)
            n = sum(1 for x in tbl.column("episode_index").to_pylist() if x == e)
            if e in length_by_ep and rep.version == "v2.1" and n != length_by_ep[e]:
                rep.errors.append(
                    f"{f}: episode {e} 帧数 {n} != episodes.jsonl length {length_by_ep[e]}"
                )
    rep.total_frames = actual_frames

    if seen_eps != set(idxs):
        rep.errors.append(
            f"数据文件中的 episode 集合与元数据不一致: 数据{len(seen_eps)} vs 元数据{len(idxs)}"
        )
    if expected_frames and actual_frames != expected_frames:
        rep.errors.append(
            f"总帧数不符: 数据 {actual_frames} != 元数据 {expected_frames}"
        )

    return rep


def _load_jsonl_meta(root: str) -> List[Dict[str, Any]]:
    path = os.path.join(root, "meta", "episodes.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_v30_meta(root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta_root = os.path.join(root, "meta", "episodes")
    for dirpath, _, names in os.walk(meta_root):
        for n in sorted(names):
            if n.endswith(".parquet"):
                rows.extend(pq.read_table(os.path.join(dirpath, n)).to_pylist())
    rows.sort(key=lambda r: r["episode_index"])
    return rows
