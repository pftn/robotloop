"""Open X-Embodiment 子集加载（RLDS 格式）。

Open X 子集通过 TFDS 发布，加载链路：
    tfds.load(name) → RLDS dict → convert.rlds → Episode → ingest

依赖 tensorflow_datasets（重，惰性导入）。常见子集的本体映射内置，
灌库时 embodiment_tag 不再需要手工查表。

size preflight：TFDS 必须整包 prepare 到本地才能逐条迭代，不存在增量
下载，因此灌库前用 GCS 公开元数据估算下载大小与 episode 数
（``preflight_openx``），由调用方决定确认或放弃。
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Dict, List, Optional

from robotloop.schema.episode import DataSource, Episode

logger = logging.getLogger(__name__)

# Open X 常见子集 → 本体标签
OPENX_EMBODIMENT_MAP = {
    "bridge_v2": "widowx",
    "fractal20220817_data": "google_robot",
    "kuka": "kuka",
    "taco_play": "franka",
    "jaco_play": "jaco",
    "berkeley_cable_routing": "franka",
    "berkeley_fanuc_manipulation": "fanuc",
    "viola": "franka",
    "toto": "franka",
    "nyu_door_opening_surprising_effectiveness": "stretch",
    "cmu_stretch": "stretch",
    "droid": "franka",
    "libero_spatial_no_noops": "franka",
    "libero_object_no_noops": "franka",
    "libero_goal_no_noops": "franka",
    "libero_10_no_noops": "franka",
    "aloha_mobile": "aloha",
    "robomimic_ph": "franka",
    "language_table": "xarm",
}

_GCS_DATA = "https://storage.googleapis.com/tfds-data"
# Open X 数据文件的两个公开 bucket：tfds-data（TFDS 官方）与
# gresearch/robotics（Open X 发布桶，多数子集实际在这里）
_GCS_BUCKETS = ["tfds-data", "gresearch"]
_GCS_PREFIXES = {"tfds-data": "datasets", "gresearch": "robotics"}


def preflight_openx(tfds_name: str, timeout: float = 10.0) -> Dict[str, Any]:
    """下载前估算 Open X 数据集的大小与 episode 数（不触发 TFDS prepare）。

    数据源：GCS 公开 bucket tfds-data 的对象列表与 dataset_info 元数据。
    返回 {"name", "version", "size_gb", "episodes", "available"}；
    网络或元数据不可达时 size_gb/episodes 为 None 且 available=False，
    由调用方决定如何处理（本函数不阻断）。
    """
    parts = tfds_name.split("/")
    name = parts[0]
    version = parts[1] if len(parts) > 1 else None
    out: Dict[str, Any] = {
        "name": name, "version": version, "size_gb": None, "episodes": None, "available": False,
    }

    def _get_json(url: str) -> Optional[dict]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            logger.debug("preflight fetch failed: %s (%s)", url, e)
            return None

    # 未指定版本时推断最新版本：先试 tfds-data 的 dataset_info 目录，
    # 再试 gresearch/robotics 的数据目录（部分子集只在一边注册）
    if version is None:
        candidates = []
        info = _get_json(
            "https://storage.googleapis.com/storage/v1/b/tfds-data/o"
            f"?prefix=dataset_info/{name}/&delimiter=/&fields=prefixes"
        )
        candidates += [p.strip("/").split("/")[-1] for p in (info or {}).get("prefixes", [])]
        info = _get_json(
            "https://storage.googleapis.com/storage/v1/b/gresearch/o"
            f"?prefix=robotics/{name}/&delimiter=/&fields=prefixes"
        )
        candidates += [p.strip("/").split("/")[-1] for p in (info or {}).get("prefixes", [])]
        if candidates:
            version = sorted(candidates)[-1]
            out["version"] = version

    if version:
        # 对象列表求和（分页）；两个候选 bucket 取第一个有对象的
        for bucket in _GCS_BUCKETS:
            prefix = f"{_GCS_PREFIXES[bucket]}/{name}/{version}/"
            total_bytes, page_token, pages = 0, None, 0
            while True:
                url = (f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
                       f"?prefix={prefix}&fields=items(size),nextPageToken&maxResults=1000")
                if page_token:
                    url += f"&pageToken={page_token}"
                page = _get_json(url)
                if page is None:
                    break
                pages += 1
                total_bytes += sum(int(it.get("size", 0)) for it in page.get("items", []))
                page_token = page.get("nextPageToken")
                if not page_token:
                    break
            if total_bytes > 0:
                out["size_gb"] = round(total_bytes / 1e9, 2)
                out["available"] = True
                break

        # episode 数：dataset_info.json 的 shardLengths 求和
        info = _get_json(f"{_GCS_DATA}/dataset_info/{name}/{version}/dataset_info.json")
        if info:
            n = 0
            for sp in info.get("splits", []):
                n += sum(int(x) for x in sp.get("shardLengths", []))
                if "numExamples" in sp:
                    n += int(sp["numExamples"])
            if n:
                out["episodes"] = n

    return out


def load_openx_subset(
    tfds_name: str,
    split: str = "train",
    max_episodes: Optional[int] = None,
    embodiment_tag: Optional[str] = None,
    source: DataSource = DataSource.TELEOP,
    fps: float = 10.0,
) -> List[Episode]:
    """加载 Open X 子集为 Episode 列表。

    例::

        eps = load_openx_subset("viola/0.1.0", split="train", max_episodes=135)
        # embodiment 自动映射为 franka，dataset_name 记为 openx/viola

    注意：TFDS 必须整包 prepare 到本地才能逐条迭代。灌库脚本侧已用
    ``preflight_openx`` 做下载前大小确认；直接调用本函数请自行评估。
    """
    from robotloop.convert.rlds import load_tfds_rlds

    short = tfds_name.split("/")[0]
    embodiment = embodiment_tag or OPENX_EMBODIMENT_MAP.get(short, "unknown")
    episodes = load_tfds_rlds(
        tfds_name,
        split=split,
        max_episodes=max_episodes,
        embodiment_tag=embodiment,
        source=source,
        fps=fps,
        dataset_name=f"openx/{short}",
    )
    for ep in episodes:
        ep.dataset_name = f"openx/{short}"
    return episodes
