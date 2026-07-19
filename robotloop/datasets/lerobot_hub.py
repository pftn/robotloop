"""LeRobot Hub 数据集加载。

HF Hub 上的 LeRobot 数据集（v2.1 / v3.0 并存 —— 这正是需要双版本兼容
的现实原因）统一下载后走 convert 层读为 Episode。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from robotloop.schema.episode import DataSource, Episode

logger = logging.getLogger(__name__)


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def preflight_lerobot(repo_id: str, timeout: float = 15.0) -> Dict[str, Any]:
    """下载前估算 LeRobot 数据集的大小与 episode 数（只读元数据，不拉数据）。

    返回 {"repo_id", "size_gb", "episodes", "available"}；元数据不可达时
    对应字段为 None 且 available=False，由调用方决定如何处理。
    """
    import json
    import urllib.request

    endpoint = _ensure_hf_endpoint()
    out: Dict[str, Any] = {"repo_id": repo_id, "size_gb": None, "episodes": None, "available": False}

    def _get_json(url: str) -> Optional[dict]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            logger.debug("preflight fetch failed: %s (%s)", url, e)
            return None

    # 文件大小：HF API 的 siblings（带 size）
    api = _get_json(f"{endpoint}/api/datasets/{repo_id}?full=true")
    if api:
        sizes = [s.get("size") for s in api.get("siblings", []) if s.get("size")]
        if sizes:
            out["size_gb"] = round(sum(sizes) / 1e9, 3)
            out["available"] = True

    # episode 数：只下载 meta/info.json（KB 级）
    info = _get_json(f"{endpoint}/datasets/{repo_id}/resolve/main/meta/info.json")
    if info and info.get("total_episodes") is not None:
        out["episodes"] = int(info["total_episodes"])
        out["available"] = True

    return out


def _ensure_hf_endpoint() -> str:
    """HF 访问走镜像（用户要求：ENV HF_ENDPOINT=https://hf-mirror.com）。

    未显式设置时默认镜像站；已设置则尊重用户环境。
    """
    return os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)


def load_lerobot_hub(
    repo_id: str,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    embodiment_tag: str = "",
    source: DataSource = DataSource.TELEOP,
) -> List[Episode]:
    """下载并读取 HF Hub 上的 LeRobot 数据集（自动识别 v2.1/v3.0）。

    默认经 HF_ENDPOINT=https://hf-mirror.com 镜像下载。

    例::

        eps = load_lerobot_hub("lerobot/aloha_mobile_cabinet")
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("需要 huggingface_hub: pip install huggingface_hub") from e

    endpoint = _ensure_hf_endpoint()
    root = local_dir or os.path.join(
        os.path.expanduser("~/.cache/robotloop/hub"), repo_id.replace("/", "__")
    )
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=root, revision=revision,
        endpoint=endpoint,
        allow_patterns=["meta/**", "data/**", "videos/**", "README.md"],
    )
    return load_lerobot_dir(root, dataset_name=f"hf/{repo_id}",
                            embodiment_tag=embodiment_tag, source=source)


def load_lerobot_dir(
    root: str,
    dataset_name: str = "",
    embodiment_tag: str = "",
    source: DataSource = DataSource.TELEOP,
) -> List[Episode]:
    """读本地 LeRobot 目录（v2.1/v3.0 自动识别）。"""
    from robotloop.convert.cli import detect_version
    from robotloop.convert.lerobot_v21 import read_lerobot_v21
    from robotloop.convert.lerobot_v30 import read_lerobot_v30

    ver = detect_version(root)
    reader = {"v2.1": read_lerobot_v21, "v3.0": read_lerobot_v30}.get(ver)
    if reader is None:
        raise ValueError(f"不支持的 LeRobot 版本: {ver}")
    return reader(root, dataset_name=dataset_name or os.path.basename(root.rstrip("/")),
                  embodiment_tag=embodiment_tag, source=source)
