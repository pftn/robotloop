"""混合检索：CLIP 语义 × Iceberg 结构化过滤，联合命中。

demo 场景："找所有 ALOHA 双臂成功抓取红色方块的轨迹"
  ├── 结构化侧（Iceberg）：embodiment_tag='aloha' AND success=true
  └── 语义侧（向量库）：encode("抓取红色方块") 做 ANN
  两侧取交集，按语义相似度排序。

NL 查询解析（``parse_nl_query``）是 demo 的"翻译层"：把自然语言里的
本体/成败/来源关键词抽成结构化谓词，剩余部分留给语义编码器。
真实系统里这一层可以换成 LLM function-calling，接口不变。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robotloop.retrieval.encoder import get_text_encoder

# 中文/英文关键词 → 结构化谓词
_EMBODIMENT_HINTS = {
    "franka": "franka", "法兰克": "franka", "panda": "franka",
    "widowx": "widowx", "bridge": "widowx",
    "google": "google_robot", "rt-1": "google_robot", "rt1": "google_robot",
    "aloha": "aloha",
    "so100": "so100", "so-100": "so100", "so101": "so100",
    "agibot": "agibot_g1", "智元": "agibot_g1",
    "unitree": "unitree_g1", "宇树": "unitree_g1",
    "xarm": "xarm", "ur5": "ur5",
}
_SUCCESS_HINTS = {"成功": True, "success": True, "successful": True, "失败": False, "fail": False, "failed": False}
_SOURCE_HINTS = {
    "仿真": "sim", "simulation": "sim", "sim": "sim",
    "遥操作": "teleop", "teleop": "teleop",
    "真机": "real", "real-robot": "real",
}


def parse_nl_query(query: str) -> Tuple[str, Dict[str, Any]]:
    """自然语言 → (残余语义文本, 结构化过滤条件)。

    抽走结构化关键词后，剩余文本仍送给 CLIP 编码（如"抓取红色方块"），
    语义与结构各司其职。
    """
    filters: Dict[str, Any] = {}
    residual = query

    for kw, tag in _EMBODIMENT_HINTS.items():
        if kw in query.lower() or kw in query:
            filters["embodiment_tag"] = tag
            residual = re.sub(re.escape(kw), " ", residual, flags=re.IGNORECASE)
            break
    for kw, val in _SUCCESS_HINTS.items():
        if kw in query.lower() or kw in query:
            filters["success"] = val
            residual = re.sub(re.escape(kw), " ", residual, flags=re.IGNORECASE)
            break
    for kw, val in _SOURCE_HINTS.items():
        if re.search(rf"\b{re.escape(kw)}\b", query.lower()) or kw in query:
            filters["source"] = val
            residual = re.sub(re.escape(kw), " ", residual, flags=re.IGNORECASE)
            break

    residual = re.sub(r"(找所有|查找|搜索|的轨迹|轨迹|请问|一下)", " ", residual)
    residual = re.sub(r"\s+", " ", residual).strip()
    return residual, filters


class HybridRetriever:
    """向量候选 + Iceberg/parquet 结构化过滤的联合检索。"""

    def __init__(self, store, encoder=None, candidate_multiplier: int = 10):
        self.store = store
        self.encoder = encoder or get_text_encoder()
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        nl_query: bool = False,
    ) -> Dict[str, Any]:
        """混合检索主入口。

        - ``nl_query=True`` 时先把 text 过一遍 parse_nl_query，抽出的结构化
          条件与显式 filters 合并（显式优先）。
        - 有文本 → 向量召回候选 + 结构化过滤取交集，按相似度排序；
        - 无文本 → 纯结构化过滤（Iceberg 谓词下推），按 created_at 倒序。
        """
        parsed: Dict[str, Any] = {}
        residual_text = text
        if nl_query and text:
            residual_text, parsed = parse_nl_query(text)
        merged_filters = {**parsed, **(filters or {})}

        meta_tbl = self.store.filter_meta(merged_filters)
        meta_rows = meta_tbl.to_pylist()
        meta_by_id = {r["episode_id"]: r for r in meta_rows}

        results: List[Dict[str, Any]] = []
        if residual_text:
            vec = self.encoder.encode(residual_text)
            candidates = self.store.search_vectors(
                np.asarray(vec), top_k * self.candidate_multiplier
            )
            for eid, score in candidates:
                if eid in meta_by_id:
                    row = dict(meta_by_id[eid])
                    row["score"] = score
                    results.append(row)
                if len(results) >= top_k:
                    break
        else:
            results = sorted(
                (dict(r) for r in meta_rows),
                key=lambda r: r.get("created_at") or 0,
                reverse=True,
            )[:top_k]
            for r in results:
                r["score"] = None

        return {
            "count": len(results),
            "results": results,
            "query": {
                "text": text,
                "residual_text": residual_text,
                "structured_filters": merged_filters,
                "top_k": top_k,
            },
        }
