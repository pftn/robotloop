"""检索 API 扩展 —— 挂到 RobotLoop 既有 FastAPI（cloud/api/main.py）上。

用法（cloud/api/main.py 末尾）::

    from robotloop.retrieval.api import make_episode_router
    from robotloop.retrieval.store import MilvusIcebergStore
    app.include_router(make_episode_router(MilvusIcebergStore()))

端点：
- POST /episodes/search   混合检索（语义 + 结构化）
- GET  /episodes/stats    任务分布统计（质量工具集的 dashboard 数据面）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel


class EpisodeSearchRequest(BaseModel):
    text_query: Optional[str] = None
    nl_query: bool = True                    # 自然语言解析（抽取结构化谓词）
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 10


def make_episode_router(store, encoder=None) -> APIRouter:
    from robotloop.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(store, encoder=encoder)
    router = APIRouter(prefix="/episodes", tags=["episodes"])

    @router.post("/search")
    def search(req: EpisodeSearchRequest):
        """例：{"text_query": "找所有 ALOHA 双臂成功抓取红色方块的轨迹"}"""
        return retriever.search(
            text=req.text_query, filters=req.filters, top_k=req.top_k, nl_query=req.nl_query
        )

    @router.get("/stats")
    def stats():
        from robotloop.quality.dashboard import task_distribution

        return task_distribution(store.all_meta())

    @router.get("/count")
    def count():
        return {"count": store.count()}

    return router
