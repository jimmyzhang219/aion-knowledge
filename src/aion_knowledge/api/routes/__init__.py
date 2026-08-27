"""路由汇总。"""

from __future__ import annotations

from fastapi import FastAPI

from aion_knowledge.api.routes.api_direct import router as api_direct_router
from aion_knowledge.api.routes.faq import router as faq_router
from aion_knowledge.api.routes.health import router as health_router
from aion_knowledge.api.routes.knowledge_base import router as knowledge_base_router
from aion_knowledge.api.routes.regular import router as ingestion_router
from aion_knowledge.api.routes.search import router as search_router


def register_routers(app: FastAPI) -> None:
    """注册所有 API 路由分组。"""
    app.include_router(health_router)
    app.include_router(ingestion_router)
    app.include_router(knowledge_base_router)
    app.include_router(faq_router)
    app.include_router(api_direct_router)
    app.include_router(search_router)
