"""FastAPI 应用工厂。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.routing import Route

from aion_knowledge.api.mcp_server import get_mcp_streamable_http_app
from aion_knowledge.api.middleware import register_middleware
from aion_knowledge.api.routes import register_routers
from aion_knowledge.common.config import settings
from aion_knowledge.common.logger import setup_logging
from aion_knowledge.infrastructure.db import dispose_engine, init_db
from aion_knowledge.infrastructure.workers import pipeline_worker, postproc_worker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动后台 Worker，MCP Session Manager，关闭时释放资源。"""
    setup_logging()
    logger.info(
        "Aion Knowledge API starting on port %d",
        app.state.port if hasattr(app.state, "port") else 19531,
    )

    await init_db()

    # 重新配置日志，恢复正常的 level 和 handler（历史：Alembic 迁移的 fileConfig()
    # 曾禁用 logger 并覆盖根级别；当前 schema 同步为进程内执行，无此副作用，
    # 重复调用幂等无害，保留作防御）。
    setup_logging()
    logger.info("Database tables initialized.")

    # 启动 Pipeline Worker（解析 → 分块）
    task = asyncio.create_task(pipeline_worker(), name="pipeline_worker")
    app.state.pipeline_worker = task

    # INDEX_ONLY 模式下不启动后处理 Worker
    if not settings.index_only:
        task = asyncio.create_task(postproc_worker(), name="postproc_worker")
        app.state.postproc_worker = task
    else:
        logger.info("INDEX_ONLY 模式：后处理 Worker 未启动")

    # 初始化 MCP Session Manager 生命周期
    # 注：Starlette 的 Router 不会将 lifespan 事件传递给 mount 的子应用，
    # 因此需要在父应用 lifespan 中手动启动 MCP Session Manager。
    from aion_knowledge.api.mcp_server import mcp as mcp_instance

    if mcp_instance._session_manager is not None:
        async with mcp_instance._session_manager.run():
            yield
    else:
        yield

    # 关闭
    app.state.pipeline_worker.cancel()
    if hasattr(app.state, "postproc_worker"):
        app.state.postproc_worker.cancel()
    await dispose_engine()
    logger.info("Aion Knowledge API shut down.")


def create_app() -> FastAPI:
    """创建 FastAPI 实例并注册路由和中间件。"""
    app = FastAPI(
        title="Aion Knowledge API",
        description="Aion 知识库",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.port = 19531

    register_middleware(app)
    register_routers(app)

    # 注册 MCP Streamable HTTP 端点（直接用 Route 避免 Mount 的尾缀斜杠重定向）
    mcp_app = get_mcp_streamable_http_app()
    # routes[0] 类型为 BaseRoute（无 endpoint 属性），MCP 实际注册的是 Route，getattr 取端点
    mcp_route = mcp_app.routes[0]
    app.router.routes.append(
        Route("/mcp", endpoint=getattr(mcp_route, "endpoint"), methods=["POST"])
    )
    logger.info("MCP Streamable HTTP endpoint registered at /mcp")

    return app
