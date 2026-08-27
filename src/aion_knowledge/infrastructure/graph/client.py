"""Neo4j 连接管理 — 异步驱动封装。

设计原则：
- 所有操作通过 neo4j_enabled 开关控制，禁用时不做任何连接
- graph 启用但 Neo4j 未运行时不崩溃，仅记录警告
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TypeVar

from neo4j import AsyncDriver, AsyncGraphDatabase

from aion_knowledge.common.config import settings

logger = logging.getLogger(__name__)

# 事务回调返回值的类型参数：由调用方传入的回调函数返回类型推断
T = TypeVar("T")


class Neo4jConnection:
    """Neo4j 异步连接管理。"""

    _driver: AsyncDriver | None = None

    def __init__(
        self,
        uri: str = "",
        user: str = "",
        password: str = "",
    ):
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._enabled = settings.neo4j_enabled

    async def _ensure_driver(self) -> AsyncDriver | None:
        if not self._enabled:
            return None
        if self._driver is None:
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=10,
                )
                await self._driver.verify_connectivity()
                logger.info("Neo4j 连接成功：%s", self._uri)
            except Exception as exc:
                logger.warning("Neo4j 不可用（图谱后处理将跳过）: %s", exc)
                return None
        return self._driver

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j 连接已关闭")

    async def execute_write(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T | None:
        """执行写入事务。

        未启用时返回 None（调用方按 None 处理）；否则透传回调返回值。
        """
        driver = await self._ensure_driver()
        if driver is None:
            logger.debug("Neo4j 未启用（neo4j_enabled=false）")
            return None
        async with driver.session() as session:
            return await session.execute_write(func, *args, **kwargs)

    async def execute_read(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T | None:
        """执行读取事务。

        未启用时返回 None（调用方按 None 处理）；否则透传回调返回值。
        """
        driver = await self._ensure_driver()
        if driver is None:
            logger.debug("Neo4j 未启用（neo4j_enabled=false）")
            return None
        async with driver.session() as session:
            return await session.execute_read(func, *args, **kwargs)
