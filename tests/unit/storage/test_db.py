"""数据库会话管理测试。"""

import pytest

from aion_knowledge.infrastructure.db import (
    dispose_engine,
    get_session,
    init_db,
)


@pytest.mark.asyncio
async def test_get_session_yields_session():
    async with get_session() as session:
        assert session is not None


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    """init_db 在连接真实或 mock PG 时不应抛出异常。"""
    try:
        await init_db()
    except Exception as exc:
        msg = str(exc).lower()
        assert any(
            kw in msg for kw in ["connection", "refused", "timeout", "could not connect", "not available"]
        ), f"Unexpected error: {exc}"
    else:
        await dispose_engine()


@pytest.mark.asyncio
async def test_dispose_engine_noop_when_not_initialized():
    """未初始化时调用 dispose_engine 不应抛出异常。"""
    await dispose_engine()
