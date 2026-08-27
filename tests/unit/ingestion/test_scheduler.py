"""Scheduler 测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.ingestion.scheduler import start_workers, stop_workers


@pytest.mark.asyncio
async def test_start_workers_creates_tasks():
    """验证 start_workers 会创建两个 worker task。"""
    tasks = await start_workers()
    assert len(tasks) == 2
    assert all(t is not None for t in tasks)
    assert all(hasattr(t, "cancel") for t in tasks)
    # 清理
    await stop_workers(tasks)


@pytest.mark.asyncio
async def test_start_workers_stale_checkpoint_logging():
    """验证启动时扫描 stale checkpoint 不会抛异常。"""
    with patch("aion_knowledge.ingestion.scheduler.CheckpointManager") as mock_mgr:
        instance = mock_mgr.return_value
        instance.list_stale = AsyncMock(return_value=[])
        tasks = await start_workers()
        instance.list_stale.assert_awaited_once()
        await stop_workers(tasks)
