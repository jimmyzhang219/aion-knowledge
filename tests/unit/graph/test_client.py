"""GraphClient 单元测试。"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from aion_knowledge.infrastructure.graph.client import Neo4jConnection


@pytest.mark.asyncio
async def test_execute_write_disabled():
    """neo4j_enabled=False 时跳过执行。"""
    with patch("aion_knowledge.common.config.settings.neo4j_enabled", False):
        conn = Neo4jConnection()
        result = await conn.execute_write(lambda tx: "should_not_run")
        assert result is None


@pytest.mark.asyncio
async def test_execute_read_disabled():
    """neo4j_enabled=False 时跳过读取。"""
    with patch("aion_knowledge.common.config.settings.neo4j_enabled", False):
        conn = Neo4jConnection()
        result = await conn.execute_read(lambda tx: "should_not_run")
        assert result is None
