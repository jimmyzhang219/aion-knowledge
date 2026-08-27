"""CommunityRetriever 测试 — chunk_community.embedding 余弦向量检索。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.community_retriever import CommunityRetriever


def _mock_session(rows: list) -> MagicMock:
    session = MagicMock()
    session.execute = AsyncMock(return_value=rows)
    return session


@pytest.mark.asyncio
async def test_retrieve_no_embedding_returns_empty():
    """query_embedding 为 None 时跳过（与 VectorRetriever 一致）。"""
    retriever = CommunityRetriever()
    ctx = RetrieverContext(query="问题", kb_id="kb-1", query_embedding=None)
    assert await retriever.retrieve(ctx) == []


@pytest.mark.asyncio
async def test_retrieve_vector_search_includes_kb_level_communities():
    """向量检索返回摘要文本；KB 级社区（chunk_id 零值）不再被排除。"""
    retriever = CommunityRetriever()
    ctx = RetrieverContext(query="问题", kb_id="kb-1", query_embedding=[0.1, 0.2, 0.3])

    rows = [
        SimpleNamespace(
            id="comm-id-1", community_id="L0_9", community_level=0,
            title="沉浸式VR技术栈", summary="围绕 VR 大空间技术",
            findings=[{"summary": "硬件", "explanation": "头显与定位"}],
            members=["沉浸式VR"], score=0.92,
        ),
        SimpleNamespace(
            id="comm-id-2", community_id="L1_3", community_level=1,
            title="渲染引擎", summary="软件栈", findings=[], members=[], score=0.87,
        ),
    ]
    session = _mock_session(rows)

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.community_retriever.get_session", mock_get_session):
        results = await retriever.retrieve(ctx)

    community_sql = session.execute.await_args.args[0].text
    assert "embedding <=>" in community_sql
    assert "ORDER BY score DESC" in community_sql
    assert "INNER JOIN chunk_text" not in community_sql
    assert "00000000-0000-0000-0000-000000000000" not in community_sql

    assert len(results) == 2
    first = results[0]
    assert first.chunk_id == "comm-id-1"          # 用 chunk_community.id，不撞 RRF 键
    assert first.document_id == ""
    assert first.score == 0.92
    assert first.content == "沉浸式VR技术栈\n围绕 VR 大空间技术\n硬件: 头显与定位"
    assert first.source_paths == ["community"]
    assert first.metadata["community_id"] == "L0_9"
    assert first.metadata["community_level"] == 0
    assert first.metadata["members"] == ["沉浸式VR"]
    assert results[1].metadata["members"] == []


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_empty():
    """空查询返回 []（query_embedding 为空同路径）。"""
    retriever = CommunityRetriever()
    ctx = RetrieverContext(query="", kb_id="kb-1", query_embedding=None)
    assert await retriever.retrieve(ctx) == []
