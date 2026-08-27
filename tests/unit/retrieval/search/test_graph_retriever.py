"""GraphRetriever 测试 — 回捞真实 chunk 原文。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.graph_retriever import GraphRetriever


@pytest.mark.asyncio
async def test_retrieve_returns_real_chunks():
    """graph_retriever 应回捞真实 chunk 原文，返回真实 ChunkResult。"""
    retriever = GraphRetriever()
    ctx = RetrieverContext(query="马云", kb_id="kb-1", entities=["马云"])

    # DB 返回顺序故意乱序（c2 在前），验证 graph_retriever 按 score 重排
    mock_rows = [
        SimpleNamespace(id="c2", content="张勇是CEO", document_id="d1",
                        chunk_type="text", chunk_metadata={}),
        SimpleNamespace(id="c1", content="马云创建阿里巴巴", document_id="d1",
                        chunk_type="text", chunk_metadata={}),
    ]
    mock_repo = MagicMock()
    mock_repo.get_by_ids = AsyncMock(return_value=mock_rows)
    # 前置检查：KB deleted 查询须返回 None（KB 未删），流程继续走 search_entities
    mock_session = MagicMock()
    mock_session.scalar = AsyncMock(return_value=None)

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    with patch("aion_knowledge.infrastructure.graph.search_entities", AsyncMock(return_value=[
        {"entity_name": "马云", "entity_type": "PERSON", "description": "创始人",
         "weight": 1.0, "similarity": 0.9, "chunk_ids": ["c1", "c2"]},
    ])):
        with patch("aion_knowledge.infrastructure.db.get_session", mock_get_session):
            with patch("aion_knowledge.storage.relational.chunk_repo.ChunkRepository",
                       return_value=mock_repo):
                results = await retriever.retrieve(ctx)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].content == "马云创建阿里巴巴"
    assert results[0].document_id == "d1"
    assert results[0].source_paths == ["graph"]
    assert results[0].score == 0.9
    assert results[1].chunk_id == "c2"
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1)), "应按 score 降序"
    # 不再有伪造的 kg_{hash} chunk_id
    assert all(not r.chunk_id.startswith("kg_") for r in results)


@pytest.mark.asyncio
async def test_retrieve_empty_entities_returns_empty():
    """search_entities 返回空列表时，retrieve 返回 []。"""
    retriever = GraphRetriever()
    ctx = RetrieverContext(query="x", kb_id="kb-1", entities=["x"])

    mock_session = MagicMock()
    mock_session.scalar = AsyncMock(return_value=None)

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    with patch("aion_knowledge.infrastructure.db.get_session", mock_get_session), \
         patch("aion_knowledge.infrastructure.graph.search_entities", AsyncMock(return_value=[])):
        results = await retriever.retrieve(ctx)

    assert results == []


@pytest.mark.asyncio
async def test_retrieve_short_circuits_when_kb_deleted():
    """KB deleted=true：不调 Neo4j search_entities，返回空。"""
    ctx = RetrieverContext(query="实体", kb_id="kb-1", top_k=10)
    mock_session = AsyncMock()
    mock_session.scalar = AsyncMock(return_value=True)
    with patch("aion_knowledge.infrastructure.db.get_session") as mock_gs, \
         patch("aion_knowledge.infrastructure.graph.search_entities") as mock_se:
        mock_gs.return_value.__aenter__.return_value = mock_session
        results = await GraphRetriever().retrieve(ctx)
    assert results == []
    mock_se.assert_not_awaited()


def test_aggregate_chunks_takes_max_similarity_per_chunk():
    """同 chunk 被多实体关联时取最高 similarity，去重、降序。"""
    entities = [
        {"entity_name": "马云", "entity_type": "PERSON", "similarity": 0.9, "chunk_ids": ["c1", "c2"]},
        {"entity_name": "阿里巴巴", "entity_type": "ORG", "similarity": 0.8, "chunk_ids": ["c2", "c3"]},
    ]
    result = GraphRetriever._aggregate_chunks(entities, top_k=10)

    chunk_scores = dict(result)
    assert set(chunk_scores) == {"c1", "c2", "c3"}, "应去重聚合所有 chunk_id"
    # c2 被两实体关联，取最高 similarity (0.9)
    assert chunk_scores["c2"] == 0.9
    # 按 score 降序
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_aggregate_chunks_top_k_limit():
    """按 similarity 排序后取前 top_k。"""
    entities = [
        {"entity_name": "E1", "entity_type": "T", "similarity": 0.9, "chunk_ids": ["c1"]},
        {"entity_name": "E2", "entity_type": "T", "similarity": 0.5, "chunk_ids": ["c2"]},
    ]
    result = GraphRetriever._aggregate_chunks(entities, top_k=1)
    assert result == [("c1", 0.9)]


def test_aggregate_chunks_empty():
    """空实体列表返回空列表。"""
    assert GraphRetriever._aggregate_chunks([], top_k=10) == []
