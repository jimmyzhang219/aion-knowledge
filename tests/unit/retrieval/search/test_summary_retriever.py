"""SummaryRetriever 测试 — 验证 document_id/content 回填路径。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.summary_retriever import SummaryRetriever


@pytest.mark.asyncio
async def test_retrieve_fills_document_id_from_chunk_text():
    """BM25 命中后应回查 chunk_text 回填 content 与 document_id。"""
    retriever = SummaryRetriever()
    ctx = RetrieverContext(query="摘要", kb_id="kb-1")

    bm25_rows = [
        SimpleNamespace(id="c1", content="", document_id="", kb_id="kb-1", score=0.8),
    ]
    fill_rows = [
        SimpleNamespace(id="c1", content="真实摘要内容", document_id="d1"),
    ]
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[bm25_rows, fill_rows])

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.summary_retriever.get_session", mock_get_session):
        results = await retriever.retrieve(ctx)

    assert len(results) == 1
    assert results[0].document_id == "d1"
    assert results[0].content == "真实摘要内容"
    assert results[0].source_paths == ["summary"]


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_empty():
    """空查询返回 []。"""
    retriever = SummaryRetriever()
    ctx = RetrieverContext(query="", kb_id="kb-1")
    assert await retriever.retrieve(ctx) == []


@pytest.mark.asyncio
async def test_vector_branch_filters_by_kb_id():
    """向量分支应带 kb_id 过滤，避免跨知识库召回。"""
    retriever = SummaryRetriever()
    ctx = RetrieverContext(query="摘要", kb_id="kb-1", query_embedding=[0.1, 0.2, 0.3])

    bm25_rows = []
    vec_rows = [SimpleNamespace(chunk_id="c1", score=0.9)]
    fill_rows = [SimpleNamespace(id="c1", content="原文", document_id="d1")]
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[bm25_rows, vec_rows, fill_rows])

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.summary_retriever.get_session", mock_get_session):
        await retriever.retrieve(ctx)

    # 第二次 execute 是向量分支（第一次 BM25、第三次回填 content）
    vec_call = session.execute.await_args_list[1]
    vec_sql = vec_call.args[0].text  # sqlalchemy text 对象的 SQL 文本
    vec_params = vec_call.args[1]  # params 是 execute 的位置参数，非 kwargs
    assert "v.kb_id = :kb_id" in vec_sql
    assert vec_params["kb_id"] == "kb-1"
