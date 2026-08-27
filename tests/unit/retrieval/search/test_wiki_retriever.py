"""WikiRetriever 单元测试 — 命中后回捞源 chunk 原文。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.wiki_retriever import WikiRetriever


@pytest.mark.asyncio
async def test_retrieve_returns_source_chunk():
    """wiki 命中应 INNER JOIN chunk_text，返回源 chunk 原文与 chunk_text.id。"""
    retriever = WikiRetriever()
    ctx = RetrieverContext(query="人工智能", kb_id="kb-1", top_k=10)

    # mock 行：id/content/document_id 来自 chunk_text，wiki_page_id 来自 chunk_wiki
    rows = [
        SimpleNamespace(
            id="chunk-text-1",
            content="源 chunk 原文一",
            document_id="doc-1",
            page_title="人工智能",
            page_slug="ai",
            wiki_page_id="wiki-1",
            score=0.9,
        ),
        SimpleNamespace(
            id="chunk-text-2",
            content="源 chunk 原文二",
            document_id="doc-2",
            page_title="机器学习",
            page_slug="ml",
            wiki_page_id="wiki-2",
            score=0.5,
        ),
    ]
    session = MagicMock()
    session.execute = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.wiki_retriever.get_session", mock_get_session):
        results = await retriever.retrieve(ctx)

    assert len(results) == 2
    # chunk_id 来自 chunk_text（不是 wiki 主键 wiki_page_id）
    assert results[0].chunk_id == "chunk-text-1"
    assert results[0].chunk_id != "wiki-1"
    # content 是源 chunk 原文（无 "# title" 前缀）
    assert results[0].content == "源 chunk 原文一"
    assert not results[0].content.startswith("# ")
    # document_id 来自 chunk_text（不再是空串）
    assert results[0].document_id == "doc-1"
    # wiki 溯源信息保留在 metadata
    assert results[0].metadata["wiki_page_id"] == "wiki-1"
    assert results[0].metadata["page_title"] == "人工智能"
    assert results[0].metadata["page_slug"] == "ai"
    # source_paths / score 降序
    assert results[0].source_paths == ["wiki"]
    assert results[0].score == 0.9
    assert results[1].score == 0.5


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_empty():
    """空查询返回 []。"""
    retriever = WikiRetriever()
    ctx = RetrieverContext(query="", kb_id="kb-1")
    assert await retriever.retrieve(ctx) == []


@pytest.mark.asyncio
async def test_retrieve_multiple_chunks_per_page():
    """一页引用多 chunk 时应返回每个源 chunk 一条结果，score 继承页面分。"""
    retriever = WikiRetriever()
    ctx = RetrieverContext(query="人工智能", kb_id="kb-1", top_k=10)
    rows = [
        SimpleNamespace(id="c1", content="chunk 一", document_id="doc-1",
                        page_title="人工智能", page_slug="ai", wiki_page_id="w1", score=0.9),
        SimpleNamespace(id="c2", content="chunk 二", document_id="doc-1",
                        page_title="人工智能", page_slug="ai", wiki_page_id="w1", score=0.9),
    ]
    session = MagicMock()
    session.execute = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.wiki_retriever.get_session", mock_get_session):
        results = await retriever.retrieve(ctx)
    assert len(results) == 2
    assert {r.chunk_id for r in results} == {"c1", "c2"}
    assert all(r.metadata["page_slug"] == "ai" for r in results)
    assert all(r.score == 0.9 for r in results)
