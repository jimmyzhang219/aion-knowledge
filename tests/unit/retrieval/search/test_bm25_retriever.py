"""BM25Retriever 单元测试（pg_textsearch + zhparser 版本）。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.bm25_retriever import BM25Retriever


class TestBM25Retriever:
    """验证 BM25Retriever 的 pg_textsearch SQL 检索。"""

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self):
        """基础检索应返回 ChunkResult 列表。"""
        ctx = RetrieverContext(
            query="人工智能",
            kb_id="kb-1",
            top_k=10,
        )

        mock_row = MagicMock()
        mock_row.id = "chunk-1"
        mock_row.content = "人工智能技术内容"
        mock_row.document_id = "doc-1"
        mock_row.kb_id = "kb-1"
        mock_row.score = -0.5
        mock_row.metadata = None

        mock_bm25_result = MagicMock()
        mock_bm25_result.__iter__.return_value = [mock_row]

        with patch("aion_knowledge.retrieval.search.bm25_retriever.get_session") as mock_gs:
            mock_session = AsyncMock()
            mock_gs.return_value.__aenter__.return_value = mock_session
            mock_session.execute = AsyncMock(return_value=mock_bm25_result)

            retriever = BM25Retriever()
            results = await retriever.retrieve(ctx)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"
        assert results[0].score == -0.5
        assert "bm25" in results[0].source_paths

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self):
        """空查询应返回空列表。"""
        ctx = RetrieverContext(query="", kb_id="kb-1", top_k=10)
        retriever = BM25Retriever()
        results = await retriever.retrieve(ctx)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_with_expansion_keywords(self):
        """expansion_keywords 应与原查询拼接后传入。"""
        ctx = RetrieverContext(
            query="人工智能",
            kb_id="kb-1",
            top_k=10,
            expansion_keywords=["机器学习", "深度学习"],
        )

        mock_row = MagicMock()
        mock_row.id = "chunk-1"
        mock_row.content = "内容"
        mock_row.document_id = "doc-1"
        mock_row.kb_id = "kb-1"
        mock_row.score = -0.3
        mock_row.metadata = None

        mock_bm25_result = MagicMock()
        mock_bm25_result.__iter__.return_value = [mock_row]

        with patch("aion_knowledge.retrieval.search.bm25_retriever.get_session") as mock_gs:
            mock_session = AsyncMock()
            mock_gs.return_value.__aenter__.return_value = mock_session
            mock_session.execute = AsyncMock(return_value=mock_bm25_result)

            retriever = BM25Retriever()
            results = await retriever.retrieve(ctx)

        assert len(results) == 1
