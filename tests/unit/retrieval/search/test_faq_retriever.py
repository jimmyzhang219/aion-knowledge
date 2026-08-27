"""FAQRetriever 测试 — 验证使用 original_query 做文本匹配（P1b）。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.faq_retriever import FAQRetriever
from aion_knowledge.storage.relational.chunk_repo import FAQRow


def _faq_row() -> FAQRow:
    return FAQRow(
        id="faq-1",
        kb_id="kb-1",
        document_id="doc-1",
        content="Q: 什么是AION知识库\nA: AION知识库是一个专注于RAG三阶段核心流程的知识库系统。",
        standard_question="什么是AION知识库",
        similar_questions=["AION知识库是什么", "AION知识库有哪些功能"],
        negative_questions=[],
        answers=["AION知识库是一个专注于RAG三阶段核心流程的知识库系统。"],
        category="产品",
    )


def _session_ctxmgr() -> MagicMock:
    session = MagicMock()

    @asynccontextmanager
    async def _get_session():
        yield session

    return _get_session


@pytest.mark.asyncio
async def test_faq_retriever_uses_original_query_for_text_match():
    """改写后 ctx.query 是关键词串，FAQRetriever 应把 original_query 传给 search_faq。"""
    retriever = FAQRetriever()
    ctx = RetrieverContext(
        query="AION知识库 介绍 说明 定义",   # 改写后
        kb_id="kb-1",
        original_query="AION知识库是什么",   # 原始
        query_embedding=None,               # 关闭向量层，聚焦文本层
    )

    with (
        patch("aion_knowledge.retrieval.search.faq_retriever.get_session", _session_ctxmgr()),
        patch.object(
            __import__(
                "aion_knowledge.storage.relational.chunk_repo", fromlist=["ChunkRepository"]
            ).ChunkRepository,
            "search_faq",
            AsyncMock(return_value=[_faq_row()]),
        ) as mock_search,
    ):
        results = await retriever.retrieve(ctx)

    assert mock_search.call_args.kwargs["query"] == "AION知识库是什么"
    assert len(results) == 1


@pytest.mark.asyncio
async def test_faq_retriever_falls_back_to_query_when_no_original():
    """original_query 缺省（如 rewrite 未启用）时回退到 ctx.query。"""
    retriever = FAQRetriever()
    ctx = RetrieverContext(
        query="如何上传文档",
        kb_id="kb-1",
        original_query=None,
        query_embedding=None,
    )

    with (
        patch("aion_knowledge.retrieval.search.faq_retriever.get_session", _session_ctxmgr()),
        patch.object(
            __import__(
                "aion_knowledge.storage.relational.chunk_repo", fromlist=["ChunkRepository"]
            ).ChunkRepository,
            "search_faq",
            AsyncMock(return_value=[]),
        ) as mock_search,
    ):
        await retriever.retrieve(ctx)

    assert mock_search.call_args.kwargs["query"] == "如何上传文档"
