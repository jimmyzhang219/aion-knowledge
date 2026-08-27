"""VectorRepository 单元测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from aion_knowledge.storage.relational.vector_repo import VectorRepository


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.fixture
def repo(mock_session):
    return VectorRepository(mock_session)


@pytest.mark.asyncio
async def test_similarity_search(mock_session, repo):
    mock_result = MagicMock()
    mock_row = MagicMock()
    mock_row.id = "chunk-1"
    mock_row.content = "content"
    mock_row.document_id = "doc-1"
    mock_row.kb_id = "kb-1"
    mock_row.score = 0.85
    mock_result.__iter__.return_value = [mock_row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.similarity_search("kb-1", [0.1, 0.2, 0.3], 10)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 0.85


@pytest.mark.asyncio
async def test_similarity_search_empty(mock_session, repo):
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.similarity_search("kb-1", [0.1, 0.2, 0.3], 10)
    assert results == []


@pytest.mark.asyncio
async def test_similarity_search_with_chunk_type(mock_session, repo):
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    await repo.similarity_search("kb-1", [0.1], 5, chunk_type="faq")
    call_args = mock_session.execute.call_args[0][0].text
    assert "chunk_type" in call_args


@pytest.mark.asyncio
async def test_similarity_search_with_column(mock_session, repo):
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    await repo.similarity_search("kb-1", [0.1], 5, embedding_column="embedding_questions")
    call_args = mock_session.execute.call_args[0][0].text
    assert "embedding_questions" in call_args


@pytest.mark.asyncio
async def test_insert(mock_session, repo):
    mock_session.execute = AsyncMock()
    await repo.insert(uuid4(), "chunk-1", "kb-1", [0.1, 0.2], {"type": "text"})
    assert mock_session.execute.called


@pytest.mark.asyncio
async def test_update_questions_with_embedding(mock_session, repo):
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    count = await repo.update_questions("chunk-1", "question?", [0.1, 0.2])
    assert count == 1


@pytest.mark.asyncio
async def test_update_questions_without_embedding(mock_session, repo):
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    count = await repo.update_questions("chunk-1", "question?")
    assert count == 1


@pytest.mark.asyncio
async def test_update_summary_embedding(mock_session, repo):
    mock_session.execute = AsyncMock()
    await repo.update_summary_embedding("chunk-1", [0.1, 0.2])
    assert mock_session.execute.called


@pytest.mark.asyncio
async def test_fetch_embeddings_parses_pgvector_text(mock_session, repo):
    """fetch_embeddings 按 chunk_id 批量取回 embedding，pgvector 文本经 json.loads 解析。"""
    mock_result = MagicMock()
    row = MagicMock()
    row.chunk_id = "chunk-1"
    row.embedding = "[0.1, 0.2, 0.3]"  # pgvector 文本表示
    mock_result.__iter__.return_value = [row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    emb_map = await repo.fetch_embeddings(["chunk-1", "chunk-2"])

    assert emb_map == {"chunk-1": [0.1, 0.2, 0.3]}  # chunk-2 未命中，不在 map
    sql = mock_session.execute.call_args[0][0].text
    assert "ANY" in sql and "uuid[]" in sql


@pytest.mark.asyncio
async def test_fetch_embeddings_empty_input(repo):
    """空 chunk_id 列表不应触库，直接返回 {}。"""
    assert await repo.fetch_embeddings([]) == {}
