"""document_repo 单元测试。"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from aion_knowledge.ingestion.document_repo import (
    create_document,
    get_document_by_hash,
    get_document_by_id,
)

_UUID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.mark.asyncio
async def test_get_document_by_id_found(mock_session):
    """按 ID 命中返回文档对象。"""
    doc = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = doc
    mock_session.execute.return_value = mock_result

    found = await get_document_by_id(mock_session, _UUID)

    assert found is doc
    mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_document_by_id_not_found(mock_session):
    """未命中返回 None。"""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    found = await get_document_by_id(mock_session, uuid.UUID(_UUID))

    assert found is None


@pytest.mark.asyncio
async def test_create_document_persists_trace_id(mock_session) -> None:
    """create_document 落库 trace_id。"""
    doc = await create_document(
        session=mock_session,
        kb_id=_UUID,
        doc_name="t.md", suffix="md", file_hash="h1", size=10,
        trace_id="doc-trace-1",
    )
    assert doc.trace_id == "doc-trace-1"


@pytest.mark.asyncio
async def test_get_document_by_hash_matches_hash_kb_not_deleted(mock_session):
    """查重条件：hash + kb_id + 未删除，不含 doc_name/size。"""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    await get_document_by_hash(mock_session, _UUID, "hash-abc")

    stmt = mock_session.execute.await_args.args[0]
    where = str(stmt.whereclause)
    assert "doc_knowledge_documents.deleted = false" in where
    assert "kb_id" in where
    assert "doc_name" not in where
    assert "size" not in where


@pytest.mark.asyncio
async def test_get_document_by_hash_returns_first_when_multiple(mock_session):
    """历史重复行（同 hash 多条未删除）时返回任一，不抛 MultipleResultsFound。"""
    doc = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = doc
    mock_session.execute.return_value = mock_result

    found = await get_document_by_hash(mock_session, _UUID, "hash-abc")

    assert found is doc


@pytest.mark.asyncio
async def test_get_document_by_id_excludes_deleted(mock_session):
    """按 ID 查询仅返回未删除文档（deleted 文档视为不存在）。"""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    await get_document_by_id(mock_session, _UUID)

    stmt = mock_session.execute.await_args.args[0]
    where = str(stmt.whereclause)
    assert "doc_knowledge_documents.id =" in where
    assert "doc_knowledge_documents.deleted = false" in where
