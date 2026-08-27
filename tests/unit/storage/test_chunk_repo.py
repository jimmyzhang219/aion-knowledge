"""ChunkRepository 单元测试。"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aion_knowledge.storage.relational.chunk_repo import (
    ChunkRepository,
    FAQRow,
    _safe_json_list,
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.fixture
def repo(mock_session):
    return ChunkRepository(mock_session)


def _make_mock_row(mapping: dict) -> MagicMock:
    row = MagicMock()
    row._mapping = mapping
    return row


# ------------------------------------------------------------------ #
# get_by_document
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_by_document(mock_session, repo):
    mock_result = MagicMock()
    mock_row = _make_mock_row({
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": "test content",
        "document_id": "doc-1",
        "kb_id": "kb-1",
        "seq_num": 1,
        "chunk_type": "text",
        "metadata": None,
    })
    mock_result.__iter__.return_value = [mock_row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.get_by_document("doc-1")

    assert len(results) == 1
    assert results[0].id == "550e8400-e29b-41d4-a716-446655440000"
    assert results[0].content == "test content"
    assert results[0].document_id == "doc-1"
    assert results[0].seq_num == 1


@pytest.mark.asyncio
async def test_get_by_document_empty(mock_session, repo):
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.get_by_document("doc-nonexistent")
    assert results == []


# ------------------------------------------------------------------ #
# get_by_kb
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_by_kb(mock_session, repo):
    mock_result = MagicMock()
    mock_row = _make_mock_row({
        "id": "chunk-kb-1",
        "content": "content-sample",
        "document_id": "doc-1",
        "kb_id": "kb-1",
        "seq_num": 1,
        "chunk_type": "text",
        "metadata": None,
    })
    mock_result.__iter__.return_value = [mock_row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.get_by_kb("kb-1")
    assert len(results) == 1


# ------------------------------------------------------------------ #
# get_by_id
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_by_id_found(mock_session, repo):
    mock_result = MagicMock()
    mock_row = _make_mock_row({
        "id": "chunk-found",
        "content": "content-sample",
        "document_id": "doc-1",
        "kb_id": "kb-1",
        "seq_num": 1,
        "chunk_type": "text",
        "metadata": None,
    })
    mock_result.first.return_value = mock_row
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await repo.get_by_id("chunk-found")
    assert result is not None
    assert result.id == "chunk-found"


@pytest.mark.asyncio
async def test_get_by_id_not_found(mock_session, repo):
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await repo.get_by_id("chunk-nonexistent")
    assert result is None


# ------------------------------------------------------------------ #
# count_by_kb
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_count_by_kb(mock_session, repo):
    mock_result = MagicMock()
    mock_result.scalar.return_value = 42
    mock_session.execute = AsyncMock(return_value=mock_result)

    count = await repo.count_by_kb("kb-1")
    assert count == 42


# ------------------------------------------------------------------ #
# search_by_keyword
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_search_by_keyword(mock_session, repo):
    mock_result = MagicMock()
    mock_row = _make_mock_row({
        "id": "chunk-keyword-1",
        "content": "keyword content",
        "document_id": "doc-1",
        "kb_id": "kb-1",
        "seq_num": 1,
        "chunk_type": "text",
        "metadata": None,
    })
    mock_result.__iter__.return_value = [mock_row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.search_by_keyword("kb-1", "test", limit=10)

    assert len(results) == 1
    assert results[0].id == "chunk-keyword-1"
    assert results[0].content == "keyword content"


@pytest.mark.asyncio
async def test_search_by_keyword_empty(mock_session, repo):
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.search_by_keyword("kb-1", "nonexistent", limit=10)
    assert results == []


# ------------------------------------------------------------------ #
# search_faq
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_search_faq(mock_session, repo):
    mock_result = MagicMock()
    mock_row = _make_mock_row({
        "id": "faq-chunk-1",
        "kb_id": "kb-1",
        "document_id": "doc-1",
        "content": "FAQ answer content",
        "standard_question": "What is test?",
        "similar_questions": json.dumps(["What is test?", "Test question"]),
        "negative_questions": json.dumps(["Not a question"]),
        "answers": json.dumps(["This is the answer.", "Another answer."]),
        "category": "general",
    })
    mock_result.__iter__.return_value = [mock_row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.search_faq("kb-1", "What is test")

    assert len(results) == 1
    faq = results[0]
    assert isinstance(faq, FAQRow)
    assert faq.id == "faq-chunk-1"
    assert faq.standard_question == "What is test?"
    assert faq.similar_questions == ["What is test?", "Test question"]
    assert faq.negative_questions == ["Not a question"]
    assert faq.answers == ["This is the answer.", "Another answer."]
    assert faq.category == "general"


@pytest.mark.asyncio
async def test_search_faq_empty(mock_session, repo):
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.search_faq("kb-1", "nothing")
    assert results == []


@pytest.mark.asyncio
async def test_search_faq_uses_jsonb_cast(mock_session, repo):
    """search_faq 对 metadata 字段使用 ::jsonb 提取，避免 json 列 ensure_ascii 转义破坏中文 ILIKE。"""
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    await repo.search_faq("kb-1", "查询")

    sql = mock_session.execute.call_args[0][0].text
    assert "metadata::jsonb->>'similar_questions'" in sql
    assert "metadata::jsonb->>'standard_question'" in sql
    assert "metadata::jsonb->>'negative_questions'" in sql


# ------------------------------------------------------------------ #
# update_keywords
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_update_keywords(mock_session, repo):
    mock_session.execute.return_value = MagicMock()

    await repo.update_keywords("chunk-1", ["kw1", "kw2"])

    mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_keywords_empty_list(mock_session, repo):
    mock_session.execute.return_value = MagicMock()

    await repo.update_keywords("chunk-1", [])

    mock_session.execute.assert_awaited_once()


# ------------------------------------------------------------------ #
# update_summary
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_update_summary(mock_session, repo):
    mock_session.execute.return_value = MagicMock()

    await repo.update_summary("chunk-1", "new summary")

    mock_session.execute.assert_awaited_once()


# ------------------------------------------------------------------ #
# update_content_tokens
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_update_content_tokens(mock_session, repo):
    mock_session.execute.return_value = MagicMock()

    await repo.update_content_tokens("chunk-1")

    mock_session.execute.assert_awaited_once()
    sql = mock_session.execute.call_args[0][0].text
    assert "content_tokens" in sql
    assert "content" in sql
    assert "zh_cfg" in sql


# ------------------------------------------------------------------ #
# update_summary_tokens
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_update_summary_tokens(mock_session, repo):
    mock_session.execute.return_value = MagicMock()

    await repo.update_summary_tokens("chunk-1")

    mock_session.execute.assert_awaited_once()
    sql = mock_session.execute.call_args[0][0].text
    assert "summary_tokens" in sql
    assert "zh_cfg" in sql


# ------------------------------------------------------------------ #
# _safe_json_list
# ------------------------------------------------------------------ #


class TestSafeJsonList:
    def test_none_input(self):
        assert _safe_json_list(None) == []

    def test_empty_string(self):
        assert _safe_json_list("") == []

    def test_valid_json_list(self):
        assert _safe_json_list('["a", "b"]') == ["a", "b"]

    def test_already_a_list(self):
        assert _safe_json_list(["x", "y"]) == ["x", "y"]

    def test_invalid_json_string(self):
        assert _safe_json_list("not-json") == []

    def test_non_list_json(self):
        assert _safe_json_list('{"key": "val"}') == []

    def test_integer_input(self):
        assert _safe_json_list(42) == []


# ------------------------------------------------------------------ #
# get_by_ids
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_by_ids(mock_session, repo):
    """批量按 chunk_ids 查 chunk_text，返回 ChunkRow 列表。"""
    mock_result = MagicMock()
    mock_row = _make_mock_row({
        "id": "c1", "content": "content1", "document_id": "d1",
        "kb_id": "kb-1", "seq_num": 1, "chunk_type": "text", "metadata": None,
    })
    mock_result.__iter__.return_value = [mock_row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await repo.get_by_ids(["c1"])
    assert len(result) == 1
    assert result[0].id == "c1"
    assert result[0].content == "content1"
    assert mock_session.execute.called


@pytest.mark.asyncio
async def test_get_by_ids_empty(repo):
    """空 chunk_ids 列表直接返回 []，不查库。"""
    result = await repo.get_by_ids([])
    assert result == []


# ------------------------------------------------------------------ #
# count_by_document
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_count_by_document(mock_session, repo):
    mock_result = MagicMock()
    mock_result.scalar.return_value = 5
    mock_session.execute.return_value = mock_result

    count = await repo.count_by_document("doc-1")

    assert count == 5
    mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_by_document_empty(mock_session, repo):
    mock_result = MagicMock()
    mock_result.scalar.return_value = None
    mock_session.execute.return_value = mock_result

    count = await repo.count_by_document("doc-1")

    assert count == 0
