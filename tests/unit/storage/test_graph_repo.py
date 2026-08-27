"""GraphMetadataRepository 单元测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aion_knowledge.storage.relational.graph_repo import GraphMetadataRepository


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.fixture
def repo(mock_session):
    return GraphMetadataRepository(mock_session)


@pytest.mark.asyncio
async def test_upsert_stats(mock_session, repo):
    mock_session.execute = AsyncMock()
    await repo.upsert_stats("kb-1", 10, 5)
    assert mock_session.execute.called


@pytest.mark.asyncio
async def test_upsert_stats_with_doc_and_community_counts(mock_session, repo):
    """doc_count/community_count 应随 upsert 写入并在冲突时更新（而非保留旧值 0）。"""
    mock_session.execute = AsyncMock()
    await repo.upsert_stats("kb-1", 10, 5, doc_count=1, community_count=15)

    params = mock_session.execute.call_args[0][1]
    assert params["doc_count"] == 1
    assert params["community_count"] == 15
    sql = mock_session.execute.call_args[0][0].text
    assert "doc_count = :doc_count2" in sql
    assert "community_count = :community_count2" in sql


@pytest.mark.asyncio
async def test_save_checkpoint(mock_session, repo):
    mock_session.execute = AsyncMock()
    await repo.save_checkpoint("kb-1", "community", {"version": 1})
    assert mock_session.execute.called


@pytest.mark.asyncio
async def test_save_checkpoint_creates_row_when_missing(mock_session, repo):
    """检查点保存应为 INSERT ON CONFLICT（graph_metadata 行不存在时自建，不静默丢失）。"""
    mock_session.execute = AsyncMock()
    await repo.save_checkpoint("kb-1", "community", {"version": 1})

    sql = mock_session.execute.call_args[0][0].text
    params = mock_session.execute.call_args[0][1]
    assert params["key"] == "community"
    assert params["path"] == ["community"]
    assert "ON CONFLICT (kb_id)" in sql


@pytest.mark.asyncio
async def test_save_checkpoint_key_and_path_params_separate(mock_session, repo):
    """:key（jsonb_build_object 键）与 :path（jsonb_set 路径）必须分开绑定。

    回归：曾把 :key 同时用于 CAST AS text 与 CAST AS text[]，asyncpg 按 Python 值解析
    codec（list→数组），导致 CAST AS text 报 DataError: invalid input ... (expected str,
    got list)。故 :key 绑 str，:path 绑 list，且 SQL 中二者只出现在类型匹配的 CAST 位置。
    """
    mock_session.execute = AsyncMock()
    await repo.save_checkpoint("kb-1", "disambiguation", {"graph_hash": "abc"})

    sql = mock_session.execute.call_args[0][0].text
    params = mock_session.execute.call_args[0][1]

    # 参数类型契约：键是 str，路径是 list
    assert params["key"] == "disambiguation"
    assert params["path"] == ["disambiguation"]

    # SQL 位置契约：每个参数只出现在类型匹配的 CAST 中
    assert "CAST(:key AS text)" in sql       # jsonb_build_object 键
    assert "CAST(:path AS text[])" in sql    # jsonb_set 路径
    assert "CAST(:key AS text[])" not in sql  # 回归点：:key 不得用于数组位置
    assert "CAST(:path AS text)" not in sql


@pytest.mark.asyncio
async def test_save_doc_checkpoint_nested_path(mock_session, repo):
    """save_doc_checkpoint 写 checkpoints.<module>.docs.<doc_id>，中间路径缺失也能创建（|| 拼接）。"""
    mock_session.execute = AsyncMock()
    await repo.save_doc_checkpoint("kb-1", "wiki", "doc-1", {"status": "completed"})

    sql = mock_session.execute.call_args[0][0].text
    params = mock_session.execute.call_args[0][1]
    assert params["module"] == "wiki"
    assert params["doc_id"] == "doc-1"
    assert "jsonb_build_object" in sql
    assert "ON CONFLICT (kb_id)" in sql


@pytest.mark.asyncio
async def test_get_checkpoint(mock_session, repo):
    """get_checkpoint 返回嵌套路径下的完整 JSON 值。"""
    mock_result = MagicMock()
    mock_result.scalar.return_value = '{"status": "completed", "page_count": 3}'
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await repo.get_checkpoint("kb-1", ["wiki", "docs", "doc-1"])
    assert result == '{"status": "completed", "page_count": 3}'
    sql = mock_session.execute.call_args[0][0].text
    assert "#>" in sql


@pytest.mark.asyncio
async def test_load_checkpoint(mock_session, repo):
    mock_result = MagicMock()
    mock_result.scalar.return_value = "abc123"
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await repo.load_checkpoint("kb-1", "community")
    assert result == "abc123"


@pytest.mark.asyncio
async def test_load_checkpoint_not_found(mock_session, repo):
    mock_result = MagicMock()
    mock_result.scalar.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await repo.load_checkpoint("kb-1", "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_clear_checkpoint(mock_session, repo):
    mock_session.execute = AsyncMock()
    await repo.clear_checkpoint("kb-1", "community")
    assert mock_session.execute.called


@pytest.mark.asyncio
async def test_get_communities(mock_session, repo):
    mock_result = MagicMock()
    mock_row = ("comm-1", "summary text", '{"finding": "test"}', "Title")
    mock_result.__iter__.return_value = [mock_row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.get_communities("kb-1")
    assert len(results) == 1
    assert results[0]["community_id"] == "comm-1"
    assert results[0]["summary"] == "summary text"


@pytest.mark.asyncio
async def test_get_communities_empty(mock_session, repo):
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await repo.get_communities("kb-1")
    assert results == []
