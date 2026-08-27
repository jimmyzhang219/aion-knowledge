"""DeletionService 单元测试：逻辑删除模式（默认）。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.ingestion.deletion import delete_document, delete_kb

_UUID_KB = "00000000-0000-0000-0000-000000000001"
_UUID_DOC = "00000000-0000-0000-0000-000000000002"
_UUID_MISSING = "00000000-0000-0000-0000-0000000000ff"


@pytest.mark.asyncio
async def test_delete_document_marks_deleted_and_commits():
    """逻辑删除：置 deleted=true 并提交，不起 purge。"""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.first.return_value = (_UUID_DOC,)  # 存在
    mock_session.execute.return_value = mock_result

    with (
        patch("aion_knowledge.ingestion.deletion.get_session") as mock_gs,
        patch("aion_knowledge.ingestion.deletion.settings") as mock_settings,
        patch("aion_knowledge.ingestion.deletion.asyncio.create_task") as mock_ct,
    ):
        mock_gs.return_value.__aenter__.return_value = mock_session
        mock_settings.deletion_logical = True

        ok = await delete_document(_UUID_KB, _UUID_DOC)

    assert ok is True
    calls = [c.args[0].text for c in mock_session.execute.await_args_list]
    assert any("UPDATE doc_knowledge_documents SET deleted = true" in c for c in calls)
    mock_session.commit.assert_awaited_once()
    mock_ct.assert_not_called()


@pytest.mark.asyncio
async def test_delete_document_not_found():
    """文档不存在返回 False，不置标记。"""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_session.execute.return_value = mock_result

    with (
        patch("aion_knowledge.ingestion.deletion.get_session") as mock_gs,
        patch("aion_knowledge.ingestion.deletion.settings") as mock_settings,
        patch("aion_knowledge.ingestion.deletion.asyncio.create_task") as mock_ct,
    ):
        mock_gs.return_value.__aenter__.return_value = mock_session
        mock_settings.deletion_logical = True

        ok = await delete_document(_UUID_KB, _UUID_MISSING)

    assert ok is False
    mock_session.commit.assert_not_awaited()
    mock_ct.assert_not_called()


@pytest.mark.asyncio
async def test_delete_document_physical_mode_starts_purge_task():
    """deletion_logical=False 时置标记后起异步 purge 协程。"""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.first.return_value = (_UUID_DOC,)
    mock_session.execute.return_value = mock_result

    with (
        patch("aion_knowledge.ingestion.deletion.get_session") as mock_gs,
        patch("aion_knowledge.ingestion.deletion.settings") as mock_settings,
        patch("aion_knowledge.ingestion.deletion.asyncio.create_task") as mock_ct,
        patch(
            "aion_knowledge.ingestion.deletion._purge_document", new_callable=MagicMock
        ) as mock_purge,
    ):
        mock_gs.return_value.__aenter__.return_value = mock_session
        mock_settings.deletion_logical = False

        await delete_document(_UUID_KB, _UUID_DOC)

    mock_ct.assert_called_once()
    mock_purge.assert_called_once_with(_UUID_KB, _UUID_DOC)


@pytest.mark.asyncio
async def test_delete_kb_marks_kb_and_all_docs():
    """KB 删除：KB 与 KB 下所有文档置 deleted=true。"""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.first.return_value = (_UUID_KB,)
    mock_session.execute.return_value = mock_result

    with (
        patch("aion_knowledge.ingestion.deletion.get_session") as mock_gs,
        patch("aion_knowledge.ingestion.deletion.settings") as mock_settings,
        patch("aion_knowledge.ingestion.deletion.asyncio.create_task") as mock_ct,
    ):
        mock_gs.return_value.__aenter__.return_value = mock_session
        mock_settings.deletion_logical = True

        ok = await delete_kb(_UUID_KB)

    assert ok is True
    calls = [c.args[0].text for c in mock_session.execute.await_args_list]
    assert any("UPDATE kb_knowledge_bases SET deleted = true" in c for c in calls)
    assert any("UPDATE doc_knowledge_documents SET deleted = true WHERE kb_id" in c for c in calls)
    mock_ct.assert_not_called()


@pytest.mark.asyncio
async def test_purge_document_deletes_all_doc_data():
    """文档级真实删除：chunk_vector→chunk_text→raptor→disambiguation→task→文档行，并清 Neo4j/文件/checkpoint。"""
    from aion_knowledge.ingestion.deletion import _purge_document

    mock_session = AsyncMock()
    mock_row = MagicMock()
    mock_row.first.return_value = ("s3://bucket/docs/doc-1/orig.pdf",)
    mock_session.execute.return_value = mock_row

    with (
        patch("aion_knowledge.ingestion.deletion.get_session") as mock_gs,
        patch("aion_knowledge.infrastructure.graph.delete_document_graph") as mock_gr,
        patch("aion_knowledge.infrastructure.storage.resolve_storage") as mock_rs,
        patch("aion_knowledge.indexing.checkpoint.CheckpointManager.delete") as mock_cp,
    ):
        mock_gs.return_value.__aenter__.return_value = mock_session
        mock_store = AsyncMock()
        mock_rs.return_value = mock_store

        await _purge_document(_UUID_KB, _UUID_DOC)

    calls = [c.args[0].text for c in mock_session.execute.await_args_list]
    assert any("DELETE FROM chunk_vector" in c for c in calls)
    assert any("DELETE FROM chunk_text WHERE document_id = :doc AND kb_id = :kb" in c for c in calls)
    assert any("DELETE FROM chunk_raptor" in c for c in calls)
    assert any("DELETE FROM chunk_disambiguation" in c for c in calls)
    assert any("DELETE FROM task_ingestion_tasks" in c for c in calls)
    # 主行物理删收尾：chunk 系事务 + 主行单独事务两次 commit，主行 DELETE 是最后一条 SQL
    assert mock_session.commit.await_count == 2
    assert calls[-1] == "DELETE FROM doc_knowledge_documents WHERE id = :doc AND kb_id = :kb"
    mock_gr.assert_awaited_once_with(_UUID_KB, _UUID_DOC)
    mock_store.delete.assert_awaited_once()
    mock_cp.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_kb_deletes_kb_level_data():
    """KB 级真实删除：文档循环 + community/wiki/raptor/disambiguation/graph_metadata/KB 行 + Neo4j。"""
    from aion_knowledge.ingestion.deletion import _purge_kb

    mock_session = AsyncMock()
    mock_rows = MagicMock()
    mock_rows.fetchall.return_value = [("doc-1",), ("doc-2",)]
    mock_session.execute.return_value = mock_rows

    with (
        patch("aion_knowledge.ingestion.deletion.get_session") as mock_gs,
        patch("aion_knowledge.ingestion.deletion._purge_document") as mock_pd,
        patch("aion_knowledge.infrastructure.graph.delete_graph") as mock_dg,
    ):
        mock_gs.return_value.__aenter__.return_value = mock_session

        await _purge_kb(_UUID_KB)

    assert mock_pd.await_count == 2
    calls = [c.args[0].text for c in mock_session.execute.await_args_list]
    assert any("DELETE FROM chunk_community" in c for c in calls)
    assert any("DELETE FROM chunk_wiki" in c for c in calls)
    assert any("DELETE FROM chunk_raptor WHERE kb_id" in c for c in calls)
    assert any("DELETE FROM graph_metadata" in c for c in calls)
    # KB 主行收尾：单独事务、最后一条 SQL（Neo4j 失败时主行仍在，可重删重试）
    assert calls[-1] == "DELETE FROM kb_knowledge_bases WHERE id = :kb"
    mock_dg.assert_awaited_once_with(_UUID_KB)


@pytest.mark.asyncio
async def test_purge_document_neo4j_failure_continues_external_cleanup():
    """异常路径：Neo4j 删除失败不中断，对象存储与 checkpoint 仍被清理。"""
    from aion_knowledge.ingestion.deletion import _purge_document

    mock_session = AsyncMock()
    mock_row = MagicMock()
    mock_row.first.return_value = ("s3://bucket/docs/doc-1/orig.pdf",)
    mock_session.execute.return_value = mock_row

    with (
        patch("aion_knowledge.ingestion.deletion.get_session") as mock_gs,
        patch(
            "aion_knowledge.infrastructure.graph.delete_document_graph",
            side_effect=RuntimeError("neo4j down"),
        ) as mock_gr,
        patch("aion_knowledge.infrastructure.storage.resolve_storage") as mock_rs,
        patch("aion_knowledge.indexing.checkpoint.CheckpointManager.delete") as mock_cp,
    ):
        mock_gs.return_value.__aenter__.return_value = mock_session
        mock_store = AsyncMock()
        mock_rs.return_value = mock_store

        await _purge_document(_UUID_KB, _UUID_DOC)  # 不应抛出

    mock_gr.assert_awaited_once_with(_UUID_KB, _UUID_DOC)
    mock_store.delete.assert_awaited_once()
    mock_cp.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_document_idempotent_when_doc_already_gone():
    """幂等：文档行已不存在（重复 purge）时 file_path 空跳过存储删除，DELETE 全部执行不抛错。"""
    from aion_knowledge.ingestion.deletion import _purge_document

    mock_session = AsyncMock()
    mock_row = MagicMock()
    mock_row.first.return_value = None  # 文档行已删（重复执行）
    mock_session.execute.return_value = mock_row

    with (
        patch("aion_knowledge.ingestion.deletion.get_session") as mock_gs,
        patch("aion_knowledge.infrastructure.graph.delete_document_graph"),
        patch("aion_knowledge.infrastructure.storage.resolve_storage") as mock_rs,
        patch("aion_knowledge.indexing.checkpoint.CheckpointManager.delete"),
    ):
        mock_gs.return_value.__aenter__.return_value = mock_session
        mock_store = AsyncMock()
        mock_rs.return_value = mock_store

        await _purge_document(_UUID_KB, _UUID_DOC)  # 不应抛出

    mock_store.delete.assert_not_awaited()
    calls = [c.args[0].text for c in mock_session.execute.await_args_list]
    assert any("DELETE FROM chunk_vector" in c for c in calls)
    assert any("DELETE FROM chunk_text" in c for c in calls)
    assert any("DELETE FROM chunk_raptor" in c for c in calls)
    assert any("DELETE FROM chunk_disambiguation" in c for c in calls)
    assert any("DELETE FROM task_ingestion_tasks" in c for c in calls)
    assert any("DELETE FROM doc_knowledge_documents" in c for c in calls)
