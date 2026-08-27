"""WikiCheckpointManager 测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.pipeline.postproc.wiki.checkpoint import WikiCheckpointManager


def _mock_session():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    return mock_session


@pytest.mark.asyncio
async def test_save_uses_nested_per_doc_key():
    """wiki 检查点按 (kb, doc) 记录，写入 graph_metadata.checkpoints.wiki.docs.<doc_id>。"""
    with patch("aion_knowledge.pipeline.postproc.wiki.checkpoint.get_session") as mock_get_session, \
         patch("aion_knowledge.pipeline.postproc.wiki.checkpoint.GraphMetadataRepository") as mock_repo_class:
        mock_get_session.return_value = _mock_session()
        mock_repo = mock_repo_class.return_value
        mock_repo.save_doc_checkpoint = AsyncMock()

        await WikiCheckpointManager().save(
            "kb1", "d1", status="completed", page_count=3, candidate_count=5,
        )

    args = mock_repo.save_doc_checkpoint.await_args
    assert args.args[0] == "kb1"
    assert args.args[1] == "wiki"
    assert args.args[2] == "d1"
    data = args.args[3]
    assert data["status"] == "completed"
    assert data["page_count"] == 3
    assert data["candidate_count"] == 5
    assert "completed_at" in data


@pytest.mark.asyncio
async def test_load_reads_nested_key():
    with patch("aion_knowledge.pipeline.postproc.wiki.checkpoint.get_session") as mock_get_session, \
         patch("aion_knowledge.pipeline.postproc.wiki.checkpoint.GraphMetadataRepository") as mock_repo_class:
        mock_get_session.return_value = _mock_session()
        mock_repo = mock_repo_class.return_value
        mock_repo.get_checkpoint = AsyncMock(return_value={"status": "no_candidates"})

        result = await WikiCheckpointManager().load("kb1", "d1")

    assert result == {"status": "no_candidates"}
    mock_repo.get_checkpoint.assert_awaited_once_with("kb1", ["wiki", "docs", "d1"])


@pytest.mark.asyncio
async def test_load_missing_returns_none():
    with patch("aion_knowledge.pipeline.postproc.wiki.checkpoint.get_session") as mock_get_session, \
         patch("aion_knowledge.pipeline.postproc.wiki.checkpoint.GraphMetadataRepository") as mock_repo_class:
        mock_get_session.return_value = _mock_session()
        mock_repo = mock_repo_class.return_value
        mock_repo.get_checkpoint = AsyncMock(return_value=None)

        result = await WikiCheckpointManager().load("kb1", "d1")

    assert result is None
