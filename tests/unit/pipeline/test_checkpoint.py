"""Checkpoint 持久化与恢复测试。"""

from __future__ import annotations

import uuid

import pytest

from aion_knowledge.indexing.checkpoint import Checkpoint, CheckpointManager


class TestCheckpoint:
    def test_create_checkpoint(self):
        cp = Checkpoint(
            document_id=str(uuid.uuid4()),
            component="parser",
            status="processing",
        )
        assert cp.component == "parser"
        assert cp.retry_count == 0
        assert cp.progress_pct == 0.0

    def test_checkpoint_increment_retry(self):
        cp = Checkpoint(
            document_id=str(uuid.uuid4()),
            component="chunker",
            status="processing",
        )
        cp.increment_retry()
        assert cp.retry_count == 1
        cp.increment_retry()
        assert cp.retry_count == 2

    def test_checkpoint_mark_done(self):
        cp = Checkpoint(
            document_id=str(uuid.uuid4()),
            component="parser",
            status="processing",
        )
        cp.mark_done()
        assert cp.status == "completed"
        assert cp.progress_pct == 100.0


class TestCheckpointManager:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        doc_id = str(uuid.uuid4())
        mgr = CheckpointManager()
        cp = Checkpoint(document_id=doc_id, component="parser", status="processing")
        await mgr.save(doc_id, cp)
        loaded = await mgr.load(doc_id)
        assert loaded is not None
        assert loaded.component == "parser"
        assert loaded.status == "processing"

    @pytest.mark.asyncio
    async def test_load_missing(self):
        mgr = CheckpointManager()
        result = await mgr.load(str(uuid.uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        doc_id = str(uuid.uuid4())
        mgr = CheckpointManager()
        cp = Checkpoint(document_id=doc_id, component="dual_write", status="processing")
        await mgr.save(doc_id, cp)
        await mgr.delete(doc_id)
        loaded = await mgr.load(doc_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_stale(self):
        mgr = CheckpointManager()
        doc_id = str(uuid.uuid4())
        cp = Checkpoint(document_id=doc_id, component="parser", status="processing")
        await mgr.save(doc_id, cp)
        stale = await mgr.list_stale(max_age_seconds=0)
        assert doc_id in stale
