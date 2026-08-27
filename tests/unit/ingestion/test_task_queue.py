"""Task queue 数据模型测试。"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from aion_knowledge.infrastructure.models import PostProcConfig, PostProcTask, UnifiedContext


class TestUnifiedContext:
    def test_create_minimal(self):
        ctx = UnifiedContext(
            context_id=str(uuid.uuid4()),
            source="regular",
            kb_id=str(uuid.uuid4()),
            doc_name="test.pdf",
            suffix="pdf",
            original_file_ref="s3://bucket/key.pdf",
        )
        assert ctx.chunk_strategy == "auto"
        assert ctx.ext_metadata == {}
        assert ctx.created_at != ""

    def test_create_with_all_fields(self, sample_unified_context_dict):
        ctx = UnifiedContext(**sample_unified_context_dict)
        assert ctx.source == "regular"
        assert ctx.chunk_strategy == "auto"

    def test_created_at_defaults_to_iso(self):
        ctx = UnifiedContext(
            context_id=str(uuid.uuid4()),
            source="manual_entry",
            kb_id=str(uuid.uuid4()),
            doc_name="note.md",
            suffix="md",
            original_file_ref="s3://bucket/key.md",
        )
        # should be an ISO datetime string
        datetime.fromisoformat(ctx.created_at)

    def test_required_fields_missing(self):
        with pytest.raises(ValidationError):
            UnifiedContext()  # type: ignore[call-arg]


class TestPostProcConfig:
    def test_defaults(self):
        cfg = PostProcConfig()
        assert cfg.enable_keyword_extract is True
        assert cfg.enable_question_gen is True
        assert cfg.enable_summarizer is True
        assert cfg.enable_raptor is True
        # 图谱系出厂硬控已全部开启（社区发现依赖消歧，均随图谱启用）
        assert cfg.enable_graph_extract is True
        assert cfg.enable_community is True
        assert cfg.enable_disambiguation is True
        assert cfg.enable_wiki is True

    def test_enable_specific(self, sample_postproc_config_dict):
        cfg = PostProcConfig(**sample_postproc_config_dict)
        assert cfg.enable_keyword_extract is True
        assert cfg.enable_question_gen is False


class TestPostProcTask:
    def test_create(self, sample_postproc_config_dict):
        task = PostProcTask(
            document_id=str(uuid.uuid4()),
            kb_id=str(uuid.uuid4()),
            doc_name="test.pdf",
            chunk_count=10,
            postproc_config=PostProcConfig(**sample_postproc_config_dict),
        )
        assert task.chunk_count == 10
        assert task.postproc_config.enable_keyword_extract is True
        assert task.created_at != ""

    def test_required_fields_missing(self):
        with pytest.raises(ValidationError):
            PostProcTask()  # type: ignore[call-arg]


class TestWorkerLoop:
    """验证 worker 循环能正确拉取消息并调用 IndexingExecutor。"""

    @pytest.mark.asyncio
    async def test_queue1_put_get(self):
        """验证 Queue1 的基本 put/get 功能。"""
        from aion_knowledge.infrastructure.models import UnifiedContext
        from aion_knowledge.infrastructure.queues import ctx_queue

        ctx = UnifiedContext(
            context_id="test-id",
            source="regular",
            kb_id="kb-1",
            doc_name="test.pdf",
            suffix="pdf",
            original_file_ref="s3://bucket/key.pdf",
        )
        await ctx_queue.put(ctx)
        got = await ctx_queue.get()
        assert got.context_id == "test-id"
        assert got.source == "regular"
        ctx_queue.task_done()

    @pytest.mark.asyncio
    async def test_queue2_put_get(self):
        """验证 Queue2 的基本 put/get 功能。"""
        from aion_knowledge.infrastructure.models import PostProcConfig, PostProcTask
        from aion_knowledge.infrastructure.queues import postproc_queue

        task = PostProcTask(
            document_id="doc-1",
            kb_id="kb-1",
            doc_name="test.pdf",
            chunk_count=5,
            postproc_config=PostProcConfig(),
        )
        await postproc_queue.put(task)
        got = await postproc_queue.get()
        assert got.document_id == "doc-1"
        postproc_queue.task_done()

    @pytest.mark.asyncio
    async def test_clear_queues(self):
        """清除队列中的残留消息（用于测试隔离）。"""
        from aion_knowledge.infrastructure.queues import ctx_queue, postproc_queue

        while not ctx_queue.empty():
            await ctx_queue.get()
            ctx_queue.task_done()
        while not postproc_queue.empty():
            await postproc_queue.get()
            postproc_queue.task_done()
        assert ctx_queue.empty()
        assert postproc_queue.empty()
