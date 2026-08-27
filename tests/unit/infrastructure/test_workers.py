"""测试 workers 编排函数 — 二批 chunks 数据契约 + trace_id 注入。"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.common.trace import get_trace_id
from aion_knowledge.infrastructure.models import PostProcConfig, PostProcTask, UnifiedContext
from aion_knowledge.infrastructure.queues import ctx_queue, postproc_queue
from aion_knowledge.infrastructure.workers import (
    _run_postproc_subtasks,
    pipeline_worker,
    postproc_worker,
)


class TestWorkerTraceId:
    """验证 worker 在消费队列消息时注入 trace_id 到 contextvar 日志上下文。"""

    @pytest.mark.asyncio
    async def test_pipeline_worker_logs_with_ctx_trace_id(self):
        """pipeline_worker 消费后：执行器内 get_trace_id() == ctx.trace_id，消费完 reset。"""
        ctx = UnifiedContext(
            source="regular", kb_id="kb-1", doc_name="d.md", suffix="md",
            original_file_ref="s3://x", trace_id="ctx-trace-9",
        )
        processed = asyncio.Event()
        seen: list[str] = []

        async def fake_run(_ctx):
            seen.append(get_trace_id())
            processed.set()
            return PostProcTask(
                document_id="doc-1", kb_id="kb-1", doc_name="d.md", chunk_count=1,
                postproc_config=PostProcConfig(),
            )

        with (
            patch("aion_knowledge.indexing.executor.IndexingExecutor",
                  return_value=AsyncMock(run=fake_run)),
            patch("aion_knowledge.infrastructure.queues.postproc_queue.put", AsyncMock()),
        ):
            worker = asyncio.create_task(pipeline_worker())
            await ctx_queue.put(ctx)
            try:
                await asyncio.wait_for(processed.wait(), timeout=5)
            finally:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

        assert seen == ["ctx-trace-9"]
        assert get_trace_id() != "ctx-trace-9"  # 消费完已 reset

    @pytest.mark.asyncio
    async def test_postproc_worker_logs_with_task_trace_id(self):
        """postproc_worker 消费后：子任务内 get_trace_id() == task.trace_id。"""
        task = PostProcTask(
            document_id="doc-2", kb_id="kb-1", doc_name="t.md", chunk_count=1,
            postproc_config=PostProcConfig(), trace_id="task-trace-8",
        )
        processed = asyncio.Event()
        seen: list[str] = []

        async def fake_subtasks(_task):
            seen.append(get_trace_id())
            processed.set()

        with patch("aion_knowledge.infrastructure.workers._run_postproc_subtasks",
                   fake_subtasks):
            worker = asyncio.create_task(postproc_worker())
            await postproc_queue.put(task)
            try:
                await asyncio.wait_for(processed.wait(), timeout=5)
            finally:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

        assert seen == ["task-trace-8"]
        assert get_trace_id() != "task-trace-8"


class TestRunPostprocSubtasks:
    @pytest.mark.asyncio
    async def test_chunk_uuid_is_str(self):
        """二批从 chunk_text 重读的 chunk_uuid 必须是 str，与首批契约一致。

        回归场景：wiki 模块按字符串契约执行 uuid.UUID(source_chunk_uuid)，
        若二批传入 UUID 对象会抛 AttributeError 致 chunk_wiki 零写入。
        """
        task = PostProcTask(
            document_id=str(uuid.uuid4()),
            kb_id=str(uuid.uuid4()),
            doc_name="t.md",
            chunk_count=2,
            postproc_config=PostProcConfig(),
        )
        fake_rows = [
            SimpleNamespace(id=uuid.uuid4(), content="机器学习是人工智能的一个分支。", seq_num=1, chunk_type="text", chunk_metadata={}),
            SimpleNamespace(id=uuid.uuid4(), content="深度学习是机器学习的子领域。", seq_num=2, chunk_type="text", chunk_metadata={}),
        ]

        session = AsyncMock()
        session.__aenter__.return_value = session
        session.__aexit__ = AsyncMock(return_value=False)
        # 文档存在性检查（get_document_by_id → session.execute）返回 truthy 结果
        session.execute.return_value = MagicMock()

        repo = AsyncMock()
        repo.get_by_document.return_value = fake_rows

        dispatcher = AsyncMock()

        with (
            patch("aion_knowledge.infrastructure.db.get_session", return_value=session),
            patch("aion_knowledge.storage.relational.chunk_repo.ChunkRepository", return_value=repo),
            patch("aion_knowledge.pipeline.postproc.dispatcher.PostProcDispatcher", return_value=dispatcher),
        ):
            await _run_postproc_subtasks(task)

        dispatcher.run_second_batch.assert_awaited_once()
        ctx, chunks = dispatcher.run_second_batch.call_args.args
        assert len(chunks) == 2
        assert all(isinstance(c["chunk_uuid"], str) for c in chunks)
        # chunk_uuid 必须是合法 UUID 字符串（PG uuid 列可绑定）
        assert all(uuid.UUID(c["chunk_uuid"]) for c in chunks)

    @pytest.mark.asyncio
    async def test_factory_gate_and_settings(self):
        """出厂硬控 AND settings：最终生效 = enable_* AND settings.postproc_*。"""
        from aion_knowledge.common.config import settings

        task = PostProcTask(
            document_id=str(uuid.uuid4()),
            kb_id=str(uuid.uuid4()),
            doc_name="t.md",
            chunk_count=1,
            postproc_config=PostProcConfig(),  # 出厂默认：全部 true
        )
        fake_rows = [
            SimpleNamespace(id=uuid.uuid4(), content="测试内容", seq_num=1,
                            chunk_type="text", chunk_metadata={}),
        ]

        session = AsyncMock()
        session.__aenter__.return_value = session
        session.__aexit__ = AsyncMock(return_value=False)
        # 文档存在性检查（get_document_by_id → session.execute）返回 truthy 结果
        session.execute.return_value = MagicMock()

        repo = AsyncMock()
        repo.get_by_document.return_value = fake_rows

        dispatcher = AsyncMock()

        with (
            patch("aion_knowledge.infrastructure.db.get_session", return_value=session),
            patch("aion_knowledge.storage.relational.chunk_repo.ChunkRepository", return_value=repo),
            patch("aion_knowledge.pipeline.postproc.dispatcher.PostProcDispatcher",
                  return_value=dispatcher) as disp_cls,
            patch.object(settings, "postproc_graph_extract", True),
            patch.object(settings, "postproc_keyword_extract", True),
            patch.object(settings, "postproc_wiki", False),
        ):
            await _run_postproc_subtasks(task)

        settings_dict = disp_cls.call_args.args[0]
        # 出厂 true AND settings true → 启用
        assert settings_dict["graph_extract"] is True
        # 出厂 true AND settings true → 启用
        assert settings_dict["keyword_extract"] is True
        # 出厂 true AND settings false → 禁用
        assert settings_dict["wiki"] is False

    @pytest.mark.asyncio
    async def test_skips_when_document_deleted_or_missing(self):
        """文档已删除/不存在：跳过整个任务，不加载 chunk、不跑二批。"""
        task = PostProcTask(
            document_id=str(uuid.uuid4()),
            kb_id=str(uuid.uuid4()),
            doc_name="t.md",
            chunk_count=1,
            postproc_config=PostProcConfig(),
        )
        session = AsyncMock()
        session.__aenter__.return_value = session
        session.__aexit__ = AsyncMock(return_value=False)

        dispatcher = AsyncMock()

        with (
            patch("aion_knowledge.infrastructure.db.get_session", return_value=session),
            patch("aion_knowledge.ingestion.document_repo.get_document_by_id",
                  AsyncMock(return_value=None)),
            patch("aion_knowledge.storage.relational.chunk_repo.ChunkRepository") as repo_cls,
            patch("aion_knowledge.pipeline.postproc.dispatcher.PostProcDispatcher",
                  return_value=dispatcher),
        ):
            await _run_postproc_subtasks(task)

        repo_cls.assert_not_called()
        dispatcher.run_second_batch.assert_not_called()
