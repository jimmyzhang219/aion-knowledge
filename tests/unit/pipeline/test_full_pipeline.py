"""全链路端到端集成测试：上传 → Queue1 → IndexingExecutor → Queue2。"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import delete, select

from aion_knowledge.indexing.executor import IndexingExecutor
from aion_knowledge.infrastructure.models import PostProcTask, UnifiedContext
from aion_knowledge.infrastructure.queues import ctx_queue, postproc_queue
from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy
from aion_knowledge.pipeline.postproc.text.orm import ChunkText

_TEST_KB_ID = "00000000-0000-0000-0000-000000000002"


@pytest.fixture(autouse=True)
async def clear_queues():
    """每个测试前后清理队列，避免测试间污染。"""
    for q in (ctx_queue, postproc_queue):
        while not q.empty():
            await q.get()
            q.task_done()
    yield
    for q in (ctx_queue, postproc_queue):
        while not q.empty():
            await q.get()
            q.task_done()


@pytest.fixture(autouse=True)
async def cleanup_db():
    """清理测试产生的 DB 记录（仅限测试 KB，禁止无 WHERE 全表删除）。

    曾因无 WHERE 删除误清真实环境数据（2026-08-02 事故），
    现按 _TEST_KB_ID 限定，绝不触碰其他 KB 的数据。
    """
    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.models.orm import IngestionTask, KnowledgeDocument
    from aion_knowledge.pipeline.postproc.vector.orm import ChunkVector

    yield

    async with get_session() as session:
        await session.execute(delete(ChunkText).where(ChunkText.kb_id == _TEST_KB_ID))
        # 首批 vector 模块也会写真实 chunk_vector，需一并清理
        await session.execute(delete(ChunkVector).where(ChunkVector.kb_id == _TEST_KB_ID))
        # IngestionTask 无 kb_id，按该 KB 的文档 id 关联删除
        doc_ids = (
            await session.execute(
                select(KnowledgeDocument.id).where(KnowledgeDocument.kb_id == _TEST_KB_ID)
            )
        ).scalars().all()
        if doc_ids:
            await session.execute(
                delete(IngestionTask).where(IngestionTask.document_id.in_(doc_ids))
            )
        await session.execute(
            delete(KnowledgeDocument).where(KnowledgeDocument.kb_id == _TEST_KB_ID)
        )
        # get_session 自动 commit


@pytest.mark.asyncio
async def test_full_pipeline(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """全链路：RegularIngestionStrategy → Queue1 → IndexingExecutor → Queue2 → log 完成。

    使用真实 InMemoryChunkStore（B6 修复后可用），
    仅 Mock _save_to_s3 控制文件落盘位置。
    """
    caplog.set_level(logging.INFO)

    # ── 准备 ────────────────────────────────────────────────────
    md_content = (
        b"# Introduction\n\n"
        b"This is a test markdown document for the pipeline.\n\n"
        b"## Section One\n\n"
        b"Content in the first section.\n\n"
        b"More details in section one.\n\n"
        b"## Section Two\n\n"
        b"Content in the second section.\n\n"
        b"Even more details in section two.\n\n"
        b"### Subsection\n\n"
        b"Deep content here.\n\n"
        b"## Section Three\n\n"
        b"Final section content.\n\n"
    )

    # Mock _save_to_s3：把文件存到本地临时目录，返回本地路径
    async def _mock_save_to_storage(content: bytes, s3_key: str) -> str:
        local_path = str(tmp_path / f"s3_{s3_key.replace('/', '_')}")
        Path(local_path).write_bytes(content)
        return local_path

    # ── Steps ────────────────────────────────────────────────────
    mock_doc = AsyncMock()
    mock_doc.id = "00000000-0000-0000-0000-000000000099"
    mock_task = AsyncMock()
    mock_task.id = "00000000-0000-0000-0000-000000000098"
    patches = [
        patch("aion_knowledge.ingestion.strategy.base.ensure_kb_exists", AsyncMock()),
        patch("aion_knowledge.ingestion.strategy.base.save_to_storage", _mock_save_to_storage),
        patch("aion_knowledge.ingestion.strategy.base.extract_storage_key", side_effect=lambda x: x),
        patch("aion_knowledge.ingestion.strategy.base.create_document", AsyncMock(return_value=mock_doc)),
        patch("aion_knowledge.ingestion.strategy.base.get_session"),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.get_session"),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.get_document_by_hash", AsyncMock(return_value=None)),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.create_ingestion_task", AsyncMock(return_value=mock_task)),
    ]
    for p in patches:
        p.start()

    try:
        # Step 1: 文件上传入队
        strategy = RegularIngestionStrategy(suffix="md", source="regular")
        result = await strategy.execute(
            kb_id=_TEST_KB_ID,
            content=md_content,
            file_name="test.md",
            creator="test",
        )
        assert result["status"] == "queued"
        context_id = result["context_id"]
        assert context_id is not None

        # Step 2: 从 Queue1 取出 UnifiedContext
        ctx: UnifiedContext = await ctx_queue.get()
        assert ctx.source == "regular"
        assert ctx.doc_name == "test.md"
        assert ctx.kb_id == _TEST_KB_ID
        assert ctx.context_id == context_id

        # Step 3: IndexingExecutor 完整执行（使用真实 InMemoryChunkStore）
        executor = IndexingExecutor()
        postproc_task = await executor.run(ctx)

        assert isinstance(postproc_task, PostProcTask)
        assert postproc_task.doc_name == "test.md"
        assert postproc_task.kb_id == _TEST_KB_ID
        assert postproc_task.document_id == ctx.ext_metadata["document_id"]
        assert postproc_task.chunk_count > 0
        # 后处理默认值检查（图谱系出厂硬控为 true）
        assert postproc_task.postproc_config.enable_keyword_extract is True
        assert postproc_task.postproc_config.enable_graph_extract is True

        # Step 4: Queue2 收到 PostProcTask
        await postproc_queue.put(postproc_task)
        q2_task: PostProcTask = await postproc_queue.get()
        assert q2_task.document_id == ctx.ext_metadata.get("document_id", ctx.context_id)
        assert q2_task.chunk_count == postproc_task.chunk_count
    finally:
        for p in patches:
            p.stop()

    # ── 日志验证 ─────────────────────────────────────────────────
    log_text = caplog.text
    assert "索引构建启动" in log_text, f"Missing indexing start in log:\n{log_text}"
    assert "首批模块完成" in log_text, f"Missing first batch done in log:\n{log_text}"

    ctx_queue.task_done()
    postproc_queue.task_done()
