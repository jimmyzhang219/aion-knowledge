"""IndexingExecutor 测试。"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.indexing.executor import IndexingExecutor
from aion_knowledge.infrastructure.models import PostProcTask


class TestIndexingExecutor:
    """验证 IndexingExecutor 编排流程的正确性。"""

    @pytest.mark.asyncio
    async def test_run_returns_postproc_task(self, temp_raw_file):
        """验证 run() 返回 PostProcTask 实例。"""
        import uuid

        from aion_knowledge.infrastructure.models import PostProcTask, UnifiedContext

        ctx = UnifiedContext(
            context_id=str(uuid.uuid4()),
            source="regular",
            kb_id=str(uuid.uuid4()),
            doc_name="test.pdf",
            suffix="pdf",
            original_file_ref=temp_raw_file,
        )

        with patch("aion_knowledge.pipeline.postproc.dispatcher.PostProcDispatcher") as mock_disp:
            mock_disp.return_value.run_first_batch = AsyncMock()

            executor = IndexingExecutor()

            # Mock strategy 而不是 executor 的内部方法
            mock_strategy = MagicMock()
            mock_strategy.execute = AsyncMock(return_value=[{"seq_num": 1, "content": "test"}])
            executor._get_strategy = MagicMock(return_value=mock_strategy)

            result = await executor.run(ctx)
            assert isinstance(result, PostProcTask)
            assert result.doc_name == "test.pdf"
            assert result.chunk_count == 1

    @pytest.mark.asyncio
    async def test_run_strategy_error_raises(self):
        """验证策略 execute 出错会抛异常。"""
        import uuid

        from aion_knowledge.infrastructure.models import UnifiedContext

        ctx = UnifiedContext(
            context_id=str(uuid.uuid4()),
            source="regular",
            kb_id=str(uuid.uuid4()),
            doc_name="buggy.pdf",
            suffix="pdf",
            original_file_ref="/tmp/nonexistent.pdf",
        )

        executor = IndexingExecutor()
        mock_strategy = MagicMock()
        mock_strategy.execute = AsyncMock(side_effect=FileNotFoundError("file not found"))
        executor._get_strategy = MagicMock(return_value=mock_strategy)

        with pytest.raises(FileNotFoundError):
            await executor.run(ctx)

    @pytest.mark.asyncio
    async def test_run_chunker_empty(self):
        """验证 Chunker 返回空列表时 postproc first batch 应该被跳过。"""
        import uuid

        from aion_knowledge.infrastructure.models import UnifiedContext

        ctx = UnifiedContext(
            context_id=str(uuid.uuid4()),
            source="regular",
            kb_id=str(uuid.uuid4()),
            doc_name="empty.md",
            suffix="md",
            original_file_ref="/tmp/empty.md",
        )

        with patch("aion_knowledge.pipeline.postproc.dispatcher.PostProcDispatcher") as mock_disp:
            executor = IndexingExecutor()
            mock_strategy = MagicMock()
            mock_strategy.execute = AsyncMock(return_value=[])
            executor._get_strategy = MagicMock(return_value=mock_strategy)

            result = await executor.run(ctx)
            mock_disp.return_value.run_first_batch.assert_not_called()
            assert result.chunk_count == 0


@pytest.mark.asyncio
async def test_executor_faq_source_creates_chunks():
    """source='faq' 的 context 应使用 FAQChunkingStrategy。"""
    from aion_knowledge.indexing.executor import IndexingExecutor
    from aion_knowledge.infrastructure.models import UnifiedContext

    faq_csv = "/tmp/test_faq_strategy.csv"
    with open(faq_csv, "w", encoding="utf-8") as f:
        f.write("分类,问题,相似问题,负问题,答案,答案策略\n")
        f.write("技术支持,产品支持哪些语言？,,,中英文支持。,all\n")

    ctx = UnifiedContext(
        source="faq",
        kb_id="kb1",
        doc_name="faq_import_test",
        suffix="csv",
        original_file_ref=faq_csv,
        ext_metadata={"document_id": "doc1", "task_id": "task1"},
    )

    mock_session = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_session

    with (
        patch("aion_knowledge.infrastructure.db.get_session", return_value=mock_cm),
        patch("aion_knowledge.ingestion.document_repo.update_ingestion_task_status"),
        patch("aion_knowledge.ingestion.document_repo.update_document_status"),
        patch("aion_knowledge.pipeline.postproc.dispatcher.PostProcDispatcher") as mock_disp,
    ):
        mock_disp.return_value.run_first_batch = AsyncMock()
        executor = IndexingExecutor()
        result = await executor.run(ctx)

    try:
        os.remove(faq_csv)
    except OSError:
        pass

    assert isinstance(result, PostProcTask)
    assert result.doc_name == "faq_import_test"
    assert result.chunk_count == 1
    assert result.suffix == "csv"
