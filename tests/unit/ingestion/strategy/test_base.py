"""IngestionStrategy 基类测试。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aion_knowledge.ingestion.strategy.base import IngestionStrategy


class _ConcreteStrategy(IngestionStrategy):
    """测试用具体子类。"""
    source = "test"
    suffix = "test"

    async def _build_context(self, *args, **kwargs):
        return None


class TestS3Key:
    """验证 _build_s3_key() 使用 kb_id + hash 目录布局并保留原始文件名。"""

    def test_regular_file(self):
        s = _ConcreteStrategy()
        assert s._build_s3_key("report.pdf", kb_id="kb-1", file_hash="h1") == "kb-1/h1/report.pdf"

    def test_csv_file(self):
        s = _ConcreteStrategy()
        assert s._build_s3_key("faq_data.csv", kb_id="kb-1", file_hash="h1") == "kb-1/h1/faq_data.csv"

    def test_no_extension(self):
        s = _ConcreteStrategy()
        assert s._build_s3_key("Makefile", kb_id="kb-1", file_hash="h1") == "kb-1/h1/Makefile"

    def test_multiple_dots(self):
        s = _ConcreteStrategy()
        assert s._build_s3_key("archive.tar.gz", kb_id="kb-1", file_hash="h1") == "kb-1/h1/archive.tar.gz"


class TestSource:
    """验证 source 属性。"""

    def test_default_source(self):
        s = _ConcreteStrategy()
        assert s.source == "test"


class TestSubSteps:
    """验证子步骤默认行为及可覆盖性。"""

    def test_compute_hash_default(self):
        s = _ConcreteStrategy()
        h = s._compute_hash(b"hello")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex

    def test_pre_process_default(self):
        """默认 _pre_process 返回 None（不短路）。"""
        s = _ConcreteStrategy()
        import asyncio
        result = asyncio.run(s._pre_process(kb_id="x", content=b"x", file_name="x"))
        assert result is None

    def test_enqueue_context_default_queues(self):
        """_enqueue_context 默认调用 ctx_queue.put。"""
        import asyncio

        from aion_knowledge.infrastructure.models import UnifiedContext
        from aion_knowledge.infrastructure.queues import ctx_queue

        # 清空队列
        while not ctx_queue.empty():
            ctx_queue.get_nowait()
            ctx_queue.task_done()

        s = _ConcreteStrategy()
        ctx = UnifiedContext(source="test", kb_id="x", doc_name="x", suffix="x", original_file_ref="x")
        asyncio.run(s._enqueue_context(ctx))

        assert not ctx_queue.empty()
        retrieved = asyncio.run(ctx_queue.get())
        assert retrieved.source == "test"
        ctx_queue.task_done()


class TestExecuteFileDir:
    """验证 execute() 从实际 s3_key 派生 ctx.file_dir。"""

    @pytest.mark.asyncio
    async def test_execute_sets_file_dir(self):
        from types import SimpleNamespace

        from aion_knowledge.infrastructure.models import UnifiedContext

        s = _ConcreteStrategy()
        s._ensure_kb_exists = AsyncMock()
        s._compute_hash = MagicMock(return_value="h1")  # hash 算法本身有专项测试，此处钉住目录布局
        s._save_to_storage = AsyncMock(return_value="s3://bucket/docs/kb-1/h1/report.pdf")
        s._create_document_record = AsyncMock(return_value=SimpleNamespace(id="doc-1"))
        ctx = UnifiedContext(source="test", kb_id="kb-1", doc_name="report.pdf", suffix="pdf",
                             original_file_ref="x")
        s._build_context = AsyncMock(return_value=ctx)
        s._enqueue_context = AsyncMock()

        result = await s.execute(kb_id="kb-1", content=b"x", file_name="report.pdf")

        assert result["status"] == "queued"
        assert ctx.file_dir == "kb-1/h1"
        assert ctx.original_file_ref == "x"  # 模板不覆盖 _build_context 已填的值

    @pytest.mark.asyncio
    async def test_execute_ensures_kb_exists(self):
        """execute() 最前应先校验知识库存在，校验失败即短路。"""
        from types import SimpleNamespace

        from aion_knowledge.infrastructure.models import UnifiedContext
        from aion_knowledge.ingestion.kb_guard import KnowledgeBaseNotFoundError

        s = _ConcreteStrategy()
        s._ensure_kb_exists = AsyncMock(
            side_effect=KnowledgeBaseNotFoundError("kb-1 not found"),
        )
        s._compute_hash = MagicMock(return_value="h1")
        s._save_to_storage = AsyncMock(return_value="s3://bucket/docs/kb-1/h1/report.pdf")
        s._create_document_record = AsyncMock(return_value=SimpleNamespace(id="doc-1"))
        ctx = UnifiedContext(source="test", kb_id="kb-1", doc_name="report.pdf", suffix="pdf",
                             original_file_ref="x")
        s._build_context = AsyncMock(return_value=ctx)
        s._enqueue_context = AsyncMock()

        with pytest.raises(KnowledgeBaseNotFoundError):
            await s.execute(kb_id="kb-1", content=b"x", file_name="report.pdf")

        s._ensure_kb_exists.assert_awaited_once_with("kb-1")
        s._create_document_record.assert_not_awaited()
        s._enqueue_context.assert_not_awaited()
