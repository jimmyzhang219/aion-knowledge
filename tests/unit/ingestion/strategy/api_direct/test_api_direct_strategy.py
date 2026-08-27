"""ApiDirectIngestionStrategy 单元测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.infrastructure.models import UnifiedContext
from aion_knowledge.models.enums import StrategyName


class TestApiDirectStrategy:
    """ApiDirectIngestionStrategy 基础行为测试。"""

    @pytest.mark.asyncio
    async def test_source(self):
        """验证 source 属性为 api_direct。"""
        from aion_knowledge.ingestion.strategy.api_direct.strategy import (
            ApiDirectIngestionStrategy,
        )

        strategy = ApiDirectIngestionStrategy()
        assert strategy.source == StrategyName.api_direct.value

    @pytest.mark.asyncio
    async def test_suffix_default(self):
        """验证默认 suffix 为 raw。"""
        from aion_knowledge.ingestion.strategy.api_direct.strategy import (
            ApiDirectIngestionStrategy,
        )

        strategy = ApiDirectIngestionStrategy()
        assert strategy.suffix == "raw"

    @pytest.mark.asyncio
    async def test_suffix_custom(self):
        """验证可设置自定义 suffix。"""
        from aion_knowledge.ingestion.strategy.api_direct.strategy import (
            ApiDirectIngestionStrategy,
        )

        strategy = ApiDirectIngestionStrategy(suffix="json")
        assert strategy.suffix == "json"

    @pytest.mark.asyncio
    async def test_build_context_sets_content(self):
        """验证 _build_context 传入的 content 被设置到 UnifiedContext。"""
        from aion_knowledge.ingestion.strategy.api_direct.strategy import (
            ApiDirectIngestionStrategy,
        )

        strategy = ApiDirectIngestionStrategy()
        ctx = await strategy._build_context(
            doc_id="doc-1",
            doc_name="test.txt",
            kb_id="kb-1",
            s3_ref="s3://bucket/key",
            suffix="txt",
            content=b"hello world",
        )

        assert isinstance(ctx, UnifiedContext)
        assert ctx.source == "api_direct"
        assert ctx.kb_id == "kb-1"
        assert ctx.doc_name == "test.txt"
        assert ctx.suffix == "txt"
        assert ctx.original_file_ref == "s3://bucket/key"
        assert ctx.content == b"hello world"
        assert ctx.ext_metadata == {"document_id": "doc-1"}

    @pytest.mark.asyncio
    async def test_register_source(self):
        """验证 ApiDirectIngestionStrategy 已注册到 api_direct source。"""
        from aion_knowledge.ingestion.strategy.api_direct.strategy import (
            ApiDirectIngestionStrategy,
        )
        from aion_knowledge.ingestion.strategy.registry import get_strategy

        strategy = get_strategy("api_direct", suffix="txt")
        assert isinstance(strategy, ApiDirectIngestionStrategy)

    @pytest.mark.asyncio
    async def test_execute_returns_dict(self):
        """验证完整 execute 流程返回标准 dict。"""
        from aion_knowledge.infrastructure.queues import ctx_queue
        from aion_knowledge.ingestion.strategy.registry import get_strategy

        # 清空队列
        while not ctx_queue.empty():
            ctx_queue.get_nowait()
            ctx_queue.task_done()

        strategy = get_strategy("api_direct", suffix="txt")

        with (
            patch("aion_knowledge.ingestion.strategy.base.ensure_kb_exists",
                  new_callable=AsyncMock),
            patch("aion_knowledge.ingestion.strategy.base.get_session"),
            patch("aion_knowledge.ingestion.strategy.base.save_to_storage",
                  return_value="s3://bucket/docs/test/original.txt"),
            patch("aion_knowledge.ingestion.strategy.base.create_document",
                  new_callable=AsyncMock) as mock_create_doc,
        ):
            mock_doc = AsyncMock()
            mock_doc.id = "00000000-0000-0000-0000-000000000001"
            mock_create_doc.return_value = mock_doc

            result = await strategy.execute(
                kb_id="kb-1",
                content=b"test data",
                file_name="test.txt",
            )

        assert result["status"] == "queued"
        assert "context_id" in result
        assert result["document_id"] == "00000000-0000-0000-0000-000000000001"
