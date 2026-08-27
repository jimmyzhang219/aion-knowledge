"""RegularIngestionStrategy 查重测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestRegularDedup:
    """测试 RegularIngestionStrategy 的查重（_pre_process）。"""

    @pytest.mark.asyncio
    async def test_dedup_returns_duplicate(self):
        """重复文件时 _pre_process 返回 duplicate dict。"""
        from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy

        existing_doc = AsyncMock()
        existing_doc.id = "00000000-0000-0000-0000-000000000099"

        strategy = RegularIngestionStrategy(suffix="pdf")

        with patch("aion_knowledge.ingestion.strategy.regular.strategy.get_session"), \
             patch("aion_knowledge.ingestion.strategy.regular.strategy.get_document_by_hash",
                   AsyncMock(return_value=existing_doc)):

            result = await strategy._pre_process(
                kb_id="kb-1",
                content=b"duplicate content",
                file_name="dup.pdf",
            )

        assert result is not None
        assert result["status"] == "duplicate"
        assert result["document_id"] == str(existing_doc.id)

    @pytest.mark.asyncio
    async def test_dedup_new_file_returns_none(self):
        """新文件时 _pre_process 返回 None（继续 enqueue）。"""
        from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy

        strategy = RegularIngestionStrategy(suffix="pdf")

        with patch("aion_knowledge.ingestion.strategy.regular.strategy.get_session"), \
             patch("aion_knowledge.ingestion.strategy.regular.strategy.get_document_by_hash",
                   AsyncMock(return_value=None)):

            result = await strategy._pre_process(
                kb_id="kb-1",
                content=b"new content",
                file_name="new.pdf",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_register_regular_source(self):
        """验证 RegularIngestionStrategy 注册到 regular source。"""
        from aion_knowledge.ingestion.strategy.registry import get_strategy
        strategy = get_strategy("regular", suffix="pdf")
        from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy
        assert isinstance(strategy, RegularIngestionStrategy)
        assert strategy.suffix == "pdf"
        assert strategy.source == "regular"

    @pytest.mark.asyncio
    async def test_dedup_calls_repo_without_doc_name_size(self):
        """_pre_process 按新签名调用 get_document_by_hash（session + kb_id + file_hash 三参数）。"""
        from unittest.mock import AsyncMock, patch

        from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy

        strategy = RegularIngestionStrategy(suffix="pdf")
        repo_mock = AsyncMock(return_value=None)

        with patch("aion_knowledge.ingestion.strategy.regular.strategy.get_session"), \
             patch("aion_knowledge.ingestion.strategy.regular.strategy.get_document_by_hash",
                   repo_mock):
            await strategy._pre_process(
                kb_id="kb-1",
                content=b"some content",
                file_name="renamed.pdf",
            )

        repo_mock.assert_awaited_once()
        args = repo_mock.await_args.args
        assert len(args) == 3                       # session, kb_id, file_hash
        assert args[1] == "kb-1"
        assert args[2] == strategy._compute_hash(b"some content")
