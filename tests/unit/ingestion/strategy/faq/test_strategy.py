"""FAQIngestionStrategy 测试（查重）。"""
from __future__ import annotations

import uuid
from unittest.mock import ANY, AsyncMock, patch

import pytest


class TestFAQIngestionStrategy:
    """测试 FAQIngestionStrategy 的 _pre_process 查重。"""

    @pytest.fixture
    def kb_id(self) -> str:
        return str(uuid.uuid4())

    @pytest.fixture
    def strategy(self):
        from aion_knowledge.ingestion.strategy.faq.strategy import (
            FAQIngestionStrategy,
        )
        return FAQIngestionStrategy(suffix="csv")

    @pytest.mark.asyncio
    async def test_pre_process_new_file_returns_none(self, strategy, kb_id):
        """新文件时 _pre_process 返回 None（继续 execute）。"""
        repo_mock = AsyncMock(return_value=None)
        with patch("aion_knowledge.ingestion.strategy.faq.strategy.get_session"), \
             patch("aion_knowledge.ingestion.strategy.faq.strategy.get_document_by_hash",
                   repo_mock):
            result = await strategy._pre_process(
                kb_id=kb_id, content=b"q,a\nQ1,A1", file_name="faq_import.csv"
            )
        assert result is None
        repo_mock.assert_awaited_once_with(ANY, kb_id, ANY)  # session + kb_id + file_hash

    @pytest.mark.asyncio
    async def test_pre_process_duplicate_returns_duplicate(self, strategy, kb_id):
        """重复文件时 _pre_process 返回 duplicate dict。"""
        existing_doc = AsyncMock()
        existing_doc.id = "00000000-0000-0000-0000-000000000099"

        repo_mock = AsyncMock(return_value=existing_doc)
        with patch("aion_knowledge.ingestion.strategy.faq.strategy.get_session"), \
             patch("aion_knowledge.ingestion.strategy.faq.strategy.get_document_by_hash",
                   repo_mock):
            result = await strategy._pre_process(
                kb_id=kb_id, content=b"q,a\nQ1,A1", file_name="faq_import.csv"
            )
        assert result is not None
        assert result["status"] == "duplicate"
        assert result["document_id"] == str(existing_doc.id)
        repo_mock.assert_awaited_once_with(ANY, kb_id, ANY)  # session + kb_id + file_hash

    @pytest.mark.asyncio
    async def test_register_source(self):
        """验证 FAQIngestionStrategy 已注册到 'faq' source。"""
        from aion_knowledge.ingestion.strategy.faq.strategy import (
            FAQIngestionStrategy,
        )
        from aion_knowledge.ingestion.strategy.registry import get_strategy

        strategy = get_strategy("faq", suffix="csv")
        assert isinstance(strategy, FAQIngestionStrategy)
