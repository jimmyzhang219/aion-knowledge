"""Tests for RegularIngestionStrategy."""

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy


def test_source_default():
    """默认 source 为 regular。"""
    strategy = RegularIngestionStrategy(suffix="pdf")
    assert strategy.source == "regular"
    assert strategy.suffix == "pdf"


def test_source_custom():
    """支持自定义 source（如 url_import）。"""
    strategy = RegularIngestionStrategy(suffix="url", source="url_import")
    assert strategy.source == "url_import"
    assert strategy.suffix == "url"


@pytest.mark.asyncio
async def test_enqueue_returns_dict():
    """enqueue 调用返回标准结构。"""
    from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy

    mock_doc = AsyncMock()
    mock_doc.id = "00000000-0000-0000-0000-000000000099"
    mock_task = AsyncMock()
    mock_task.id = "00000000-0000-0000-0000-000000000098"

    mock_ctx_queue = AsyncMock()
    mock_ctx_queue.put = AsyncMock()

    patches = [
        patch("aion_knowledge.ingestion.strategy.base.ensure_kb_exists", AsyncMock()),
        patch("aion_knowledge.ingestion.strategy.base.save_to_storage", AsyncMock(return_value="/tmp/aion_storage/test.pdf")),
        patch("aion_knowledge.ingestion.strategy.base.extract_storage_key", return_value="test.pdf"),
        patch("aion_knowledge.ingestion.strategy.base.hash_file", return_value="abc123"),
        patch("aion_knowledge.ingestion.strategy.base.create_document", AsyncMock(return_value=mock_doc)),
        patch("aion_knowledge.ingestion.strategy.base.ctx_queue", mock_ctx_queue),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.get_session"),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.get_document_by_hash", AsyncMock(return_value=None)),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.create_ingestion_task", AsyncMock(return_value=mock_task)),
    ]

    for p in patches:
        p.start()

    try:
        strategy = RegularIngestionStrategy(suffix="pdf")
        result = await strategy.execute(
            kb_id="kb-1",
            content=b"test",
            file_name="test.pdf",
            creator="test",
        )

        assert result["status"] == "queued"
        assert "context_id" in result
        assert result["document_id"] == str(mock_doc.id)
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_execute_passes_trace_id_to_create_document():
    """_create_document_record 将请求 trace_id 传入 create_document。"""
    from aion_knowledge.common.trace import reset_trace_id, set_trace_id
    from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy

    mock_doc = AsyncMock()
    mock_doc.id = "00000000-0000-0000-0000-000000000099"
    mock_task = AsyncMock()
    mock_task.id = "00000000-0000-0000-0000-000000000098"

    mock_ctx_queue = AsyncMock()
    mock_ctx_queue.put = AsyncMock()

    create_doc_mock = AsyncMock(return_value=mock_doc)

    patches = [
        patch("aion_knowledge.ingestion.strategy.base.ensure_kb_exists", AsyncMock()),
        patch("aion_knowledge.ingestion.strategy.base.save_to_storage", AsyncMock(return_value="/tmp/aion_storage/test.pdf")),
        patch("aion_knowledge.ingestion.strategy.base.extract_storage_key", return_value="test.pdf"),
        patch("aion_knowledge.ingestion.strategy.base.hash_file", return_value="abc123"),
        patch("aion_knowledge.ingestion.strategy.base.create_document", create_doc_mock),
        patch("aion_knowledge.ingestion.strategy.base.ctx_queue", mock_ctx_queue),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.get_session"),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.get_document_by_hash", AsyncMock(return_value=None)),
        patch("aion_knowledge.ingestion.strategy.regular.strategy.create_ingestion_task", AsyncMock(return_value=mock_task)),
    ]

    token = set_trace_id("strat-trace-1")
    try:
        for p in patches:
            p.start()
        try:
            strategy = RegularIngestionStrategy(suffix="pdf")
            await strategy.execute(
                kb_id="kb-1",
                content=b"test",
                file_name="test.pdf",
                creator="test",
            )
        finally:
            for p in patches:
                p.stop()
    finally:
        reset_trace_id(token)

    create_doc_mock.assert_called_once()
    assert create_doc_mock.call_args.kwargs["trace_id"] == "strat-trace-1"
